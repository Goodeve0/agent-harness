"""
Claude Code / Codex CLI 双模型交叉评测 MVP（metrics/cli_judge.py）

背景：
  原 cross-check 两侧都是 OpenAI client（gpt-4o-mini × gpt-4o），与简历声称的
  "Claude Code + Codex 双模型交叉校验" 不符且从未实测。本模块补齐真实异构后端。

设计：
  · CLIJudge 通过子进程调用真实 CLI：`claude -p "<prompt>"` / `codex exec "<prompt>"`
  · 与 LLMJudge 实现同一接口（judge() -> float），可直接注入 CrossValidator，
    两侧后端可自由组合（openai × claude / claude × codex / ...）
  · 复用 judge.py 的 NUMERIC_PROMPT / PATTERN_PROMPT：两套后端共享同一套
    Rubric 提示词，对比的是"推理后端"而非"提示词差异"，保证公平
  · fail-open：CLI 缺失 / 超时 / 非零退出 / 输出无法解析 → 返回 -1.0，
    由 CrossValidator 标记为无效侧（单侧无效只采信有效侧，双侧无效记 0）。
    绝不伪造分数 —— 评测可信度是底线

用法（CLI 直连，需先安装 claude / codex 并配置 ANTHROPIC_API_KEY / OPENAI_API_KEY）：
  python run_eval.py --task tasks/content_pipeline/task.yaml --cross-check \
      --judge-a-backend claude --judge-b-backend codex
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess

from metrics.judge import NUMERIC_PROMPT, PATTERN_PROMPT

# ─────────────────────────────────────────────────────────────────────────────
#  CLI 后端定义：命令名 + 固定参数前缀
# ─────────────────────────────────────────────────────────────────────────────

BACKEND_CMDS = {
    "claude": {"bin": "claude", "args": ["-p"]},     # claude -p "<prompt>"
    "codex": {"bin": "codex", "args": ["exec"]},     # codex exec "<prompt>"
}


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def parse_cli_score(raw: str) -> float:
    """
    解析 CLI stdout 中的评分，返回 0~1 浮点数；无法解析返回 -1.0。

    解析顺序（宽松到严格兜底）：
      1. JSON：{"score": 0.85} / {"score": "0.85"}（兼容 claude --output-format json）
      2. 分数表达式：8/10 → 0.8
      3. 百分比：85% → 0.85
      4. 0~1 连续数字（从后往前取，评分通常在末尾）
    """
    if not raw or not raw.strip():
        return -1.0

    # 1) JSON 优先
    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            s = obj.get("score", obj.get("Score"))
            if s is not None:
                return _clamp(float(s))
        except (ValueError, TypeError):
            pass  # JSON 结构不含 score，继续宽松解析

    # 2) 分数表达式
    m = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", raw)
    if m:
        denom = float(m.group(2))
        if denom > 0:
            return _clamp(float(m.group(1)) / denom)

    # 3) 百分比
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", raw)
    if m:
        return _clamp(float(m.group(1)) / 100.0)

    # 4) 0~1 数字兜底（从后往前，评分通常在末尾）
    for tok in reversed(re.findall(r"\d+(?:\.\d+)?", raw)):
        v = float(tok)
        if 0.0 <= v <= 1.0:
            return v
    return -1.0


def probe_cli(backend: str) -> tuple[bool, str]:
    """
    探测后端 CLI 是否可用。返回 (available, info)。

    available=False 时 judge() 会以 -1 进入仲裁（fail-open），
    不会崩掉主流程 —— 启动时打印此结果便于快速定位。
    """
    spec = BACKEND_CMDS[backend]
    bin_name = spec["bin"]
    path = shutil.which(bin_name)
    if not path:
        hint = {
            "claude": "安装: npm i -g @anthropic-ai/claude-code，并配置 ANTHROPIC_API_KEY",
            "codex": "安装: npm i -g @openai/codex，并配置 OPENAI_API_KEY",
        }[backend]
        return False, f"未找到 {bin_name} CLI（{hint}）"
    try:
        r = subprocess.run([bin_name, "--version"], capture_output=True,
                           text=True, timeout=10)
        ver = (r.stdout or r.stderr or "").strip().splitlines()[0][:80]
        return True, f"{bin_name}@{ver}"
    except Exception:
        return True, f"{bin_name}（版本探测失败，不影响调用）"


# ─────────────────────────────────────────────────────────────────────────────
#  CLIJudge：与 LLMJudge 同接口的 CLI 后端 judge
# ─────────────────────────────────────────────────────────────────────────────

class CLIJudge:
    """
    CLI 后端 LLM Judge，接口对齐 LLMJudge：

        judge(text, rubric, reference="") -> float   # 0~1；异常/不可用 -1.0
        judge_model                                  # 供 CrossValidator 取模型名

    Args:
        backend: "claude" | "codex"
        mode:    "numeric"（0~1 连续分） | "pattern"（YES/NO → 1.0/0.0）
        timeout: 子进程超时秒数（默认 180s，LLM 推理耗时长）
    """

    def __init__(self, backend: str = "claude", mode: str = "numeric",
                 prompt_template: str | None = None, timeout: int = 180):
        assert backend in BACKEND_CMDS, f"backend must be in {list(BACKEND_CMDS)}"
        assert mode in ("numeric", "pattern"), "mode must be 'numeric' or 'pattern'"
        self.backend = backend
        self.mode = mode
        self.timeout = timeout
        self.prompt_template = prompt_template or (
            NUMERIC_PROMPT if mode == "numeric" else PATTERN_PROMPT
        )
        # 与 CrossValidator 的 getattr(judge, "judge_model") 对齐
        self.judge_model = f"{backend}-cli"
        self.last_error: str | None = None

    # ── 可用性探测（主流程启动时打印，不阻塞） ──────────────────────────────
    def available(self) -> tuple[bool, str]:
        return probe_cli(self.backend)

    # ── 核心评分 ──────────────────────────────────────────────────────────────
    def judge(self, text: str, rubric: str, reference: str = "") -> float:
        prompt = self.prompt_template.format(
            rubric=rubric, text=text, reference=reference
        )
        spec = BACKEND_CMDS[self.backend]

        try:
            proc = subprocess.run(
                [spec["bin"], *spec["args"], prompt],
                capture_output=True, text=True, timeout=self.timeout,
            )
        except FileNotFoundError:
            self.last_error = f"{spec['bin']} CLI 未安装"
            return -1.0
        except subprocess.TimeoutExpired:
            self.last_error = f"{spec['bin']} 调用超时（>{self.timeout}s）"
            return -1.0

        if proc.returncode != 0:
            err = (proc.stderr or "").strip().replace("\n", " ")[:200]
            self.last_error = f"{spec['bin']} 退出码 {proc.returncode}: {err}"
            return -1.0

        raw = (proc.stdout or "").strip()

        # pattern 模式：YES/NO 判定
        if self.mode == "pattern":
            upper = raw.upper()
            if "YES" in upper:
                return 1.0
            if "NO" in upper:
                return 0.0
            self.last_error = f"{spec['bin']} 输出无 YES/NO: {raw[:120]!r}"
            return -1.0

        score = parse_cli_score(raw)
        if score < 0:
            self.last_error = f"{spec['bin']} 输出无法解析: {raw[:120]!r}"
        return score
