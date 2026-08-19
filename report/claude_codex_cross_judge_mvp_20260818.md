# Claude Code + Codex 双模型交叉评测 MVP

日期：2026-08-18
状态：✅ 代码落地 + 69 测试全绿；⚠️ 真实 CLI 评分需配 API Key 后实测（本机 codex 已装、claude 未装）

---

## 1. 背景

核验发现原 `--cross-check` 两侧 judge 都是 OpenAI client（`gpt-4o-mini × gpt-4o`），与简历"Claude Code + Codex 双模型交叉校验"不符且从未实测。本 MVP 补齐真实异构后端。

## 2. 设计

```
run_eval.py --cross-check --judge-a-backend claude --judge-b-backend codex
        │
        ▼
CrossValidator(judge_a=CLIJudge("claude"), judge_b=CLIJudge("codex"))
        │                 │                    │
        │                 ▼                    ▼
        │         claude -p "<prompt>"   codex exec "<prompt>"
        │                 │                    │
        │                 └──── 同一套 NUMERIC_PROMPT / PATTERN_PROMPT ────┘
        │                            （只比推理后端，不比提示词）
        ▼
  仲裁：一致取均值 / 分歧取保守 / 单侧异常采信有效侧 / 双侧异常 0
```

核心原则：

- **接口对齐**：`CLIJudge` 与 `LLMJudge` 实现同一 `judge(text, rubric, reference) -> float` 接口 + `judge_model` 属性，可自由混搭（openai×claude / claude×codex / codex×openai），注入 `CrossValidator` 零改动。
- **共享 Prompt**：复用 `metrics/judge.py` 的 NUMERIC_PROMPT / PATTERN_PROMPT。两套后端面对完全相同的评分标准，对比的是"推理后端能力"而非"提示词差异"。
- **fail-open（不造假底线）**：CLI 缺失 / 超时 / 非零退出 / 输出无法解析 → 返回 `-1.0`，由 `CrossValidator` 标记为无效侧。单侧无效只采信有效侧，双侧无效记 0，无效记录不进一致率统计。**绝不伪造分数**——评测可信度优先于"跑出好看结果"。
- **启动探测**：`probe_cli()` 在 run_eval 启动时打印两侧 CLI 可用性与版本，一眼定位配置问题。

## 3. 改动清单

| 文件 | 改动 |
|---|---|
| `metrics/cli_judge.py` | **新增**。CLIJudge（claude/codex 两种后端）+ `parse_cli_score`（JSON→分数→百分比→0-1 数字四级解析）+ `probe_cli` 探测 |
| `metrics/cross_validator.py` | 修复 `_safe_judge`：judge 返回 -1（无法判定/CLI 不可用）不再被转成 0 分，保留 -1 语义进仲裁（与注释、与 `CrossCheckRecord.valid` 判定对齐） |
| `run_eval.py` | 新增 `--judge-a-backend/--judge-b-backend`（openai/claude/codex 三选一，默认 openai 保持兼容）+ `_build_judge` 按后端构造 + 启动时 CLI 探测打印 |
| `tests/test_cli_judge.py` | **新增** 19 个测试：命令构造 / 四级解析 / pattern YES-NO / fail-open 四场景 / probe / CrossValidator 集成仲裁三场景 |

## 4. 使用方式

```bash
# 0) 安装 CLI（二选一或都装）
npm i -g @anthropic-ai/claude-code   # 需 ANTHROPIC_API_KEY
npm i -g @openai/codex               # 需 OPENAI_API_KEY（本机已装 0.147.0）

# 1) 双 CLI 交叉评测（Claude Code × Codex）
python run_eval.py --task tasks/content_pipeline/task.yaml --cross-check \
    --judge-a-backend claude --judge-b-backend codex

# 2) CLI × API 混搭（Claude Code × GPT-4o）
python run_eval.py --task tasks/customer_service/task.yaml --cross-check \
    --judge-a-backend claude --judge-b-backend openai --judge-b gpt-4o

# 3) 纯 CLI 探测（不跑评测，只看两侧 CLI 是否可用）
python -c "
import sys; sys.path.insert(0, '.')
from metrics.cli_judge import probe_cli
for b in ('claude', 'codex'):
    ok, info = probe_cli(b)
    print(f'{b}:', 'OK' if ok else 'MISSING', info)
"
```

## 5. 本机探测结果（2026-08-18）

| 后端 | 状态 | 说明 |
|---|---|---|
| `codex` | ✅ 可用 | `codex@codex-cli 0.147.0`（/opt/homebrew/bin/codex），命令签名 `codex exec <PROMPT>` 已核实 |
| `claude` | ⚠️ 未安装 | `npm i -g @anthropic-ai/claude-code` 后配置 `ANTHROPIC_API_KEY` |

未安装的一侧在评测中会以 `-1` 进入仲裁（fail-open），报告仍会产出有效侧的分数与一致率统计——这正是设计的一部分。

## 6. 评测产物

`--cross-check` 开启时每样本生成 `cross_check` 字段，结尾打印交叉校验报告：

```
Judge 交叉校验：
  一致率: 85.0%  平均分差: 0.083  争议样本: 2
  评测标准基本可靠，建议复核争议样本
```

`CrossValidator.report()` 完整结构：`agreement_rate` / `mean_abs_diff` / `mean_score_a` / `mean_score_b` / `disputed_samples`（直接产出人工复核队列）/ `verdict`。

## 7. 面试表述（收紧版，已与实现对齐）

> 落地双模型交叉校验机制，judge 后端可配置（OpenAI API / Claude Code CLI / Codex CLI 三选一，支持任意异构组合），两模型共享同一套 Rubric 提示词独立评分；一致率高采信均值、分歧取保守分、单侧异常降级采信有效侧（不伪造分数）；一致率本身作为评测系统可信度指标，分歧样本自动进入人工复核队列，配合 Trace 落盘定位工具调用偏差。

**诚实边界**：可答"CLI 后端接入与仲裁降级已落地并单测覆盖（69 个测试）；真实模型评分需配 API Key，本机已具备 Codex CLI 环境"。不要声称"跑过真实数据对比出周级→小时级"——那段宣传语没有数据支撑，已被从简历表述中移除。

## 8. 遗留事项

- [ ] 安装 claude CLI 并配 ANTHROPIC_API_KEY 后，跑一次 `--judge-a-backend claude --judge-b-backend codex` 真实评测，留一份报告 JSON 作为证据（codex 侧本机可直接跑）。
- [ ] "归因交付周期周级→小时级"如要坚持写进简历，需补充 baseline 对照计时数据（当前无任何支撑，已建议删除）。
