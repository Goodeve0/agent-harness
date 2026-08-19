"""
评测编排层：trial 执行 / 回归合并 / Judge 工厂 / CI 口径

职责边界（run_eval.py 拆分而来）：
  - CLI（参数解析 / 终端呈现 / 主流程编排）  → run_eval.py
  - Mock Agent（无 Key 链路自检替身）        → harness/mock_agent.py
  - 编排逻辑（单次 trial 执行、Golden 回归合并、
    跨后端 Judge 工厂、CI 指标口径）         → 本模块
"""
from __future__ import annotations

import json
import os

from rich.console import Console

from dataset.tracer import GoldenDataset
from harness.agent_loop import AgentLoop
from harness.mock_agent import mock_agent_run
from harness.sandbox import MockSandbox
from metrics.cli_judge import CLIJudge
from metrics.cross_validator import CrossValidator
from metrics.hybrid_scorer import HybridScorer
from metrics.judge import LLMJudge

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
#  单次 trial
# ─────────────────────────────────────────────────────────────────────────────

def run_trial(spec: dict, sample: dict, trial_id: int, scorer: HybridScorer,
              cross_validator: CrossValidator | None, mock_run: bool):
    """执行一次 trial：跑 Agent → 记录 Trace → 双层评分"""
    sandbox = MockSandbox(
        mock_apis=spec.get("mock_apis", {}),
        task_id=spec["task_id"],            # 干净的 task_id，不再拼接 sample_id
        sample_id=sample["sample_id"],
        agent_id=spec["agent_id"],
        prompt_version=spec.get("prompt_version", "v1"),
        tool_definitions=spec.get("tools", []),
    )

    # ── 驱动被测 Agent ────────────────────────────────────────────────────────
    if mock_run:
        trace = mock_agent_run(spec, sample, sandbox, trial_id)
    else:
        loop = AgentLoop(
            model=os.getenv("EVAL_MODEL", "gpt-4o-mini"),
            tools=spec.get("tools", []),
            strategy=spec.get("strategy", "function_calling"),
            max_steps=spec.get("max_steps", 10),
        )
        prompt = (f"{spec.get('agent_prompt', '')}\n\n"
                  f"当前任务输入：{json.dumps(sample['input'], ensure_ascii=False)}")
        trace = loop.run(prompt, sandbox)

    # ── 双层评分 ──────────────────────────────────────────────────────────────
    gt = dict(sample.get("ground_truth", {}))
    gt.setdefault("input_content", sample.get("input", {}).get("content", ""))

    result = scorer.score(trace, ground_truth=gt,
                          expected_behavior=sample.get("expected_behavior", ""))
    res_dict = result.to_dict()
    res_dict["sample_id"] = sample["sample_id"]
    res_dict["trial_id"] = trial_id

    # ── 双模型交叉校验（可选）：分歧取保守分，仲裁结果回写 judge 分与总分 ──────
    if cross_validator and not mock_run:
        rec = cross_validator.validate(
            sample_id=f"{sample['sample_id']}_t{trial_id}",
            text=trace.final_output,
            rubric=spec.get("scoring", {}).get("judge_rubric", ""),
            reference=str(gt),
        )
        cross = rec.to_dict()
        # merged_score：一致取均值、分歧取保守（较低分），避免高估 Agent 能力
        if scorer.w_judge > 0:
            merged = rec.merged_score
            rule_score = res_dict.get("rule_score", 0.0)
            new_final = scorer.w_rule * rule_score + scorer.w_judge * merged
            res_dict["judge_score"] = round(merged, 4)
            res_dict["overall_score"] = round(new_final, 4)
            res_dict["judge_detail"] = {
                "mode": "cross_check",
                "score": round(merged, 4),
                "disputed": not rec.agreed,
            }
            if new_final < scorer.pass_threshold and not res_dict.get("failure_reason"):
                res_dict["failure_reason"] = "LLM_JUDGE_FAIL"
        res_dict["cross_check"] = cross

    trace.eval_result = res_dict
    trace.failure_reason = res_dict.get("failure_reason")

    # ── 终端输出 ──────────────────────────────────────────────────────────────
    score = res_dict["overall_score"]
    mark = "[green]✅[/]" if score >= scorer.pass_threshold else "[red]❌[/]"
    reason = res_dict.get("failure_reason") or "ok"
    console.print(
        f"    {mark} trial{trial_id}  final={score:.3f}  "
        f"[dim](rule={res_dict['rule_score']:.2f} judge={res_dict['judge_score']:.2f})[/]  "
        f"[yellow]{reason}[/]"
    )
    if res_dict.get("rule_detail", {}).get("failed"):
        for f in res_dict["rule_detail"]["failed"][:2]:
            console.print(f"        [dim]↳ {f['name']}: {f['detail']}[/]")

    return trace, res_dict


# ─────────────────────────────────────────────────────────────────────────────
#  Judge 工厂 / CI 口径 / 回归合并
# ─────────────────────────────────────────────────────────────────────────────

def build_judge(backend: str, model: str | None, mode: str = "numeric"):
    """
    按后端构造 judge：
      openai → LLMJudge（OpenAI API 调用，model 生效）
      claude / codex → CLIJudge（CLI 子进程调用，model 由 CLI 自身决定）
    """
    if backend == "openai":
        return LLMJudge(mode=mode, judge_model=model)
    return CLIJudge(backend=backend, mode=mode)


def select_ci_metric(aggregation: dict, reporter_data: dict, metric: str) -> tuple[float, str]:
    """统一 CI 口径：Pass^k 衡量稳定性，Pass@k 衡量能力上界。"""
    metric_map = {
        "pass_hat_k": ("mean_pass^k", "样本级 Pass^k"),
        "pass_at_k": ("mean_pass@k", "样本级 Pass@k"),
        "vote_at_k": ("mean_vote@k", "样本级 Vote@k"),
    }
    key, label = metric_map[metric]
    if aggregation.get("total_samples"):
        return aggregation.get(key, 0.0), f"{label}={aggregation.get(key, 0.0):.2f}"
    value = reporter_data.get("pass_rate", 0.0)
    return value, f"trial 级 pass_rate={value:.2f}"


def load_regression_samples(golden: GoldenDataset, task_id: str,
                            task_samples: list[dict]) -> list[dict]:
    """将未修复 Golden Case 覆盖合并进本任务，保证下轮评测真实回放。"""
    merged = {sample["sample_id"]: dict(sample) for sample in task_samples}
    for case in golden.regression_cases():
        if case.get("task_id") != task_id or not case.get("sample_id"):
            continue
        sample_id = case["sample_id"]
        # Golden 快照优先：即使 YAML 后续删改了原样本，也不会丢失历史坏例。
        merged[sample_id] = {
            "sample_id": sample_id,
            "difficulty": "regression",
            "input": case.get("input", {}),
            "ground_truth": case.get("ground_truth", {}),
            "expected_behavior": case.get("expected_behavior", ""),
            "regression_case_id": case.get("case_id"),
        }
    return list(merged.values())
