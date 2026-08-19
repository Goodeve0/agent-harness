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
from harness.sandbox import AgentTrace, MockSandbox
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
    """执行一次 trial：跑单个 Agent → 记录 Trace → 双层评分"""
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

    res_dict = _score_trace(trace, sample, trial_id, spec, scorer, cross_validator)
    return trace, res_dict


def run_chain(spec: dict, sample: dict, trial_id: int, scorer: HybridScorer,
              cross_validator: CrossValidator | None, mock_run: bool):
    """顺序执行多 Agent 协作链路：上一环最终输出作为下一环的输入上下文。

    spec 形态：
      agents:
        - agent_id / agent_prompt / strategy / max_steps / tools / mock_apis
      chain_id / task_id / rubric / samples / scoring 在顶层，整链共享。

    链路 trace 合并为一条（steps/messages 拼接、最终输出取最后一环），规则层
    对整条链路轨迹评分——rubric 无需为多 Agent 改造，tool_called / tool_sequence
    等 check 天然看到全链路。mock 模式同样可用：每个 Agent 各自走 mock 注入。
    """
    agents = spec.get("agents", [])
    if not agents:
        raise ValueError("run_chain 需要 spec['agents'] 非空（否则请用 run_trial）")
    chain_id = spec.get("chain_id") or agents[0]["agent_id"]
    version = spec.get("prompt_version", "v1")

    hop_traces: list[Any] = []
    upstream = dict(sample.get("input", {}))
    for agent_cfg in agents:
        sandbox = MockSandbox(
            mock_apis=agent_cfg.get("mock_apis", {}),
            task_id=spec["task_id"],
            sample_id=sample["sample_id"],
            agent_id=agent_cfg["agent_id"],
            prompt_version=version,
            tool_definitions=agent_cfg.get("tools", []),
        )
        # 每个 Agent 用自己的 tools/mock_apis/prompt/max_steps，共享任务 rubric
        agent_spec = {
            **spec,
            "agent_id": agent_cfg["agent_id"],
            "agent_prompt": agent_cfg.get("agent_prompt", ""),
            "tools": agent_cfg.get("tools", []),
            "mock_apis": agent_cfg.get("mock_apis", {}),
            "max_steps": agent_cfg.get("max_steps", spec.get("max_steps", 10)),
        }
        if mock_run:
            trace = mock_agent_run(agent_spec, sample, sandbox, trial_id)
        else:
            loop = AgentLoop(
                model=os.getenv("EVAL_MODEL", "gpt-4o-mini"),
                tools=agent_cfg.get("tools", []),
                strategy=agent_cfg.get("strategy", "function_calling"),
                max_steps=agent_cfg.get("max_steps", spec.get("max_steps", 10)),
            )
            prompt = (f"{agent_cfg.get('agent_prompt', '')}\n\n"
                      f"当前任务输入：{json.dumps(sample['input'], ensure_ascii=False)}\n"
                      f"上游 Agent 输出：{json.dumps(upstream, ensure_ascii=False)}")
            trace = loop.run(prompt, sandbox)
        hop_traces.append(trace)
        # 下游上下文 = 原始输入 + 上游最终输出
        upstream = {**sample.get("input", {}), "upstream_output": trace.final_output}

    # ── 合并链路 trace：整条链的轨迹交给规则层统一评分 ────────────────────────
    merged = AgentTrace(
        task_id=spec["task_id"],
        sample_id=sample["sample_id"],
        agent_id=chain_id,
        prompt_version=version,
    )
    for tr in hop_traces:
        merged.steps.extend(tr.steps)
        merged.messages.extend(tr.messages)
    merged.final_output = hop_traces[-1].final_output
    merged.total_tokens = sum(t.total_tokens for t in hop_traces)
    merged.total_latency_ms = sum(t.total_latency_ms for t in hop_traces)
    merged.success = hop_traces[-1].success
    merged.failure_reason = hop_traces[-1].failure_reason

    res_dict = _score_trace(merged, sample, trial_id, spec, scorer, cross_validator)
    return merged, res_dict


def _score_trace(trace: AgentTrace, sample: dict, trial_id: int, spec: dict,
                 scorer: HybridScorer, cross_validator: CrossValidator | None) -> dict:
    """双层评分 + 可选交叉校验 + 终端输出（run_trial / run_chain 共用，保证口径一致）"""
    gt = dict(sample.get("ground_truth", {}))
    gt.setdefault("input_content", sample.get("input", {}).get("content", ""))

    result = scorer.score(trace, ground_truth=gt,
                          expected_behavior=sample.get("expected_behavior", ""))
    res_dict = result.to_dict()
    res_dict["sample_id"] = sample["sample_id"]
    res_dict["trial_id"] = trial_id

    # ── 双模型交叉校验（可选）：分歧取保守分，仲裁结果回写 judge 分与总分 ──────
    if cross_validator:
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

    return res_dict


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
