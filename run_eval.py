#!/usr/bin/env python3
"""
AgentHarness 主入口 —— 多 Agent 协作链路评测平台

完整评测闭环：
  Task Spec (YAML)
      ↓
  MockSandbox 拦截工具调用 + AgentLoop 驱动被测 Agent → Trace
      ↓
  HybridScorer 双层评分（规则层确定性 + Judge 层语义，加权合并）
      ↓
  [可选] CrossValidator 双模型交叉校验 → Judge 一致率
      ↓
  Pass@k / Vote@k / Pass^k 聚合
      ↓
  Bad Case 自动归因打标 → Golden Dataset 落盘（回归集）
      ↓
  报告 + 跨版本 Diff + CI Gate

用法：
  # 基础评测（规则层，零 LLM 成本）
  python run_eval.py --task tasks/content_pipeline/task.yaml --mock-run

  # 真实评测，每样本 3 次 trial 算 Pass^3
  python run_eval.py --task tasks/content_pipeline/task.yaml --runs 3

  # 双模型交叉校验
  python run_eval.py --task tasks/content_pipeline/task.yaml --cross-check \
      --judge-a gpt-4o-mini --judge-b claude-3-5-sonnet-20241022

  # 跨版本对比
  python run_eval.py --task tasks/content_pipeline/task.yaml \
      --diff --baseline report/output/xxx_v1.json

  # CI 门禁
  python run_eval.py --task tasks/content_pipeline/task.yaml --ci --threshold 0.8
"""
from __future__ import annotations

import json
import os
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path

import click
import yaml
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
console = Console()
sys.path.insert(0, str(Path(__file__).parent))

from harness.sandbox import MockSandbox
from harness.agent_loop import AgentLoop
from metrics.hybrid_scorer import HybridScorer
from metrics.judge import LLMJudge
from metrics.cli_judge import CLIJudge, probe_cli
from metrics.cross_validator import CrossValidator
from metrics.aggregators import MetricAggregator
from dataset.tracer import Tracer, GoldenDataset
from report.reporter import Reporter, DiffReporter, ci_gate


# ─────────────────────────────────────────────────────────────────────────────
#  Mock Agent：无 API Key 时用于验证评测链路本身
# ─────────────────────────────────────────────────────────────────────────────

def _mock_agent_run(spec: dict, sample: dict, sandbox: MockSandbox, trial_id: int):
    """
    模拟被测 Agent 行为，用于本地跑通评测链路（无需 API Key）。
    按任务 rubric 声明式注入缺陷，验证规则层能正确捕获并归因：
    注入什么缺陷 → 就该归因出什么标签，mock 模式的失败归因因此有验证价值。

    注入点自动从 rubric 推断（也可用 YAML 的 mock_injection.<difficulty> 显式覆盖）：
      l3 field_non_empty   → field_empty             （→ MISSING_FIELD 类归因）
      l2 critical field_equals → field_wrong          （→ WRONG_CHOICE / CRITICAL）
      l2 critical tool_param_equals → tool_param_wrong（→ TOOL_PARAM_ERROR）
      l3 critical regex_absent → reason_contains_violation（→ SAFETY_VIOLATION）
      l2 tool_sequence     → skip_last_tool           （→ TRAJECTORY_DEVIATION）
    """
    trace = sandbox.get_trace()
    tools = [t["function"]["name"] for t in spec.get("tools", [])]
    gt = dict(sample.get("ground_truth", {}))
    difficulty = sample.get("difficulty", "medium")
    rubric = spec.get("rubric", {})

    plan = _injection_plan(rubric)
    yaml_plan = (spec.get("mock_injection") or {}).get(difficulty)
    if yaml_plan:
        plan = yaml_plan                    # YAML 显式声明优先于自动推断

    # 注入调度：每轮周期 = 1 次正常路径 + 全部缺陷。
    # 例：k=3 → [正常, 缺陷A, 缺陷B]；k=4 → [正常, 缺陷A, 缺陷B, 缺陷C]
    slots: list[dict | None] = [None] + (plan or [])
    inject = (trial_id - 1) % len(slots)
    defect = slots[inject]

    # 工具调用：skip_last_tool 缺陷只走前置链路（缺最后一环），其余按序全调用
    call_tools = list(tools)
    if defect and defect.get("defect") == "skip_last_tool":
        seq = defect.get("tools") or []
        if seq:
            call_tools = seq[:-1]
    for name in call_tools:
        sandbox.call_tool(name, _mock_params(name, sample, gt))

    # 输出尽量贴合 ground_truth（剔除 expected_range 等评测元字段）
    output = {k: v for k, v in gt.items() if k != "expected_range"}
    output.setdefault("reason", "工具返回综合评分符合阈值，各维度均无违规风险，故作出该判定。")

    if defect:
        _apply_defect(defect, output, gt, sandbox)

    trace.final_output = json.dumps(output, ensure_ascii=False)
    trace.success = True
    trace.total_tokens = 350
    return trace


def _injection_plan(rubric: dict) -> list[dict]:
    """从 rubric 自动推断 mock 缺陷注入点，保证"注入什么缺陷 → 归因什么标签"一一对应"""
    plan: list[dict] = []
    for rule in rubric.get("l3", []) or []:
        if isinstance(rule, dict) and rule.get("check") == "field_non_empty":
            plan.append({"defect": "field_empty", "field": rule.get("field")})
    for rule in rubric.get("l2", []) or []:
        if (isinstance(rule, dict) and rule.get("check") == "field_equals"
                and _as_bool(rule.get("critical", False))):
            plan.append({"defect": "field_wrong", "field": rule.get("field")})
    for rule in rubric.get("l2", []) or []:
        if (isinstance(rule, dict) and rule.get("check") == "tool_param_equals"
                and _as_bool(rule.get("critical", False))):
            plan.append({"defect": "tool_param_wrong",
                         "tool": rule.get("tool"), "param": rule.get("param")})
    for rule in rubric.get("l3", []) or []:
        if (isinstance(rule, dict) and rule.get("check") == "regex_absent"
                and _as_bool(rule.get("critical", False))):
            plan.append({"defect": "reason_contains_violation",
                         "pattern": rule.get("pattern", "")})
    for rule in rubric.get("l2", []) or []:
        if isinstance(rule, dict) and rule.get("check") == "tool_sequence" and rule.get("sequence"):
            plan.append({"defect": "skip_last_tool", "tools": rule["sequence"]})
            break
    return plan


def _apply_defect(defect: dict, output: dict, gt: dict, sandbox: MockSandbox):
    """按声明式缺陷配置篡改输出 / 追加错误工具调用，模拟真实 Agent 典型失败"""
    kind = defect.get("defect")

    if kind == "field_empty":
        field = defect.get("field")
        if field:
            output[field] = ""

    elif kind == "field_wrong":
        field = defect.get("field")
        if field and field in output:
            output[field] = _flip_value(output[field])

    elif kind == "tool_param_wrong":
        tool, param = defect.get("tool"), defect.get("param")
        if tool and param is not None and param in gt:
            wrong_val = _flip_value(gt[param])
            # 追加一次参数错误的调用：副作用工具多调一次错参数即资损，规则层必须抓到
            sandbox.call_tool(tool, {**gt, param: wrong_val})
            if param in output:
                output[param] = wrong_val

    elif kind == "reason_contains_violation":
        word = _extract_first_word(defect.get("pattern") or "")
        output["reason"] = f"该商品包含违禁词「{word}」，应予以过滤。"


def _flip_value(v: Any) -> Any:
    """给字段造一个明显错误的值：bool 取反 / 数值 +1 / 字符串加后缀"""
    if isinstance(v, bool):
        return not v
    if isinstance(v, float):
        return v + 1.0
    if isinstance(v, int):
        return v + 1
    if isinstance(v, str):
        return v + "_x"
    return v


def _extract_first_word(pattern: str) -> str:
    """从正则 pattern 里提取第一个候选词用于注入（如 "(高仿|假冒)" → "高仿"）"""
    cleaned = pattern.replace("(", "").replace(")", "").replace("|", " ")
    m = re.search(r"[\u4e00-\u9fffA-Za-z0-9]+", cleaned)
    return m.group() if m else "违禁词"


def _as_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on", "是")
    if v is None:
        return default
    return bool(v)


def _mock_params(tool_name: str, sample: dict, gt: dict) -> dict:
    """构造 mock 工具调用参数，尽量贴合 ground_truth 以通过参数校验"""
    inp = sample.get("input", {})
    if tool_name == "review_content":
        return {"content": inp.get("content", "")}
    if tool_name == "query_order":
        return {"order_id": gt.get("order_id", "")}
    if tool_name == "submit_refund":
        return {"order_id": gt.get("order_id", ""), "amount": gt.get("amount", 0)}
    if tool_name == "send_notification":
        return {"user_id": "user_123", "message": "您的退款已提交"}
    return {}


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
        trace = _mock_agent_run(spec, sample, sandbox, trial_id)
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
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_judge(backend: str, model: str | None, mode: str = "numeric"):
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


@click.command()
@click.option("--task", required=True, help="Task Spec YAML 路径")
@click.option("--runs", default=None, type=int, show_default=True,
              help="每样本 trial 次数（Pass^k），默认取 YAML aggregation.k")
@click.option("--mock-run", is_flag=True, help="Mock 模式：不调 LLM，验证评测链路")
@click.option("--cross-check", is_flag=True, help="开启双模型交叉校验")
@click.option("--judge-a", default=None, help="交叉校验模型 A（openai 后端时生效）")
@click.option("--judge-b", default=None, help="交叉校验模型 B（openai 后端时生效）")
@click.option("--judge-a-backend", default="openai", show_default=True,
              type=click.Choice(["openai", "claude", "codex"]),
              help="交叉校验模型 A 后端：openai=API 调用；claude/codex=CLI 子进程调用")
@click.option("--judge-b-backend", default="openai", show_default=True,
              type=click.Choice(["openai", "claude", "codex"]),
              help="交叉校验模型 B 后端：openai=API 调用；claude/codex=CLI 子进程调用")
@click.option("--ci", is_flag=True, help="CI 门禁：低于阈值 exit 1")
@click.option("--ci-metric", default="pass_hat_k", show_default=True,
              type=click.Choice(["pass_hat_k", "pass_at_k", "vote_at_k"]),
              help="CI 指标：默认 Pass^k（全部 trial 稳定通过）")
@click.option("--threshold", default=0.8, show_default=True, help="CI Gate 阈值")
@click.option("--diff", is_flag=True, help="与 baseline 报告对比")
@click.option("--baseline", default=None, help="baseline 报告 JSON 路径")
@click.option("--no-ingest", is_flag=True, help="不将 Bad Case 写入 Golden Dataset")
@click.option("--no-regression", is_flag=True, help="不加载未修复 Golden Case 回归测试")
def main(task, runs, mock_run, cross_check, judge_a, judge_b,
         judge_a_backend, judge_b_backend,
         ci, ci_metric, threshold, diff, baseline, no_ingest, no_regression):
    """AgentHarness —— 多 Agent 协作链路评测平台"""

    task_path = Path(task)
    if not task_path.exists():
        console.print(f"[red]Task Spec 不存在: {task}[/]")
        sys.exit(1)
    spec = yaml.safe_load(task_path.read_text())

    task_id = spec["task_id"]
    agent_id = spec["agent_id"]
    version = spec.get("prompt_version", "v1")
    task_samples = spec.get("samples", [])
    golden = GoldenDataset()
    samples = task_samples if no_regression else load_regression_samples(golden, task_id, task_samples)
    agg_cfg = spec.get("aggregation", {})
    scoring_cfg = spec.get("scoring", {})
    pass_threshold = agg_cfg.get("pass_threshold", 0.6)
    runs = int(runs or agg_cfg.get("k", 3))     # CLI 未指定时使用 YAML aggregation.k
    run_id = f"{agent_id}_{version}_{datetime.now():%Y%m%d_%H%M%S}"

    console.print(f"\n[bold green]▶ AgentHarness[/]  "
                  f"task=[cyan]{task_id}[/]  agent=[cyan]{agent_id}[/]  "
                  f"version=[cyan]{version}[/]  samples={len(samples)}  runs={runs}"
                  + ("  [yellow][MOCK][/]" if mock_run else ""))
    regression_count = sum(1 for s in samples if s.get("regression_case_id"))
    if regression_count:
        console.print(f"[dim]  已自动加载 {regression_count} 条未修复 Golden Case[/]")

    # ── 组装评分器 ────────────────────────────────────────────────────────────
    judge = None
    if not mock_run and scoring_cfg.get("judge_rubric"):
        judge = LLMJudge(mode=scoring_cfg.get("judge_mode", "numeric"))

    scorer = HybridScorer(
        rubric=spec.get("rubric", {}),
        judge=judge,
        weights=scoring_cfg.get("weights"),
        judge_rubric=scoring_cfg.get("judge_rubric", ""),
        pass_threshold=pass_threshold,
        layer_weights=scoring_cfg.get("layer_weights"),   # {"l1": 0.2, "l2": 0.6, "l3": 0.2}
    )
    console.print(f"[dim]  双层权重: rule={scorer.w_rule:.1f} / judge={scorer.w_judge:.1f}[/]")

    cross_validator = None
    if cross_check and not mock_run:
        model_a = judge_a or "gpt-4o-mini"
        model_b = judge_b or "gpt-4o"
        judge_mode = scoring_cfg.get("judge_mode", "numeric")
        judge_a_obj = _build_judge(judge_a_backend, model_a, judge_mode)
        judge_b_obj = _build_judge(judge_b_backend, model_b, judge_mode)
        # CLI 后端启动前探测可用性（fail-open：不可用以 -1 进入仲裁，不崩主流程）
        for label, j in (("A", judge_a_obj), ("B", judge_b_obj)):
            if isinstance(j, CLIJudge):
                ok, info = j.available()
                icon = "✅" if ok else "⚠️"
                console.print(f"[dim]  judge {label}[{j.backend}] {icon} {info}[/]")
        cross_validator = CrossValidator(
            judge_a=judge_a_obj,
            judge_b=judge_b_obj,
            threshold=pass_threshold,
        )
        console.print(f"[dim]  交叉校验: {judge_a_obj.judge_model} × {judge_b_obj.judge_model}"
                      f"（backend: {judge_a_backend}/{judge_b_backend}）[/]")

    aggregator = MetricAggregator(pass_threshold=pass_threshold)
    tracer = Tracer(pass_threshold=pass_threshold)
    reporter = Reporter(task_id=task_id, agent_id=agent_id, prompt_version=version,
                        pass_threshold=pass_threshold)
    sample_map = {s["sample_id"]: s for s in samples}
    passed_keys: set[str] = set()

    # ── 主评测循环 ────────────────────────────────────────────────────────────
    for sample in samples:
        console.print(f"\n  [bold]▪ {sample['sample_id']}[/]")
        sample_scores = []
        for trial_id in range(1, runs + 1):
            try:
                trace, res = run_trial(spec, sample, trial_id, scorer,
                                       cross_validator, mock_run)
            except Exception as e:
                console.print(f"    [red]❌ trial{trial_id} 执行异常: {e}[/]")
                res = {"overall_score": 0.0, "failure_reason": "EXECUTION_ERROR",
                       "rule_score": 0.0, "judge_score": 0.0}
                trace = None

            score = res.get("overall_score", 0.0)
            sample_scores.append(score)
            reporter.add_result(res)
            aggregator.add(sample["sample_id"], trial_id,   # 干净 sample_id，不再拼 task 前缀
                           score, sample.get("difficulty", "medium"))
            if trace:
                tracer.add(trace)

        # 全部 trial 通过 → 该样本视为稳定通过（可用于回归 resolved 标记）
        if all(s >= pass_threshold for s in sample_scores):
            passed_keys.add(f"{task_id}::{sample['sample_id']}")

    # ── 聚合与报告 ────────────────────────────────────────────────────────────
    agg_summary = aggregator.summary()
    reporter.print_summary(aggregated=agg_summary)

    if runs > 1:
        console.print("\n[bold]采样鲁棒性（Pass@k / Vote@k / Pass^k）：[/]")
        for r in aggregator.aggregate():
            flags = " ".join("✅" if p else "❌" for p in r.trial_passed)
            console.print(f"  {r.sample_id}: pass@k={r.pass_at_k:.2f}  "
                          f"vote@k={r.vote_at_k:.0f}  "
                          f"[bold]pass^k={r.pass_hat_k:.0f}[/]  [{flags}]")

    # ── 交叉校验报告 ──────────────────────────────────────────────────────────
    if cross_validator:
        cr = cross_validator.report()
        console.print(f"\n[bold]Judge 交叉校验：[/]")
        console.print(f"  一致率: [cyan]{cr['agreement_rate']:.1%}[/]  "
                      f"平均分差: {cr['mean_abs_diff']:.3f}  "
                      f"争议样本: {cr['disputed_count']}")
        console.print(f"  [dim]{cr['verdict']}[/]")

    # ── Trace 落盘 ────────────────────────────────────────────────────────────
    trace_path = tracer.dump_traces(run_id)
    console.print(f"\n[dim]Trace 已落盘: {trace_path}[/]")

    # ── Bad Case → Golden Dataset 闭环 ────────────────────────────────────────
    bad = tracer.bad_cases()
    if mock_run:
        # Mock 假数据不得进入回归集，也不得把历史真实 Bad Case 误标为已修复
        console.print("[yellow]Mock 模式：跳过 Golden Dataset 写入与回归 resolved 标记[/]")
    else:
        if bad and not no_ingest:
            r = golden.ingest(bad, sample_map=sample_map)
            console.print(f"[yellow]Bad Case {len(bad)} 条 → Golden Dataset "
                          f"新增 {r['added']} / 复现 {r['updated']}[/]")
        if passed_keys:
            n = golden.mark_resolved(passed_keys)
            if n:
                console.print(f"[green]回归验证：{n} 条历史 Bad Case 已修复[/]")

        gstats = golden.stats()
        if gstats["total_cases"]:
            console.print(f"[dim]回归集: 共 {gstats['total_cases']} 条，"
                          f"已修复 {gstats['resolved']}，修复率 {gstats['resolve_rate']:.1%}"
                          + (f"，长期未修复 {len(gstats['stale_cases'])} 条" if gstats["stale_cases"] else "")
                          + "[/]")

    tstats = tracer.stats()
    if tstats["failure_breakdown"]:
        console.print("\n[bold]失败归因分布：[/]")
        for reason, cnt in sorted(tstats["failure_breakdown"].items(),
                                  key=lambda x: -x[1]):
            console.print(f"  · {reason}: {cnt}")

    # ── 保存 + Diff ───────────────────────────────────────────────────────────
    report_path = reporter.save(run_id=run_id, extra={"aggregation": agg_summary})
    console.print(f"[dim]报告已保存: {report_path}[/]")

    if diff and baseline:
        bp = Path(baseline)
        if bp.exists():
            DiffReporter(json.loads(bp.read_text()), reporter.to_dict()).print_diff()
        else:
            console.print(f"[yellow]baseline 不存在: {baseline}[/]")

    # ── CI Gate ───────────────────────────────────────────────────────────────
    ci_pass_rate, ci_metric_label = select_ci_metric(agg_summary, reporter.to_dict(), ci_metric)
    console.print(f"[dim]CI 指标: {ci_metric_label}[/]")
    ci_gate(ci_pass_rate, threshold=agg_cfg.get("ci_threshold", threshold), strict=ci)
    console.print()


if __name__ == "__main__":
    main()
