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
import sys
from datetime import datetime
from pathlib import Path

import click
import yaml
from dotenv import load_dotenv
from rich.console import Console

# 业务逻辑拆分（P3-1）：本文件只保留 CLI 壳与主流程编排，
# trial 执行 / 回归合并 / Judge 工厂 / CI 口径见 harness/pipeline.py，
# Mock Agent 见 harness/mock_agent.py。
from harness.pipeline import (          # noqa: F401（下划线别名供旧 import 兼容）
    load_regression_samples,
    run_trial,
    run_chain,
    select_ci_metric,
    build_judge as _build_judge,        # 兼容 tests/test_pipeline.py 的 from run_eval import _build_judge
)

load_dotenv()
console = Console()
sys.path.insert(0, str(Path(__file__).parent))

from dataset.tracer import Tracer, GoldenDataset
from metrics.aggregators import MetricAggregator
from metrics.cli_judge import CLIJudge
from metrics.cross_validator import CrossValidator
from metrics.hybrid_scorer import HybridScorer
from metrics.judge import LLMJudge
from report.reporter import Reporter, DiffReporter, ci_gate


@click.command()
@click.option("--task", required=True, help="Task Spec YAML 路径")
@click.option("--runs", default=None, type=int, show_default=True,
              help="每样本 trial 次数（Pass^k），默认取 YAML aggregation.k")
@click.option("--mock-run", is_flag=True, help="Mock 模式：不调 LLM，验证评测链路")
@click.option("--judge", "with_judge", is_flag=True,
              help="Mock 模式下也真实调用 LLM Judge / 交叉校验（需 OPENAI_API_KEY，验证 Judge 链路）")
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
def main(task, runs, mock_run, with_judge, cross_check, judge_a, judge_b,
         judge_a_backend, judge_b_backend,
         ci, ci_metric, threshold, diff, baseline, no_ingest, no_regression):
    """AgentHarness —— 多 Agent 协作链路评测平台"""

    task_path = Path(task)
    if not task_path.exists():
        console.print(f"[red]Task Spec 不存在: {task}[/]")
        sys.exit(1)
    spec = yaml.safe_load(task_path.read_text())

    task_id = spec["task_id"]
    agents = spec.get("agents")
    is_chain = bool(agents)             # agents 非空 → 多 Agent 协作链路模式
    agent_id = (spec.get("chain_id") or agents[0]["agent_id"]) if is_chain else spec["agent_id"]
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
    if is_chain:
        console.print("[dim]  协作链路: " + " → ".join(a["agent_id"] for a in agents) + "[/]")
    regression_count = sum(1 for s in samples if s.get("regression_case_id"))
    if regression_count:
        console.print(f"[dim]  已自动加载 {regression_count} 条未修复 Golden Case[/]")

    # --judge 只在显式开启时让 mock 模式发真实 LLM 调用；缺 Key 时友好降级
    allow_real_judge = (not mock_run) or with_judge
    has_llm_key = bool(os.getenv("OPENAI_API_KEY"))

    # ── 组装评分器 ────────────────────────────────────────────────────────────
    judge = None
    if scoring_cfg.get("judge_rubric"):
        if allow_real_judge and has_llm_key:
            judge = LLMJudge(mode=scoring_cfg.get("judge_mode", "numeric"))
            if mock_run:
                console.print("[dim]  --judge：mock 模式下真实调用 LLM Judge（产生真实 LLM 调用）[/]")
        elif allow_real_judge and not has_llm_key:
            # OpenAI SDK 在构造时即校验 Key，缺 Key 构造会直接抛错；提前拦截并降级。
            # 真实模式（非 mock）下 Agent 调用同样会失败（AgentLoop 会给友好报错）。
            console.print("[yellow]  未检测到 OPENAI_API_KEY：Judge 不可用，回退纯规则评分[/]")

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
    if cross_check and allow_real_judge:
        need_key = judge_a_backend == "openai" or judge_b_backend == "openai"
        if need_key and not has_llm_key:
            console.print("[yellow]  未检测到 OPENAI_API_KEY：交叉校验（openai 后端）不可用，跳过[/]")
        else:
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
                if is_chain:
                    trace, res = run_chain(spec, sample, trial_id, scorer,
                                           cross_validator, mock_run)
                else:
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
