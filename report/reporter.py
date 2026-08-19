"""
跨版本 Prompt diff 报告 + CI Gate

跨批次对比：和上一版结果对比，明确哪些指标进步/退步
给出可操作建议：不只说"分数低"，而是给出具体的 Prompt 改进方向

CI Gate：Prompt 改动后 pass rate < threshold 则 exit(1)，阻止发布
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


# ─── 评测报告 ──────────────────────────────────────────────────────────────

class Reporter:
    """生成单次评测的汇总报告"""

    def __init__(self, task_id: str, agent_id: str, prompt_version: str,
                 pass_threshold: float = 0.6):
        self.task_id = task_id
        self.agent_id = agent_id
        self.prompt_version = prompt_version
        self.pass_threshold = pass_threshold
        self.eval_results: list[dict] = []

    def add_result(self, result: dict):
        self.eval_results.append(result)

    def to_dict(self) -> dict:
        if not self.eval_results:
            return {}
        # 口径区分：n = samples × runs 是 trial 总数；真实样本数按 sample_id 去重
        #（历史版本 total_samples 误存 trial 数，靠注释兼容；现已显式区分）。
        n = len(self.eval_results)
        sample_ids = {r.get("sample_id") for r in self.eval_results if r.get("sample_id")}
        total_samples = len(sample_ids) or n   # 异常分支可能缺 sample_id，回退 trial 数
        scores = [r.get("overall_score", 0.0) for r in self.eval_results]
        rule_scores = [r.get("rule_score", 0.0) for r in self.eval_results]
        judge_scores = [r.get("judge_score", 0.0) for r in self.eval_results]
        pass_rate = sum(1 for s in scores if s >= self.pass_threshold) / n

        failure_breakdown: dict[str, int] = {}
        for r in self.eval_results:
            fr = r.get("failure_reason")
            if fr:
                failure_breakdown[fr] = failure_breakdown.get(fr, 0) + 1

        # 逐条 check 的通过率：定位到底是哪个能力项拖后腿
        check_stats: dict[str, dict] = {}
        for r in self.eval_results:
            detail = r.get("rule_detail", {})
            for layer in ("l1", "l2", "l3"):
                for name, sc in (detail.get(layer) or {}).items():
                    if sc < 0:
                        continue
                    st = check_stats.setdefault(name, {"layer": layer, "pass": 0, "total": 0})
                    st["total"] += 1
                    st["pass"] += int(sc == 1)
        for st in check_stats.values():
            st["pass_rate"] = round(st["pass"] / st["total"], 4) if st["total"] else 0.0

        # Judge 调用节省量：短路机制拦下的 LLM 调用
        skipped = sum(1 for r in self.eval_results if r.get("judge_skipped"))

        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "prompt_version": self.prompt_version,
            "total_trials": n,              # trial 总数（samples × runs），主口径
            "total_samples": total_samples, # 真实样本数（按 sample_id 去重）
            # trial 级口径（随 --runs 漂移，跨 runs 对比请用样本级 pass@k）
            "pass_rate": round(pass_rate, 4),
            "avg_score": round(sum(scores) / n, 4),
            "avg_rule_score": round(sum(rule_scores) / n, 4),
            "avg_judge_score": round(sum(judge_scores) / n, 4),
            "judge_skipped": skipped,
            "judge_saved_rate": round(skipped / n, 4),
            "failure_breakdown": failure_breakdown,
            "check_stats": check_stats,
            "created_at": datetime.now().isoformat(),
        }

    def print_summary(self, aggregated: dict | None = None):
        """Rich 终端输出评测汇总"""
        data = self.to_dict()
        console.print(f"\n[bold cyan]═══ 评测报告：{self.agent_id} / {self.prompt_version} ═══[/]")

        table = Table(box=box.SIMPLE, show_header=True)
        table.add_column("指标", style="bold")
        table.add_column("值")

        table.add_row("总 trial 数", str(data.get("total_trials", data.get("total_samples", 0))))
        pass_rate = data.get("pass_rate", 0.0)
        color = "green" if pass_rate >= 0.8 else ("yellow" if pass_rate >= 0.6 else "red")
        table.add_row("Pass Rate (trial 级)", f"[{color}]{pass_rate:.1%}[/]")
        table.add_row("Final Score", f"{data.get('avg_score', 0.0):.4f}")
        table.add_row("  ├ 规则层", f"{data.get('avg_rule_score', 0.0):.4f}")
        table.add_row("  └ Judge层", f"{data.get('avg_judge_score', 0.0):.4f}")

        if aggregated:
            table.add_row("样本级 Pass@k", f"{aggregated.get('mean_pass@k', 0.0):.4f}")
            table.add_row("Vote@k", f"{aggregated.get('mean_vote@k', 0.0):.4f}")
            table.add_row("Pass^k", f"{aggregated.get('mean_pass^k', 0.0):.4f}")

        console.print(table)

        # 逐 check 通过率：直接定位薄弱能力项
        stats = data.get("check_stats", {})
        weak = {k: v for k, v in stats.items() if v["pass_rate"] < 1.0}
        if weak:
            console.print("[yellow]未满分 check（按通过率升序）：[/]")
            for name, st in sorted(weak.items(), key=lambda x: x[1]["pass_rate"]):
                console.print(f"  · [{st['layer'].upper()}] {name}: "
                              f"{st['pass']}/{st['total']} = {st['pass_rate']:.0%}")

    def save(self, output_dir: str = "report/output",
             run_id: str | None = None, extra: dict | None = None):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        rid = run_id or f"{self.agent_id}_{self.prompt_version}_{datetime.now():%Y%m%d_%H%M%S}"
        filename = out / f"{rid}.json"
        data = self.to_dict()
        if extra:
            data.update(extra)
        with open(filename, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return str(filename)


# ─── 跨版本 Diff 报告 ──────────────────────────────────────────────────────

class DiffReporter:
    """
    比较两个版本 Prompt 的评测结果，输出：
    - 哪些 task 从 pass → fail（退步）
    - 哪些 task 从 fail → pass（进步）
    - 整体指标对比

    """

    def __init__(self, report_v1: dict, report_v2: dict):
        self.v1 = report_v1
        self.v2 = report_v2

    def compare(self) -> dict:
        # 优先比较样本级 Pass^k：它不随 runs 改变口径，且代表实际稳定性。
        v1_rate = (self.v1.get("aggregation", {}) or {}).get(
            "mean_pass^k", self.v1.get("pass_rate", 0.0))
        v2_rate = (self.v2.get("aggregation", {}) or {}).get(
            "mean_pass^k", self.v2.get("pass_rate", 0.0))
        delta = v2_rate - v1_rate

        return {
            "v1": self.v1.get("prompt_version"),
            "v2": self.v2.get("prompt_version"),
            "pass_rate_v1": v1_rate,
            "pass_rate_v2": v2_rate,
            "primary_metric": "Pass^k（样本级稳定通过率）",
            "delta": round(delta, 4),
            "trend": "✅ 进步" if delta > 0.01 else ("⚠️ 持平" if abs(delta) <= 0.01 else "❌ 退步"),
            "avg_score_v1": self.v1.get("avg_score", 0.0),
            "avg_score_v2": self.v2.get("avg_score", 0.0),
            "failure_delta": self._diff_failures(),
            "check_delta": self._diff_checks(),
        }

    def _diff_checks(self) -> dict:
        """逐 check 对比：精确定位是哪个能力项进步/退步（而非只看总分）"""
        c1 = self.v1.get("check_stats", {}) or {}
        c2 = self.v2.get("check_stats", {}) or {}
        out = {}
        for k in set(c1) | set(c2):
            r1 = c1.get(k, {}).get("pass_rate", 0.0)
            r2 = c2.get(k, {}).get("pass_rate", 0.0)
            if abs(r2 - r1) > 1e-6:
                out[k] = {"v1": r1, "v2": r2, "delta": round(r2 - r1, 4)}
        return out

    def _diff_failures(self) -> dict:
        f1 = self.v1.get("failure_breakdown", {})
        f2 = self.v2.get("failure_breakdown", {})
        all_keys = set(f1) | set(f2)
        return {
            k: {"v1": f1.get(k, 0), "v2": f2.get(k, 0),
                "delta": f2.get(k, 0) - f1.get(k, 0)}
            for k in all_keys
        }

    def print_diff(self):
        diff = self.compare()
        console.print(f"\n[bold magenta]═══ Prompt Diff: {diff['v1']} → {diff['v2']} ═══[/]")
        console.print(f"{diff['primary_metric']}:  {diff['pass_rate_v1']:.1%}  →  {diff['pass_rate_v2']:.1%}  {diff['trend']}")
        console.print(f"Avg Score:  {diff['avg_score_v1']:.4f}  →  {diff['avg_score_v2']:.4f}")
        for reason, d in diff.get("failure_delta", {}).items():
            sign = "+" if d["delta"] > 0 else ""
            console.print(f"  · {reason}: {d['v1']} → {d['v2']} ({sign}{d['delta']})")

        cd = diff.get("check_delta", {})
        if cd:
            console.print("[bold]逐 check 变化（精确定位进步/退步项）：[/]")
            for name, d in sorted(cd.items(), key=lambda x: x[1]["delta"]):
                arrow = "[green]↑[/]" if d["delta"] > 0 else "[red]↓[/]"
                console.print(f"  {arrow} {name}: {d['v1']:.0%} → {d['v2']:.0%}")


# ─── CI Gate ───────────────────────────────────────────────────────────────

def ci_gate(pass_rate: float, threshold: float = 0.8,
            strict: bool = True) -> bool:
    """
    CI 门禁：pass_rate 低于 threshold 时输出警告，strict 模式下 sys.exit(1)。

    用法：在 GitHub Actions / CI 流程中调用。
    Prompt 改动不达标则阻止发布。
    """
    if pass_rate >= threshold:
        console.print(f"[green]✅ CI Gate 通过: pass_rate={pass_rate:.1%} >= {threshold:.1%}[/]")
        return True
    else:
        console.print(
            f"[red]❌ CI Gate 失败: pass_rate={pass_rate:.1%} < {threshold:.1%}[/]\n"
            f"   需要修复 Agent Prompt 或 Task Spec 后重新评测。"
        )
        if strict:
            sys.exit(1)
        return False
