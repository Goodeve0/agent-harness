"""
Trace-to-Dataset 闭环：Bad Case 自动打标 → Golden Dataset → 回归验证

数据飞轮：

    ┌─────────────────────────────────────────────────────────┐
    │  评测执行 → Trace 落盘                                    │
    │      ↓                                                   │
    │  失败样本自动打归因标签（FORMAT_ERROR / TRAJECTORY_...）   │
    │      ↓                                                   │
    │  一键归入 Golden Dataset（带标签的回归样本）               │
    │      ↓                                                   │
    │  下次评测自动加载回归集 → 验证问题是否真修复               │
    │      ↓                                                   │
    │  回归通过则标记 resolved，未通过则计数 +1（长期未修复告警） │
    └─────────────────────────────────────────────────────────┘

关键设计：
  · 归因标签由规则层直接产出（哪条 check 挂了 → 什么原因），不依赖人工分类
  · Golden Dataset 以 JSONL 落盘，可直接被 CI 加载，天然支持 Git 版本管理
  · 回归样本记录 first_seen / last_seen / fail_count，暴露长期未修复问题
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from harness.sandbox import AgentTrace


# ─────────────────────────────────────────────────────────────────────────────
#  Tracer：Trace 收集 + Bad Case 打标
# ─────────────────────────────────────────────────────────────────────────────

class Tracer:
    """收集本轮所有 Trace，识别 Bad Case 并落盘"""

    def __init__(self, pass_threshold: float = 0.6,
                 trace_dir: str = "report/traces"):
        self.pass_threshold = pass_threshold
        self.trace_dir = Path(trace_dir)
        self._traces: list[AgentTrace] = []

    def add(self, trace: AgentTrace):
        self._traces.append(trace)

    @property
    def traces(self) -> list[AgentTrace]:
        return self._traces

    def bad_cases(self) -> list[AgentTrace]:
        """所有 overall_score 低于阈值的 Trace"""
        out = []
        for t in self._traces:
            score = (t.eval_result or {}).get("overall_score", 0.0)
            if score < self.pass_threshold:
                t.is_bad_case = True
                if not t.failure_reason:
                    t.failure_reason = (t.eval_result or {}).get("failure_reason") or "UNKNOWN"
                out.append(t)
        return out

    def dump_traces(self, run_id: str) -> str:
        """全量 Trace 落盘（可复现 / 可回放，评测平台的基础设施）"""
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        path = self.trace_dir / f"{run_id}.jsonl"
        with open(path, "w") as f:
            for t in self._traces:
                f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")
        return str(path)

    def stats(self) -> dict:
        total = len(self._traces)
        if not total:
            return {"total": 0, "bad_cases": 0, "pass_rate": 0.0, "failure_breakdown": {}}
        bad = self.bad_cases()
        breakdown: dict[str, int] = {}
        for t in bad:
            r = t.failure_reason or "UNKNOWN"
            breakdown[r] = breakdown.get(r, 0) + 1
        return {
            "total": total,
            "bad_cases": len(bad),
            "pass_rate": round((total - len(bad)) / total, 4),
            "failure_breakdown": breakdown,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  GoldenDataset：回归测试集
# ─────────────────────────────────────────────────────────────────────────────

class GoldenDataset:
    """
    带标签的回归测试集，JSONL 落盘。

    去重键：(task_id, sample_id, failure_reason)
      同类问题只留一条代表样本，避免回归集被同质 Bad Case 淹没。
    """

    def __init__(self, path: str = "dataset/golden_set.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path) as f:
            return [json.loads(l) for l in f if l.strip()]

    def _save(self):
        with open(self.path, "w") as f:
            for e in self._entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    @staticmethod
    def _key(task_id: str, sample_id: str, reason: str) -> str:
        return f"{task_id}::{sample_id}::{reason}"

    # ── 核心：一键从 Trace 归入回归集 ────────────────────────────────────────
    def ingest(self, bad_cases: list[AgentTrace],
               sample_map: dict[str, dict] | None = None) -> dict:
        """
        将 Bad Case 转为标准回归样本。

        Args:
            bad_cases:  Tracer 识别出的失败 Trace
            sample_map: {sample_id: sample_spec}，用于回填 input / ground_truth

        Returns:
            {"added": n, "updated": m}  新增 / 复现计数
        """
        sample_map = sample_map or {}
        index = {self._key(e["task_id"], e["sample_id"], e["failure_reason"]): e
                 for e in self._entries}
        now = datetime.now().isoformat(timespec="seconds")
        added = updated = 0

        for t in bad_cases:
            reason = t.failure_reason or "UNKNOWN"
            # 新链路始终有干净的 sample_id；为空时兼容旧 trace（task_id 形如 "taskId_sampleId"）
            sample_id = t.sample_id or ""
            if not sample_id:
                sample_id = t.task_id.rsplit("_", 1)[-1]
            key = self._key(t.task_id, sample_id, reason)
            eval_res = t.eval_result or {}
            spec = sample_map.get(sample_id, {})

            if key in index:
                # 同类问题复现 → 计数 +1，说明尚未修复
                e = index[key]
                e["fail_count"] = e.get("fail_count", 1) + 1
                e["last_seen"] = now
                e["resolved"] = False
                e["latest_score"] = eval_res.get("overall_score", 0.0)
                updated += 1
            else:
                entry = {
                    "case_id": f"gc_{len(self._entries) + 1:04d}",
                    "task_id": t.task_id,
                    "sample_id": sample_id,
                    "agent_id": t.agent_id,
                    "prompt_version": t.prompt_version,
                    "failure_reason": reason,
                    "latest_score": eval_res.get("overall_score", 0.0),
                    # 回归所需的完整输入（可脱离原 YAML 独立重放）
                    "input": spec.get("input", {}),
                    "ground_truth": spec.get("ground_truth", {}),
                    "expected_behavior": spec.get("expected_behavior", ""),
                    # 归因证据：哪几条 check 挂了 + 改进建议
                    "failed_checks": eval_res.get("rule_detail", {}).get("failed", []),
                    "suggestion": eval_res.get("suggestion", ""),
                    "trace_snapshot": t.to_dict(),
                    "first_seen": now,
                    "last_seen": now,
                    "fail_count": 1,
                    "resolved": False,
                    "regression": True,
                }
                self._entries.append(entry)
                index[key] = entry
                added += 1

        self._save()
        return {"added": added, "updated": updated}

    # ── 回归验证：本轮通过的历史 Bad Case 标记为 resolved ────────────────────
    def mark_resolved(self, passed_keys: set[str]) -> int:
        """
        Args:
            passed_keys: 本轮通过的 "{task_id}::{sample_id}" 集合
        Returns:
            本次新标记为已修复的样本数
        """
        n = 0
        now = datetime.now().isoformat(timespec="seconds")
        for e in self._entries:
            k = f"{e['task_id']}::{e['sample_id']}"
            if k in passed_keys and not e.get("resolved"):
                e["resolved"] = True
                e["resolved_at"] = now
                n += 1
        self._save()
        return n

    def regression_cases(self, include_resolved: bool = False) -> list[dict]:
        """获取回归样本，默认只返回未修复的"""
        return [e for e in self._entries
                if e.get("regression") and (include_resolved or not e.get("resolved"))]

    def stats(self) -> dict:
        total = len(self._entries)
        resolved = sum(1 for e in self._entries if e.get("resolved"))
        breakdown: dict[str, int] = {}
        for e in self._entries:
            r = e.get("failure_reason", "UNKNOWN")
            breakdown[r] = breakdown.get(r, 0) + 1
        # 长期未修复：复现 >= 3 次仍未 resolved
        stale = [e["case_id"] for e in self._entries
                 if e.get("fail_count", 0) >= 3 and not e.get("resolved")]
        return {
            "total_cases": total,
            "resolved": resolved,
            "pending": total - resolved,
            "resolve_rate": round(resolved / total, 4) if total else 0.0,
            "breakdown": breakdown,
            "stale_cases": stale,
        }
