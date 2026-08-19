"""
聚合器 + 数据闭环单元测试

覆盖：
  · Pass@k / Vote@k / Pass^k 数学正确性
  · Bad Case → Golden Dataset 落盘、去重、复现计数
  · 回归验证 resolved 标记
"""
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.sandbox import AgentTrace
from metrics.aggregators import MetricAggregator, _mean_pass_at_k, _mean_pass_hat_k
from dataset.tracer import Tracer, GoldenDataset
from run_eval import load_regression_samples, select_ci_metric, _build_judge


# ─── 聚合指标 ────────────────────────────────────────────────────────────────

def test_pass_at_k_formula():
    """n=3 次采样中 c=1 次通过，Pass@3 应为 1.0（至少一次通过）"""
    assert _mean_pass_at_k(n=3, c=1, k=3) == pytest.approx(1.0)
    assert _mean_pass_at_k(n=3, c=0, k=3) == pytest.approx(0.0)


def test_pass_hat_k_requires_all_pass():
    """Pass^k 是鲁棒性指标：必须 k 次全通过"""
    assert _mean_pass_hat_k([True, True, True]) == 1.0
    assert _mean_pass_hat_k([True, False, True]) == 0.0


def test_aggregator_distinguishes_capability_and_robustness():
    """
    核心场景：3 次采样 2 次通过
      Pass@k = 1.0（能力上界：有能力做对）
      Pass^k = 0.0（鲁棒性：不稳定，不可上线）
    """
    agg = MetricAggregator(pass_threshold=0.6)
    for i, s in enumerate([0.9, 0.3, 0.8], 1):
        agg.add("task_a", i, s)
    r = agg.aggregate()[0]
    assert r.pass_at_k == pytest.approx(1.0)
    assert r.pass_hat_k == 0.0
    assert r.vote_at_k == 1.0        # 多数票通过


def test_aggregator_summary_keys():
    agg = MetricAggregator()
    agg.add("t", 1, 1.0)
    s = agg.summary()
    assert {"mean_pass@k", "mean_vote@k", "mean_pass^k"} <= set(s)


# ─── 数据闭环 ────────────────────────────────────────────────────────────────

def _bad_trace(task_id="task_s001", sample_id="s001",
               reason="WRONG_CHOICE", score=0.2):
    t = AgentTrace(task_id=task_id, sample_id=sample_id, agent_id="a1")
    t.final_output = '{"x":1}'
    t.eval_result = {"overall_score": score, "failure_reason": reason,
                     "rule_detail": {"failed": [{"name": "c1", "detail": "d"}]},
                     "suggestion": "改 prompt"}
    t.failure_reason = reason
    return t


def test_tracer_identifies_bad_cases():
    tr = Tracer(pass_threshold=0.6)
    tr.add(_bad_trace(score=0.2))
    good = AgentTrace(task_id="task_s002", agent_id="a1")
    good.eval_result = {"overall_score": 0.9}
    tr.add(good)

    bad = tr.bad_cases()
    assert len(bad) == 1
    assert tr.stats()["pass_rate"] == 0.5


def test_golden_dataset_ingest_and_dedup():
    """同类问题只留一条代表样本，复现则计数 +1"""
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "g.jsonl")
        gd = GoldenDataset(path)

        r1 = gd.ingest([_bad_trace()])
        assert r1 == {"added": 1, "updated": 0}

        r2 = gd.ingest([_bad_trace()])         # 同类问题复现
        assert r2 == {"added": 0, "updated": 1}
        assert len(gd.regression_cases()) == 1
        assert gd.regression_cases()[0]["fail_count"] == 2


def test_golden_dataset_persists_across_instances():
    """JSONL 落盘，可被下一轮 CI 加载"""
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "g.jsonl")
        GoldenDataset(path).ingest([_bad_trace()])
        assert len(GoldenDataset(path).regression_cases()) == 1


def test_regression_mark_resolved():
    """回归验证：修复后的 Bad Case 标记 resolved，形成闭环"""
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "g.jsonl")
        gd = GoldenDataset(path)
        gd.ingest([_bad_trace(task_id="task_s001")])

        n = gd.mark_resolved({"task_s001::s001"})
        assert n == 1
        assert gd.regression_cases() == []                    # 未修复列表已清空
        assert gd.stats()["resolve_rate"] == 1.0


def test_stale_case_detection():
    """长期未修复（复现 >= 3 次）应被单独暴露"""
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "g.jsonl")
        gd = GoldenDataset(path)
        for _ in range(3):
            gd.ingest([_bad_trace()])
        assert len(gd.stats()["stale_cases"]) == 1


def test_golden_cases_are_loaded_into_next_task_run():
    """闭环关键：沉淀的坏例会覆盖合并进同 task 的下一轮评测。"""
    with tempfile.TemporaryDirectory() as d:
        gd = GoldenDataset(str(Path(d) / "g.jsonl"))
        t = _bad_trace(task_id="task_a", sample_id="historic")
        gd.ingest([t], sample_map={"historic": {
            "input": {"q": "old"}, "ground_truth": {"answer": 1},
            "expected_behavior": "历史失败样本",
        }})
        samples = load_regression_samples(gd, "task_a", [{
            "sample_id": "current", "input": {}, "ground_truth": {},
        }])
        loaded = {s["sample_id"]: s for s in samples}
        assert set(loaded) == {"current", "historic"}
        assert loaded["historic"]["difficulty"] == "regression"
        assert loaded["historic"]["input"] == {"q": "old"}


def test_ci_defaults_can_distinguish_stability_from_capability():
    aggregation = {"total_samples": 2, "mean_pass@k": 1.0,
                   "mean_pass^k": 0.0, "mean_vote@k": 0.5}
    score, label = select_ci_metric(aggregation, {}, "pass_hat_k")
    assert score == 0.0 and "Pass^k" in label
    score, label = select_ci_metric(aggregation, {}, "pass_at_k")
    assert score == 1.0 and "Pass@k" in label


def test_cross_judge_inherits_task_pattern_mode():
    judge = _build_judge("claude", "test-model", mode="pattern")
    assert judge.mode == "pattern"
