"""
规则层 + 双层评分单元测试

覆盖：
  · 轨迹类 check（工具调用 / 顺序 / 参数）
  · 结果类 check（字段匹配 / 区间）
  · critical 一票否决机制
  · 加权计算
  · 双层加权合并与短路策略
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.sandbox import AgentTrace, MockSandbox
from metrics.rule_checker import RuleChecker
from metrics.hybrid_scorer import HybridScorer, FailureReason


def make_trace(output: str, tool_calls: list[tuple[str, dict]] | None = None) -> AgentTrace:
    t = AgentTrace(task_id="t1", agent_id="a1")
    for name, params in (tool_calls or []):
        t.add_step(name, params, {"ok": True})
    t.final_output = output
    t.success = True
    return t


# ─── 沙箱 ────────────────────────────────────────────────────────────────────

def test_sandbox_records_trace():
    sb = MockSandbox(mock_apis={"foo": {"v": 1}}, task_id="t", agent_id="a")
    assert sb.call_tool("foo", {"x": 1}) == {"v": 1}
    assert len(sb.get_trace().steps) == 1
    assert sb.get_trace().steps[0].tool_name == "foo"


def test_sandbox_blocks_unknown_tool():
    """未注册工具不允许穿透到真实 API"""
    sb = MockSandbox(mock_apis={}, task_id="t", agent_id="a")
    resp = sb.call_tool("unknown", {})
    assert "error" in resp
    assert sb.get_trace().steps[0].error is not None


# ─── L1 格式层 ───────────────────────────────────────────────────────────────

def test_l1_json_and_fields():
    rc = RuleChecker({"l1": {"format": "json", "required_fields": ["a", "b"]}})
    rs = rc.check(make_trace('{"a":1,"b":2}'))
    assert rs.by_layer("l1") == {"json_format": 1, "required_fields": 1}
    assert rs.score() == 1.0


def test_l1_detects_missing_field():
    rc = RuleChecker({"l1": {"format": "json", "required_fields": ["a", "b"]}})
    rs = rc.check(make_trace('{"a":1}'))
    assert rs.by_layer("l1")["required_fields"] == 0
    assert rs.score() == 0.0        # L1 天然 critical


def test_l1_parses_markdown_fenced_json():
    """真实 Agent 常用 ```json 包裹输出，不应误判为格式错误"""
    rc = RuleChecker({"l1": {"format": "json", "required_fields": ["a"]}})
    rs = rc.check(make_trace('好的，结果如下：\n```json\n{"a": 1}\n```'))
    assert rs.by_layer("l1")["json_format"] == 1


def test_l1_strict_json_rejects_surrounding_text():
    rc = RuleChecker({"l1": {"format": "json", "strict_json": True,
                              "required_fields": ["a"]}})
    rs = rc.check(make_trace('结果如下： {"a": 1}'))
    assert rs.by_layer("l1")["json_format"] == 0


def test_l1_tool_call_errors_are_critical():
    rc = RuleChecker({"l1": {"tool_calls_valid": True}})
    trace = make_trace("{}", [("unknown", {})])
    trace.steps[0].error = "ToolNotFound: unknown"
    assert rc.check(trace).score() == 0.0


# ─── L2 轨迹类 check ─────────────────────────────────────────────────────────

def test_tool_called_and_sequence():
    rubric = {"l2": [
        {"name": "c1", "check": "tool_called", "tool": "query"},
        {"name": "c2", "check": "tool_sequence", "sequence": ["query", "refund"]},
    ]}
    trace = make_trace("{}", [("query", {}), ("refund", {})])
    rs = RuleChecker(rubric).check(trace)
    assert rs.by_layer("l2") == {"c1": 1, "c2": 1}


def test_tool_sequence_detects_wrong_order():
    rubric = {"l2": [{"name": "seq", "check": "tool_sequence",
                      "sequence": ["query", "refund"]}]}
    trace = make_trace("{}", [("refund", {}), ("query", {})])
    rs = RuleChecker(rubric).check(trace)
    assert rs.by_layer("l2")["seq"] == 0


def test_tool_param_equals_catches_hallucinated_amount():
    """Agent 幻觉高发场景：退款金额与订单实际金额不符"""
    rubric = {"l2": [{"name": "amt", "check": "tool_param_equals",
                      "tool": "refund", "param": "amount", "expect": "$gt.amount"}]}
    trace = make_trace("{}", [("refund", {"amount": 199.0})])
    rs = RuleChecker(rubric).check(trace, {"amount": 299.0})
    assert rs.by_layer("l2")["amt"] == 0
    assert "299" in rs.failed_checks()[0].detail


def test_max_steps_efficiency():
    rubric = {"l2": [{"name": "eff", "check": "max_steps", "limit": 2}]}
    trace = make_trace("{}", [("a", {}), ("b", {}), ("c", {})])
    assert RuleChecker(rubric).check(trace).by_layer("l2")["eff"] == 0


# ─── L2 结果类 check ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("actual,expect,want", [
    (299, 299.0, 1),        # int / float 宽松相等
    ("299", 299.0, 1),      # str / float
    (True, True, 1),
    (199.0, 299.0, 0),
])
def test_field_equals_loose_match(actual, expect, want):
    rubric = {"l2": [{"name": "f", "check": "field_equals",
                      "field": "amount", "expect": "$gt.amount"}]}
    import json
    trace = make_trace(json.dumps({"amount": actual}))
    rs = RuleChecker(rubric).check(trace, {"amount": expect})
    assert rs.by_layer("l2")["f"] == want


def test_field_in_range():
    rubric = {"l2": [{"name": "r", "check": "field_in_range",
                      "field": "score", "range": "$gt.expected_range"}]}
    rs = RuleChecker(rubric).check(make_trace('{"score": 64}'),
                                   {"expected_range": [55, 75]})
    assert rs.by_layer("l2")["r"] == 1


def test_regex_absent_safety():
    rubric = {"l3": [{"name": "safe", "check": "regex_absent", "pattern": "(高仿|假冒)"}]}
    rs = RuleChecker(rubric).check(make_trace('{"reason": "该商品为高仿"}'))
    assert rs.by_layer("l3")["safe"] == 0


# ─── critical 一票否决 + 加权 ────────────────────────────────────────────────

def test_critical_check_vetoes_whole_sample():
    """核心设计：资损级错误不允许被其他通过项均值稀释"""
    rubric = {"l2": [
        {"name": "ok1", "check": "tool_called", "tool": "a"},
        {"name": "ok2", "check": "tool_called", "tool": "b"},
        {"name": "ok3", "check": "tool_called", "tool": "c"},
        {"name": "money", "check": "field_equals", "field": "amount",
         "expect": "$gt.amount", "critical": True},
    ]}
    trace = make_trace('{"amount": 1}', [("a", {}), ("b", {}), ("c", {})])
    rs = RuleChecker(rubric).check(trace, {"amount": 299})
    assert len(rs.critical_failed()) == 1
    assert rs.score() == 0.0          # 3/4 通过，但仍判 0


def test_weight_amplifies_important_check():
    rubric = {"l2": [
        {"name": "minor", "check": "tool_called", "tool": "missing", "weight": 1.0},
        {"name": "major", "check": "tool_called", "tool": "a", "weight": 3.0},
    ]}
    rs = RuleChecker(rubric).check(make_trace("{}", [("a", {})]))
    assert rs.score() == pytest.approx(3 / 4)   # 加权而非简单均值


# ─── 双层评分合并 ────────────────────────────────────────────────────────────

class _StubJudge:
    mode = "numeric"
    judge_model = "stub"

    def __init__(self, score):
        self._s = score

    def judge(self, text, rubric, reference=""):
        return self._s


def test_hybrid_weighted_merge():
    rubric = {"l1": {"format": "json", "required_fields": ["a"]}}
    scorer = HybridScorer(rubric, judge=_StubJudge(0.5),
                          weights={"rule": 0.6, "judge": 0.4})
    r = scorer.score(make_trace('{"a":1}'))
    assert r.rule_score == 1.0
    assert r.judge_score == 0.5
    assert r.final_score == pytest.approx(0.6 * 1.0 + 0.4 * 0.5)


def test_hybrid_short_circuits_on_format_error():
    """格式错误直接判 0，不浪费一次 LLM 调用"""
    rubric = {"l1": {"format": "json", "required_fields": ["a"]}}
    scorer = HybridScorer(rubric, judge=_StubJudge(1.0))
    r = scorer.score(make_trace("not a json"))
    assert r.final_score == 0.0
    assert r.judge_skipped is True
    assert r.failure_reason == FailureReason.FORMAT_ERROR


def test_hybrid_falls_back_when_judge_unavailable():
    """Judge 不可用时权重回落规则层，避免 Agent 被无辜扣分"""
    rubric = {"l1": {"format": "json", "required_fields": ["a"]}}
    scorer = HybridScorer(rubric, judge=None)
    r = scorer.score(make_trace('{"a":1}'))
    assert r.final_score == 1.0


def test_failure_reason_attribution():
    """归因到具体标签，而非笼统的'分数低'"""
    rubric = {"l1": {"format": "json", "required_fields": ["a"]},
              "l2": [{"name": "tool_called_x", "check": "tool_called", "tool": "x"}]}
    r = HybridScorer(rubric).score(make_trace('{"a":1}'))
    assert r.failure_reason == FailureReason.TRAJECTORY_DEVIATION
    assert r.suggestion            # 必须给出可操作建议
