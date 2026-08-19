"""
修复回归测试：P0/P1 修复项的行为固化

覆盖：
  · AgentLoop 超步判定（耗尽=失败 / 主动收尾=成功 / 末轮收尾=成功）
  · mark_resolved 的 key 闭环（{task_id}::{sample_id} 可命中）
  · 规则层 layer_weights 分层权重（L1/L2/L3）
  · YAML 字符串布尔解析（critical: "false" 不再是 True）
  · 交叉校验仲裁（异常侧 -1 不拉低分 / 分歧取保守 / 一致取均值）
  · tool_param_equals 全调用校验（重复调用任一参数错即失败）
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import MagicMock, patch

from harness.sandbox import AgentTrace, MockSandbox
from harness.agent_loop import AgentLoop
from metrics.rule_checker import RuleChecker, _as_bool
from metrics.cross_validator import CrossCheckRecord
from dataset.tracer import GoldenDataset


# ─── AgentLoop 超步判定（P0）─────────────────────────────────────────────────

def _fake_response(content="", tool_calls=None):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    resp = MagicMock()
    resp.choices = [MagicMock(message=msg)]
    resp.usage.total_tokens = 10
    return resp


def _tool_call(name="query_order"):
    tc = MagicMock()
    tc.id = "call_1"
    tc.function.name = name
    tc.function.arguments = '{"order_id":"x"}'
    return tc


def _make_loop(max_steps=3):
    return AgentLoop(model="m", tools=[], strategy="function_calling",
                     max_steps=max_steps, api_key="sk-test")


def test_max_steps_exhausted_marks_failure():
    """P0：Agent 一直调工具直到步数耗尽 → 必须失败并归因 MAX_STEPS_EXCEEDED"""
    sb = MockSandbox(mock_apis={"query_order": {"ok": True}},
                     task_id="t", sample_id="s", agent_id="a")
    loop = _make_loop(max_steps=2)
    with patch.object(loop.client.chat.completions, "create",
                      side_effect=[_fake_response(tool_calls=[_tool_call()]),
                                   _fake_response(tool_calls=[_tool_call()])]):
        tr = loop.run("task", sb)
    assert tr.success is False
    assert tr.failure_reason == "MAX_STEPS_EXCEEDED"


def test_early_stop_is_success():
    """P0：Agent 主动输出最终答案 → 成功"""
    sb = MockSandbox(mock_apis={}, task_id="t", sample_id="s", agent_id="a")
    loop = _make_loop(max_steps=5)
    with patch.object(loop.client.chat.completions, "create",
                      side_effect=[_fake_response(content='{"done":true}')]):
        tr = loop.run("task", sb)
    assert tr.success is True
    assert tr.failure_reason is None


def test_final_step_answer_still_success():
    """P0 边界：最后一轮恰好收尾（无 tool_calls）→ 成功，不算超步"""
    sb = MockSandbox(mock_apis={}, task_id="t", sample_id="s", agent_id="a")
    loop = _make_loop(max_steps=2)
    with patch.object(loop.client.chat.completions, "create",
                      side_effect=[_fake_response(tool_calls=[_tool_call()]),
                                   _fake_response(content='{"done":true}')]):
        tr = loop.run("task", sb)
    assert tr.success is True


def test_react_strategy_executes_text_action_and_observation():
    """ReAct 不是只改 prompt：文本 Action 必须实际触发工具并收到 Observation。"""
    sb = MockSandbox(mock_apis={"query_order": {"amount": 299}}, task_id="t", sample_id="s", agent_id="a")
    loop = AgentLoop(model="m", tools=[{
        "type": "function", "function": {"name": "query_order", "parameters": {
            "type": "object", "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        }},
    }], strategy="react", max_steps=3, api_key="sk-test")
    with patch.object(loop.client.chat.completions, "create",
                      side_effect=[
                          _fake_response(content='Thought: 查订单\nAction: query_order\nAction Input: {"order_id":"x"}'),
                          _fake_response(content='Final Answer: {"done":true}'),
                      ]) as create:
        tr = loop.run("task", sb)
    assert tr.success is True
    assert tr.final_output == '{"done":true}'
    assert [s.tool_name for s in tr.steps] == ["query_order"]
    second_messages = create.call_args_list[1].kwargs["messages"]
    assert any("Observation:" in (m.get("content") or "") for m in second_messages)


def test_sandbox_validates_required_tool_parameters():
    sb = MockSandbox(mock_apis={"refund": {"ok": True}}, tool_definitions=[{
        "type": "function", "function": {"name": "refund", "parameters": {
            "type": "object", "properties": {"amount": {"type": "number"}},
            "required": ["amount"],
        }},
    }])
    result = sb.call_tool("refund", {})
    assert "ToolParamValidationError" in result["error"]
    assert sb.get_trace().steps[0].error is not None


# ─── mark_resolved key 闭环（P1）─────────────────────────────────────────────

def test_mark_resolved_key_match(tmp_path):
    """P1：ingest 后 mark_resolved({task_id}::{sample_id}) 必须命中"""
    gd = GoldenDataset(str(tmp_path / "g.jsonl"))
    t = AgentTrace(task_id="content_review_001", sample_id="easy_001", agent_id="a")
    t.final_output = '{"x":1}'
    t.eval_result = {"overall_score": 0.2, "failure_reason": "WRONG_CHOICE",
                     "rule_detail": {"failed": [{"name": "c1", "detail": "d"}]},
                     "suggestion": "改 prompt"}
    t.failure_reason = "WRONG_CHOICE"
    gd.ingest([t], sample_map={"easy_001": {"input": {}, "ground_truth": {}}})
    n = gd.mark_resolved({"content_review_001::easy_001"})
    assert n == 1
    assert gd.stats()["resolve_rate"] == 1.0


# ─── 规则层分层权重（P1）─────────────────────────────────────────────────────

def test_layer_weights_scoring():
    """P1：layer_weights 按 L1/L2/L3 分层加权；l3 全挂 → 0.2*1+0.6*1+0.2*0=0.8"""
    rubric = {"l1": {"format": "json", "required_fields": ["a"]},
              "l2": [{"name": "ok", "check": "tool_called", "tool": "q"}],
              "l3": [{"name": "bad", "check": "field_non_empty", "field": "reason"}]}
    rc = RuleChecker(rubric, layer_weights={"l1": 0.2, "l2": 0.6, "l3": 0.2})
    tr = AgentTrace(task_id="t", sample_id="s", agent_id="a")
    tr.add_step("q", {}, {})
    tr.final_output = '{"a":1, "reason":""}'
    assert rc.check(tr).score() == pytest.approx(0.8)


def test_default_scoring_backward_compatible():
    """P1：未配置 layer_weights 时保持全 check 加权平均（向后兼容）"""
    rubric = {"l1": {"format": "json", "required_fields": ["a"]},
              "l2": [{"name": "ok", "check": "tool_called", "tool": "q"}],
              "l3": [{"name": "bad", "check": "field_non_empty", "field": "reason"}]}
    tr = AgentTrace(task_id="t", sample_id="s", agent_id="a")
    tr.add_step("q", {}, {})
    tr.final_output = '{"a":1, "reason":""}'
    assert RuleChecker(rubric).check(tr).score() == pytest.approx(0.75)


# ─── YAML 字符串布尔解析（低风险项）──────────────────────────────────────────

@pytest.mark.parametrize("raw,want", [
    ("false", False), ("true", True), ("0", False), ("1", True),
    ("no", False), ("yes", True), ("off", False), ("on", True),
])
def test_as_bool_string(raw, want):
    assert _as_bool(raw) is want


def test_as_bool_types():
    assert _as_bool(True) is True
    assert _as_bool(False) is False
    assert _as_bool(None) is False
    assert _as_bool(1) is True


def test_critical_string_false_is_not_critical():
    """低风险项：YAML critical: "false" 之前会被 bool("false") 转成 True"""
    rubric = {"l2": [{"name": "c", "check": "field_equals", "field": "a",
                      "expect": "$gt.a", "critical": "false"}]}
    tr = AgentTrace(task_id="t", sample_id="s", agent_id="a")
    tr.final_output = '{"a": 2}'
    rs2 = RuleChecker(rubric).check(tr, {"a": 1})
    assert rs2.critical_failed() == []          # "false" 不再触发一票否决
    assert rs2.failed_checks()[0].critical is False


# ─── 交叉校验仲裁（P1）───────────────────────────────────────────────────────

def test_cross_record_uses_valid_scores_only():
    """P1：单侧异常（-1）不拉低仲裁分，采信有效侧"""
    r = CrossCheckRecord("s1", score_a=-1.0, score_b=0.8,
                         model_a="A", model_b="B")
    assert r.valid is False
    assert r.merged_score == 0.8


def test_cross_record_dispute_takes_conservative():
    """P1：分歧时取保守值（较低分），避免高估 Agent 能力"""
    r = CrossCheckRecord("s2", score_a=0.9, score_b=0.4,
                         model_a="A", model_b="B")
    assert r.agreed is False
    assert r.merged_score == 0.4


def test_cross_record_agreement_takes_mean():
    r = CrossCheckRecord("s3", score_a=0.8, score_b=0.9,
                         model_a="A", model_b="B")
    assert r.agreed is True
    assert r.merged_score == pytest.approx(0.85)


# ─── tool_param_equals 全调用校验（低风险项）─────────────────────────────────

def test_tool_param_checks_all_calls():
    """低风险项：工具被调用两次、任一次参数错 → 失败（副作用工具重复调用即风险）"""
    rubric = {"l2": [{"name": "amt", "check": "tool_param_equals",
                      "tool": "refund", "param": "amount", "expect": "$gt.amount"}]}
    tr = AgentTrace(task_id="t", sample_id="s", agent_id="a")
    tr.add_step("refund", {"amount": 299.0}, {})
    tr.add_step("refund", {"amount": 199.0}, {})
    rs = RuleChecker(rubric).check(tr, {"amount": 299.0})
    assert rs.by_layer("l2")["amt"] == 0
    assert "1/2" in rs.failed_checks()[0].detail
