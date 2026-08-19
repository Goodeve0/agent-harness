"""
代码审查修复回归测试（2026-08-19，两轮合并）

第一轮（历史遗留，P0/P1 修复项行为固化）：
  · AgentLoop 超步判定（耗尽=失败 / 主动收尾=成功 / 末轮收尾=成功）
  · mark_resolved 的 key 闭环（{task_id}::{sample_id} 可命中）
  · 规则层 layer_weights 分层权重（L1/L2/L3）与默认口径向后兼容
  · YAML 字符串布尔解析（critical: "false" 不再是 True）
  · 交叉校验仲裁（异常侧 -1 不拉低分 / 分歧取保守 / 一致取均值）
  · tool_param_equals 全调用校验（重复调用任一参数错即失败）

第二轮（2026-08-19 全量审查，P1/P2 修复项行为固化）：
  · P1-1  CrossValidator.report() 空记录返回结构完整 dict，不崩主流程（fail-open）
  · P1-2  AgentTrace.to_dict() 导出完整对话 messages，Trace 可回放
  · P2-2  LLMJudge._parse_numeric 分制推断：8分/85分/漏小数点/计数词不伪造
  · P2-3  ReAct Action Input 嵌套 JSON 提取（栈式花括号，非贪婪正则不再截断）
  · P2-1  ReAct 格式漂移产生失败信号：format_error_feedback + AgentLoop 纠错循环
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import MagicMock, patch

from harness.sandbox import AgentTrace, MockSandbox
from harness.agent_loop import AgentLoop, _MAX_FORMAT_RETRIES
from metrics.judge import LLMJudge
from metrics.cross_validator import CrossValidator, CrossCheckRecord
from metrics.rule_checker import RuleChecker, _as_bool
from dataset.tracer import GoldenDataset
from strategies.react import ReActStrategy, _extract_json_object


# ─── P1-1：交叉校验空记录 fail-open ─────────────────────────────────────────

class _StubJudge:
    """不真正调用，模拟不可用 judge（judge 抛异常 → -1）"""

    def __init__(self, name: str):
        self.judge_model = name

    def judge(self, text, rubric, reference=""):
        raise RuntimeError("CLI 不可用")


def test_cross_validator_empty_report_has_all_keys():
    """两侧 CLI 都不可用时 report() 必须返回结构完整 dict（fail-open 不崩主流程）"""
    cv = CrossValidator(judge_a=_StubJudge("claude-cli"),
                        judge_b=_StubJudge("codex-cli"))
    rec = cv.validate("s1", text="t", rubric="r")
    assert not rec.valid                      # -1/-1 → 无效记录

    cr = cv.report()
    # run_eval.py 直接索引这些键，缺任何一个都会 KeyError
    for key in ("total", "model_a", "model_b", "agreement_rate", "mean_abs_diff",
                "mean_score_a", "mean_score_b", "disputed_count",
                "disputed_samples", "verdict"):
        assert key in cr, f"空报告缺少键: {key}"
    assert cr["total"] == 0
    assert cr["agreement_rate"] == 0.0
    assert cr["mean_abs_diff"] == 0.0
    assert cr["disputed_count"] == 0
    assert cr["disputed_samples"] == []
    assert cr["model_a"] == "claude-cli"      # 仍能标识两侧模型
    assert cr["verdict"]                      # 走"一致率偏低"提示路径


def test_cross_validator_empty_report_indexing_safe():
    """模拟 run_eval.py 的报告消费路径：任何情况下都不该抛 KeyError"""
    cv = CrossValidator(judge_a=_StubJudge("a"), judge_b=_StubJudge("b"))
    cv.validate("s1", text="t", rubric="r")
    cr = cv.report()
    _ = f"一致率: {cr['agreement_rate']:.1%}  平均分差: {cr['mean_abs_diff']:.3f}  " \
        f"争议样本: {cr['disputed_count']}  {cr['verdict']}"


# ─── P1-2：Trace 导出完整对话 ───────────────────────────────────────────────

def test_trace_to_dict_contains_full_messages():
    """完整多轮对话必须随 to_dict 落盘，保证 JSONL / golden trace_snapshot 可回放"""
    t = AgentTrace(task_id="t1", sample_id="s1", agent_id="a1")
    t.messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "任务输入"},
        {"role": "assistant", "content": "Thought: 查订单", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "query_order", "arguments": '{"order_id": "o1"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": '{"amount": 299}'},
        {"role": "assistant", "content": "Final Answer: 完成"},
    ]
    t.add_step("query_order", {"order_id": "o1"}, {"amount": 299})
    t.final_output = "完成"

    d = t.to_dict()
    assert d["messages"] == t.messages          # 对话历史逐条保留
    assert len(d["steps"]) == 1                 # 工具调用与对话互不丢失
    assert d["messages"][-1]["role"] == "assistant"


def test_trace_jsonl_roundtrip_with_messages(tmp_path):
    """dump_traces 落盘 JSONL 后，messages 仍可 JSON 反序列化回放"""
    import json
    from dataset.tracer import Tracer
    t = AgentTrace(task_id="t1", sample_id="s1", agent_id="a1")
    t.messages = [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好"}]
    tr = Tracer(trace_dir=str(tmp_path))
    tr.add(t)
    path = tr.dump_traces("run_test")

    line = json.loads(Path(path).read_text().strip().splitlines()[0])
    assert line["messages"][1]["content"] == "你好"


# ─── P2-2：数值解析分制推断 ─────────────────────────────────────────────────

def _numeric_judge() -> LLMJudge:
    """绕过 __init__ 的 OpenAI client 构造，只测纯解析逻辑"""
    return object.__new__(LLMJudge)


@pytest.mark.parametrize("raw,want", [
    ("0.8", 0.8),                     # 标准 0~1
    ("8/10", 0.8),                    # 分数表达式
    ("0.8/1", 0.8),
    ("85%", 0.85),                    # 百分比
    ("8分", 0.8),                     # 10 分制
    ("85分", 0.85),                   # 百分制
    ("我打 8 分（满分 10）", 0.8),      # 中文分制上下文
    ("8.5 分", 0.85),
    ("85", 0.85),                     # 漏写小数点（注释原意图）
    ("8", 0.8),                       # 10 分制推断
    ("0.8 分", 0.8),                  # <=1 的"分"不应被 /10
    ("总分 0.75，理由略", 0.75),
    ("", 0.0),                        # 空输出
    ("无法判断", 0.0),                 # 无数字
    ("出错了 3 个维度", 0.0),           # 计数场景：不伪造分数
])
def test_parse_numeric_robust(raw, want):
    assert _numeric_judge()._parse_numeric(raw) == pytest.approx(want)


def test_parse_numeric_no_dead_code_branch():
    """兜底分支每个路径都返回值（原实现存在永远走不到的 return v 死代码）"""
    j = _numeric_judge()
    # 覆盖第 5 步三个分支：<=10 / <=100 / >100
    assert j._parse_numeric("8") == pytest.approx(0.8)
    assert j._parse_numeric("85") == pytest.approx(0.85)
    assert j._parse_numeric("123") == 1.0
    assert j._parse_numeric("123分") == 1.0


# ─── P2-3：ReAct 嵌套 JSON 提取 ─────────────────────────────────────────────

def test_extract_json_object_nested():
    """嵌套对象 / 数组 / 字符串内花括号都要完整提取"""
    text = '{"a": {"b": [1, 2]}, "c": "}} 是右括号", "d": {"e": {"f": 1}}} 尾随文本'
    obj = _extract_json_object(text)
    assert obj == {"a": {"b": [1, 2]}, "c": "}} 是右括号", "d": {"e": {"f": 1}}}


def test_extract_json_object_string_with_escaped_quotes():
    text = r'{"q": "他说 \"好\"", "n": 1} extra'
    assert _extract_json_object(text) == {"q": '他说 "好"', "n": 1}


def test_extract_json_object_no_json():
    assert _extract_json_object("没有花括号") is None
    assert _extract_json_object('{"a": 1') is None      # 未闭合


def test_react_extract_action_nested_json():
    """嵌套 JSON 的 Action Input 不应被非贪婪正则截断成解析失败"""
    s = ReActStrategy()
    action = s.extract_action({"role": "assistant", "content":
        "Thought: 需要查订单\n"
        'Action: query_order\n'
        'Action Input: {"order_id": "o1", "filters": {"status": ["paid", "refunded"]},'
        ' "note": "含 } 的备注"}'
    })
    assert action is not None
    name, params = action
    assert name == "query_order"
    assert params == {"order_id": "o1",
                      "filters": {"status": ["paid", "refunded"]},
                      "note": "含 } 的备注"}


def test_react_extract_action_multiline_json():
    s = ReActStrategy()
    content = ("Thought: 先查\n"
               "Action: query_order\n"
               "Action Input: {\n"
               '  "order_id": "o1",\n'
               '  "amount": 299\n'
               "}\n"
               "Thought: 继续")
    assert s.extract_action({"role": "assistant", "content": content}) == \
        ("query_order", {"order_id": "o1", "amount": 299})


def test_react_extract_action_final_answer_is_none():
    s = ReActStrategy()
    assert s.extract_action({"role": "assistant", "content": "Final Answer: 完成"}) is None
    assert s.format_error_feedback({"role": "assistant", "content": "Final Answer: 完成"}) is None


# ─── P2-1：格式漂移失败信号 ─────────────────────────────────────────────────

def test_react_format_error_missing_input():
    """有 Action 无 Action Input → extract None + 明确纠错反馈（不再静默丢步）"""
    s = ReActStrategy()
    msg = {"role": "assistant", "content": "Thought: 查一下\nAction: query_order"}
    assert s.extract_action(msg) is None
    feedback = s.format_error_feedback(msg)
    assert feedback is not None
    assert "Action Input" in feedback


def test_react_format_error_invalid_json():
    s = ReActStrategy()
    msg = {"role": "assistant", "content":
           "Thought: 查\nAction: query_order\nAction Input: {order_id: o1}"}  # 非合法 JSON
    assert s.extract_action(msg) is None
    feedback = s.format_error_feedback(msg)
    assert feedback is not None
    assert "JSON" in feedback


def test_react_plain_text_no_feedback():
    """纯文本中间回复（无 Action 标记）不属于格式错误，不打断循环"""
    s = ReActStrategy()
    msg = {"role": "assistant", "content": "让我先分析一下这个任务。"}
    assert s.extract_action(msg) is None
    assert s.format_error_feedback(msg) is None


# ─── AgentLoop 纠错循环集成 ─────────────────────────────────────────────────

class _FakeMsg:
    def __init__(self, content: str):
        self.content = content
        self.tool_calls = None


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMsg(content)


class _FakeResp:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]
        self.usage = SimpleNamespace(total_tokens=10)


class _FakeCompletions:
    def __init__(self, contents: list[str]):
        self._it = iter(contents)

    def create(self, **kwargs):
        return _FakeResp(next(self._it))


class _FakeChat:
    def __init__(self, contents):
        self.completions = _FakeCompletions(contents)


class _FakeClient:
    def __init__(self, contents):
        self.chat = _FakeChat(contents)


def _run_loop(contents: list[str], mock_apis: dict | None = None,
              max_steps: int = 10) -> AgentTrace:
    loop = AgentLoop(model="fake", strategy="react", max_steps=max_steps)
    sandbox = MockSandbox(mock_apis=mock_apis or {}, task_id="t1", agent_id="a1")
    with patch("harness.agent_loop.OpenAI", return_value=_FakeClient(contents)):
        return loop.run("任务", sandbox)


def test_loop_recovers_from_format_error():
    """格式错误 → 回灌纠错 → 模型修正 → 正常完成（纠错机制有效）"""
    trace = _run_loop([
        "Thought: 查一下\nAction: query_order",            # 缺 Action Input（格式错误）
        "Thought: 修正\nAction: search\nAction Input: {\"q\": \"x\"}",
        "Observation: ... 已经拿到结果\nFinal Answer: 完成",
    ], mock_apis={"search": {"ok": True}})
    assert trace.success is True
    assert trace.failure_reason is None
    # 纠错反馈确实被回灌进对话
    assert any(m.get("role") == "user" and "Action Input" in m.get("content", "")
               for m in trace.messages)


def test_loop_marks_format_error_after_retries():
    """连续格式失败达到上限 → 明确归因 FORMAT_ERROR（不再静默空转/误报步数耗尽）"""
    bad = "Thought: 查\nAction: query_order\nAction Input: {bad json}"
    contents = [bad] * (_MAX_FORMAT_RETRIES + 1)
    trace = _run_loop(contents, max_steps=10)
    assert trace.success is False
    assert trace.failure_reason == "FORMAT_ERROR"
    # 纠错反馈已回灌（_MAX_FORMAT_RETRIES 次），而非 0 次静默吞掉
    feedback_count = sum(1 for m in trace.messages
                         if m.get("role") == "user" and "Observation:" in m.get("content", "")
                         and "格式错误" in m.get("content", ""))
    assert feedback_count == _MAX_FORMAT_RETRIES


def test_loop_step_exhaustion_still_max_steps():
    """格式正常但步数耗尽 → 仍归因 MAX_STEPS_EXCEEDED（两路径不混淆）"""
    trace = _run_loop(["让我想想"] * 5, max_steps=3)
    assert trace.success is False
    assert trace.failure_reason == "MAX_STEPS_EXCEEDED"


# ─── 第一轮回归：AgentLoop 超步判定（P0）────────────────────────────────────

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


def _make_loop(max_steps=3, strategy="function_calling", tools=None):
    return AgentLoop(model="m", tools=tools or [], strategy=strategy,
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
    loop = _make_loop(max_steps=3, strategy="react", tools=[{
        "type": "function", "function": {"name": "query_order", "parameters": {
            "type": "object", "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        }},
    }])
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


# ─── 第一轮回归：mark_resolved key 闭环（P1）────────────────────────────────

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


# ─── 第一轮回归：规则层分层权重（P1）────────────────────────────────────────

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


# ─── 第一轮回归：YAML 字符串布尔解析（低风险项）─────────────────────────────

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


# ─── 第一轮回归：交叉校验仲裁（P1）──────────────────────────────────────────

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


# ─── 第一轮回归：tool_param_equals 全调用校验（低风险项）────────────────────

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
