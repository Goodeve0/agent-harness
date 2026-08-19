"""
MockSandbox：工具调用拦截 + 全链路 Trace 记录

核心职责：
  1. 注册 mock_apis，拦截 Agent 的工具调用，返回预设结果，不依赖真实 API
  2. 记录每步 ToolCall 到 AgentTrace，保证评测可复现
  3. 提供 parse_tool_calls 接口，供 AgentLoop 批量执行工具调用
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """单次工具调用记录"""
    step: int
    tool_name: str
    params: dict
    response: Any
    latency_ms: float
    is_mock: bool = True
    error: str | None = None


@dataclass
class AgentTrace:
    """
    完整 Agent 执行轨迹：Prompt → Expected Behavior → Trace 三元组中的 Trace 部分。
    记录工具调用、推理、中间结果，作为评测与归因的基础数据。
    """
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    task_id: str = ""
    sample_id: str = ""
    agent_id: str = ""
    prompt_version: str = "v1"

    steps: list[ToolCall] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)

    final_output: str = ""
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    success: bool = False

    eval_result: dict | None = None
    failure_reason: str | None = None
    is_bad_case: bool = False

    def add_step(self, tool_name: str, params: dict, response: Any,
                 latency_ms: float = 0.0, error: str | None = None) -> ToolCall:
        step = ToolCall(
            step=len(self.steps) + 1,
            tool_name=tool_name,
            params=params,
            response=response,
            latency_ms=latency_ms,
            error=error,
        )
        self.steps.append(step)
        return step

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "task_id": self.task_id,
            "sample_id": self.sample_id,
            "agent_id": self.agent_id,
            "prompt_version": self.prompt_version,
            "steps": [
                {
                    "step": s.step,
                    "tool": s.tool_name,
                    "params": s.params,
                    "response": s.response,
                    "latency_ms": round(s.latency_ms, 2),
                    "error": s.error,
                }
                for s in self.steps
            ],
            "final_output": self.final_output,
            "total_tokens": self.total_tokens,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "success": self.success,
            "eval_result": self.eval_result,
            "failure_reason": self.failure_reason,
            "is_bad_case": self.is_bad_case,
        }


class MockSandbox:
    """
    工具调用拦截沙箱。

    - mock_apis: {"tool_name": response_value_or_callable}
                 callable 收到 params dict，返回 response
    - 未注册的工具调用会被拒绝，避免误触真实 API
    - 扩展方向：docker 沙箱隔离，支持带副作用的工具调用
    """

    def __init__(self, mock_apis: dict[str, Any], task_id: str = "",
                 sample_id: str = "", agent_id: str = "",
                 prompt_version: str = "v1", tool_definitions: list[dict] | None = None):
        self._mock_apis = {name: _resolve_mock(v) for name, v in mock_apis.items()}
        self._tool_schemas = {
            t.get("function", {}).get("name"): t.get("function", {}).get("parameters", {})
            for t in (tool_definitions or [])
            if t.get("function", {}).get("name")
        }
        self.trace = AgentTrace(
            task_id=task_id,
            sample_id=sample_id,
            agent_id=agent_id,
            prompt_version=prompt_version,
        )

    def call_tool(self, tool_name: str, params: dict) -> Any:
        t0 = time.perf_counter()
        error = None
        response = None

        if tool_name not in self._mock_apis:
            error = f"ToolNotFound: {tool_name}"
            response = {"error": error}
        elif validation_error := self._validate_params(tool_name, params):
            error = validation_error
            response = {"error": error}
        else:
            mock = self._mock_apis[tool_name]
            try:
                response = mock(params) if callable(mock) else mock
            except Exception as e:
                error = str(e)
                response = {"error": error}

        latency_ms = (time.perf_counter() - t0) * 1000
        self.trace.add_step(
            tool_name=tool_name,
            params=params,
            response=response,
            latency_ms=latency_ms,
            error=error,
        )
        return response

    def _validate_params(self, tool_name: str, params: dict) -> str | None:
        """校验声明的 required 字段和基础 JSON 类型，失败时阻止 mock 副作用。"""
        schema = self._tool_schemas.get(tool_name)
        if not schema:
            return None
        if not isinstance(params, dict):
            return "ToolParamValidationError: 参数必须是 JSON 对象"
        missing = [key for key in schema.get("required", []) if key not in params]
        if missing:
            return f"ToolParamValidationError: 缺少必填参数 {missing}"
        type_map = {"string": str, "number": (int, float), "integer": int, "boolean": bool,
                    "object": dict, "array": list}
        for key, prop in schema.get("properties", {}).items():
            if key not in params or "type" not in prop:
                continue
            expected = type_map.get(prop["type"])
            value = params[key]
            # bool 是 int 子类，不能被 integer / number 误接收。
            if expected and (not isinstance(value, expected)
                             or (prop["type"] in ("integer", "number") and isinstance(value, bool))):
                return (f"ToolParamValidationError: {key} 应为 {prop['type']}，"
                        f"实际为 {type(value).__name__}")
        return None

    def parse_tool_calls(self, openai_tool_calls: list) -> list[dict]:
        """解析 OpenAI function_call 格式，批量执行工具调用，返回 tool messages"""
        results = []
        for tc in openai_tool_calls:
            tool_name = tc.function.name
            try:
                params = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                params = {}
            response = self.call_tool(tool_name, params)
            results.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tool_name,
                "content": json.dumps(response, ensure_ascii=False),
            })
        return results

    def get_trace(self) -> AgentTrace:
        return self.trace

    def reset(self, task_id: str = "", sample_id: str = "",
              agent_id: str = "", prompt_version: str = "v1"):
        self.trace = AgentTrace(
            task_id=task_id,
            sample_id=sample_id,
            agent_id=agent_id,
            prompt_version=prompt_version,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  条件 mock 解析：让 YAML 声明式 mock 能根据入参返回不同结果
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_mock(spec: Any) -> Any:
    """
    把 mock_apis 的声明式 value 解析为可执行对象。

    支持两种形式：
      1. callable / 普通值：原样返回
      2. dict 含 _conditional：转为 callable，按规则匹配入参后返回

    _conditional 结构：
      _conditional:
        - if: {content_contains_any: ["高仿", "假冒"]}
          return: {...}
        - default: {...}
    """
    if not isinstance(spec, dict) or "_conditional" not in spec:
        return spec

    rules = spec["_conditional"]
    default = None
    for r in rules:
        if "default" in r:
            default = r["default"]

    def _matcher(params: dict) -> Any:
        for r in rules:
            cond = r.get("if")
            if cond is None:
                continue
            if _match_condition(cond, params):
                return r.get("return", {})
        return default if default is not None else {}

    return _matcher


def _match_condition(cond: dict, params: dict) -> bool:
    """支持 content_contains_any：params['content'] 含任一关键词即匹配"""
    keywords = cond.get("content_contains_any")
    if keywords is None:
        return False
    text = str(params.get("content", ""))
    return any(kw in text for kw in keywords)
