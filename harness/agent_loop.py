"""
AgentLoop：多轮对话主循环

职责：
  1. 驱动 LLM 多轮调用
  2. 通过 MockSandbox 拦截工具调用
  3. 将完整消息历史写入 AgentTrace
  4. 统计 token 消耗

Strategy 插拔（function_calling / react），AgentLoop 主体不变。
"""
from __future__ import annotations

import os
import time
import json
from typing import Any

from openai import OpenAI

from harness.sandbox import MockSandbox, AgentTrace
from strategies.base import get_strategy
import strategies.function_calling  # noqa: F401
import strategies.react  # noqa: F401

# 文本协议（ReAct）格式漂移的连续纠错上限。
# 参考 LangChain AgentExecutor 惯例：解析失败以 Observation 形式回灌模型重试，
# 但重试次数必须有限，防止模型在错误格式上死循环烧 token。
_MAX_FORMAT_RETRIES = 2


class AgentLoop:
    """
    Args:
        model:    被测 Agent 使用的 LLM 模型名
        tools:    OpenAI tools 格式的工具定义列表
        strategy: "function_calling" | "react"
        max_steps: 最大工具调用轮数
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        tools: list[dict] | None = None,
        strategy: str = "function_calling",
        max_steps: int = 10,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self.model = model
        self.tools = tools or []
        self.max_steps = max_steps
        self.strategy = get_strategy(strategy)()

        # 惰性创建 OpenAI 客户端：构造时不强制要求 API Key（装配/测试阶段可先
        # 实例化 loop），首次 run() 真正调用 LLM 时才创建——Key 缺失在调用时
        # 才报错，也便于测试注入 fake client。
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def run(self, task_prompt: str, sandbox: MockSandbox) -> AgentTrace:
        """执行一次完整的 Agent 运行，返回 AgentTrace"""
        trace = sandbox.get_trace()
        task_prompt = self.strategy.prepare_prompt(task_prompt, self.tools)
        messages = self.strategy.build_messages(task_prompt, [])
        total_tokens = 0
        t_total_start = time.perf_counter()

        completed = False
        format_failures = 0            # 连续格式纠错失败的轮数（不静默丢步的依据）
        for step in range(self.max_steps):
            kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
            if self.tools and self.strategy.uses_native_tools():
                kwargs["tools"] = self.tools
                kwargs["tool_choice"] = "auto"

            response = self.client.chat.completions.create(**kwargs)
            msg = response.choices[0].message
            total_tokens += response.usage.total_tokens if response.usage else 0

            msg_dict = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(msg_dict)

            # 仅当 Agent 主动收尾（无 tool_calls / Final Answer）才算成功完成；
            # 步数耗尽由循环结束后的 completed 标志判定，策略层不再把 max_steps 当"成功"。
            if self.strategy.should_stop(msg_dict, step + 1, self.max_steps):
                trace.final_output = msg.content or ""
                if not self.strategy.uses_native_tools() and "Final Answer:" in trace.final_output:
                    trace.final_output = trace.final_output.split("Final Answer:", 1)[1].strip()
                trace.success = True
                completed = True
                break

            action = self.strategy.extract_action(msg_dict)
            if action:
                format_failures = 0    # 正常行动，重置纠错计数
                tool_name, params = action
                response = sandbox.call_tool(tool_name, params)
                messages.append({
                    "role": "user",
                    "content": "Observation: " + json.dumps(response, ensure_ascii=False)
                               + "\n请继续执行 Thought / Action，或给出 Final Answer。",
                })
            elif msg.tool_calls:
                format_failures = 0
                tool_msgs = sandbox.parse_tool_calls(msg.tool_calls)
                messages.extend(tool_msgs)
            else:
                # 文本协议格式漂移：把解析错误回灌给模型自我修正（不静默丢步）；
                # 连续失败达到上限则提前终止，避免死循环烧 token。
                # 注意：最后一次失败的反馈也要先落进 messages（trace 留痕），再 break。
                feedback = self.strategy.format_error_feedback(msg_dict)
                if feedback:
                    format_failures += 1
                    messages.append({"role": "user", "content": feedback})
                    if format_failures >= _MAX_FORMAT_RETRIES:
                        break

        trace.messages = messages
        trace.total_tokens = total_tokens
        trace.total_latency_ms = (time.perf_counter() - t_total_start) * 1000

        # 循环自然结束 = 步数耗尽：无论最后一条消息是否有内容都不得视为成功
        if not completed:
            trace.final_output = messages[-1].get("content", "") if messages else ""
            trace.success = False
            # 连续格式纠错失败 → 归因 FORMAT_ERROR（明确的失败信号，区别于步数耗尽）；
            # 否则为 MAX_STEPS_EXCEEDED
            trace.failure_reason = (
                "FORMAT_ERROR" if format_failures >= _MAX_FORMAT_RETRIES
                else "MAX_STEPS_EXCEEDED"
            )

        return trace
