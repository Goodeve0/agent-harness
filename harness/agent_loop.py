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

        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        )

    def run(self, task_prompt: str, sandbox: MockSandbox) -> AgentTrace:
        """执行一次完整的 Agent 运行，返回 AgentTrace"""
        trace = sandbox.get_trace()
        task_prompt = self.strategy.prepare_prompt(task_prompt, self.tools)
        messages = self.strategy.build_messages(task_prompt, [])
        total_tokens = 0
        t_total_start = time.perf_counter()

        completed = False
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
                tool_name, params = action
                response = sandbox.call_tool(tool_name, params)
                messages.append({
                    "role": "user",
                    "content": "Observation: " + json.dumps(response, ensure_ascii=False)
                               + "\n请继续执行 Thought / Action，或给出 Final Answer。",
                })
            elif msg.tool_calls:
                tool_msgs = sandbox.parse_tool_calls(msg.tool_calls)
                messages.extend(tool_msgs)

        trace.messages = messages
        trace.total_tokens = total_tokens
        trace.total_latency_ms = (time.perf_counter() - t_total_start) * 1000

        # 循环自然结束 = 步数耗尽：无论最后一条消息是否有内容都不得视为成功
        if not completed:
            trace.final_output = messages[-1].get("content", "") if messages else ""
            trace.success = False
            trace.failure_reason = "MAX_STEPS_EXCEEDED"

        return trace
