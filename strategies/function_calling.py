"""
FunctionCalling Strategy —— 标准 OpenAI tool_calls 格式

Agent 通过 function calling 调用工具，MockSandbox 拦截并记录。
"""
from __future__ import annotations

from strategies.base import BaseStrategy, register_strategy


@register_strategy("function_calling")
class FunctionCallingStrategy(BaseStrategy):

    def system_prompt(self) -> str:
        return (
            "你是一个能力强大的 AI 助手，可以调用工具完成用户任务。\n"
            "严格按照工具定义传递参数，工具返回结果后继续推理直到任务完成。\n"
            "任务完成后，输出最终结果并停止调用工具。"
        )

    def build_messages(self, task_prompt: str, history: list[dict]) -> list[dict]:
        if not history:
            return [
                {"role": "system", "content": self.system_prompt()},
                {"role": "user", "content": task_prompt},
            ]
        return history  # 续接历史

    def should_stop(self, response_message: dict, step: int, max_steps: int) -> bool:
        # 注意：超步判断由 AgentLoop 负责（步数耗尽 = 失败），此处不返回 True 兜底
        tool_calls = response_message.get("tool_calls")
        return not tool_calls
