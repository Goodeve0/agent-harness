"""
FunctionCalling Strategy —— 标准 OpenAI tool_calls 格式

为什么这个文件"这么薄"（~40 行 vs react.py 130 行）：
  原生 function calling 的协议解析（tool_calls 字段、tool 消息回填、参数 JSON
  解析）由 OpenAI API 与 AgentLoop 直接承担（见 harness/agent_loop.py 的
  `msg.tool_calls` 分支 + MockSandbox.parse_tool_calls）。策略层只需回答三个问题：
    1. system_prompt   —— 行为约束（何时调工具、何时收尾、输出格式）
    2. build_messages  —— 如何组装首轮消息 / 续接历史
    3. should_stop     —— 模型不再请求工具调用即视为任务完成
  文本协议策略（react.py）必须自行实现 Action / Action Input 解析与格式纠错，
  因此体量大得多。这是两种策略的本质差异，不是功能缺失。
"""
from __future__ import annotations

from strategies.base import BaseStrategy, register_strategy


@register_strategy("function_calling")
class FunctionCallingStrategy(BaseStrategy):
    """
    走 OpenAI 原生 tool_calls：
      - uses_native_tools() = True（base 默认实现），AgentLoop 据此把 tools
        直接传给 chat.completions，模型返回结构化 tool_calls；
      - 工具结果以 role=tool 消息回灌，模型自行继续推理或收尾；
      - 模型输出无 tool_calls → should_stop True → 任务完成。
    """

    def system_prompt(self) -> str:
        return (
            "你是一个能力强大的 AI 助手，可以调用工具完成用户任务。\n"
            "严格按工具定义传递参数：必填字段齐全、类型正确、取值有依据（来自工具"
            "返回结果或用户输入），不得臆造。\n"
            "工具返回结果后继续推理，直到任务完成；不要声称已调用工具——只有真实"
            "发起调用并收到结果后，才能把该结果写入最终输出。\n"
            "任务完成时：若任务要求 JSON 输出，只输出 JSON 对象，不要附加任何解释文字。"
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
