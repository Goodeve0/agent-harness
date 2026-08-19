"""
ReAct Strategy —— THOUGHT → ACT → OBSERVE 循环
用于推理密集型任务，每步需要显式思考再行动。
"""
from __future__ import annotations

import json
import re

from strategies.base import BaseStrategy, register_strategy

_REACT_SYSTEM = """你是一个使用 ReAct 框架的 AI 助手。每一步按以下格式输出：

Thought: 分析当前状态，决定下一步行动
Action: 调用哪个工具（或 Final Answer 表示任务完成）
Observation: [工具调用后自动填充]

当任务完成时，输出：
Final Answer: <最终结果>
"""


@register_strategy("react")
class ReActStrategy(BaseStrategy):

    def system_prompt(self) -> str:
        return _REACT_SYSTEM

    def build_messages(self, task_prompt: str, history: list[dict]) -> list[dict]:
        if not history:
            return [
                {"role": "system", "content": self.system_prompt()},
                {"role": "user", "content": task_prompt},
            ]
        return history

    def prepare_prompt(self, task_prompt: str, tools: list[dict]) -> str:
        """把工具 schema 注入 ReAct 文本协议，避免依赖 OpenAI tool_calls。"""
        catalog = []
        for tool in tools:
            func = tool.get("function", {})
            if func.get("name"):
                catalog.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {}),
                })
        return (
            f"{task_prompt}\n\n可用工具（只能从中选择）：\n"
            f"{json.dumps(catalog, ensure_ascii=False)}\n\n"
            "调用工具时严格输出：\nAction: <工具名>\nAction Input: <JSON 对象>"
        )

    def uses_native_tools(self) -> bool:
        return False

    def extract_action(self, response_message: dict) -> tuple[str, dict] | None:
        """解析 ReAct 文本 Action，供 AgentLoop 执行并回填 Observation。"""
        content = response_message.get("content", "") or ""
        if "Final Answer:" in content:
            return None
        match = re.search(
            r"(?mi)^Action:\s*([A-Za-z_][A-Za-z0-9_]*)\s*$"
            r"\s*^Action Input:\s*(\{.*?\})(?=\s*$|\n(?:Thought|Action|Observation):)",
            content,
            re.S,
        )
        if not match:
            return None
        try:
            params = json.loads(match.group(2))
        except json.JSONDecodeError:
            return None
        return (match.group(1), params) if isinstance(params, dict) else None

    def should_stop(self, response_message: dict, step: int, max_steps: int) -> bool:
        # 注意：超步判断由 AgentLoop 负责（步数耗尽 = 失败），此处不返回 True 兜底
        content = response_message.get("content", "") or ""
        return "Final Answer:" in content
