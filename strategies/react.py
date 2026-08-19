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


def _extract_json_object(text: str) -> dict | None:
    """从任意文本中提取最外层 JSON 对象。

    用花括号栈（字符串感知 + 转义感知）定位最外层边界，再交给 json.loads 校验，
    支持嵌套对象 {"a": {"b": 1}} / 数组 / 字符串内花括号，避免非贪婪正则
    在嵌套 JSON 处提前截断导致解析失败。
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        return obj if isinstance(obj, dict) else None
                    except json.JSONDecodeError:
                        return None
    return None


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
        """解析 ReAct 文本 Action，供 AgentLoop 执行并回填 Observation。

        返回 None 仅表示"本轮没有可执行的 Action"（Final Answer / 纯文本回复）。
        格式错误（有 Action 但 Action Input 缺失/非法）由 format_error_feedback
        返回可回灌的错误信息，交给 AgentLoop 让模型自我修正，而非静默丢步。
        """
        content = response_message.get("content", "") or ""
        if "Final Answer:" in content:
            return None
        match = re.search(r"(?mi)^\s*Action\s*:\s*([A-Za-z_][A-Za-z0-9_]*)", content)
        if not match:
            return None
        tool_name = match.group(1)

        # Action Input 允许跨行，但必须紧跟 Action（在下一个 ReAct 关键字前截断，
        # 或仅当 JSON 之后再无内容时以 \s*\Z 收尾），避免与后续轮次的 Action Input
        # 错误配对。注意收尾必须用 \Z（绝对末尾）而非 $——MULTILINE 模式下 $ 会
        # 零宽匹配每个换行前的位置，导致多行 JSON 在首个换行处被提前截断。
        rest = content[match.end():]
        input_match = re.search(
            r"(?mi)^\s*Action\s*Input\s*:\s*(.*?)"
            r"(?=\n\s*(?:Thought|Action|Observation|Final Answer)\s*:|\s*\Z)",
            rest,
            re.S,
        )
        if not input_match:
            return None
        params = _extract_json_object(input_match.group(1))
        return (tool_name, params) if params is not None else None

    def format_error_feedback(self, response_message: dict) -> str | None:
        """Action 格式漂移时返回可回灌的错误提示。

        参考 LangChain ReAct 惯例：解析失败（OutputParserException）不应静默吞掉
        该步，而是把错误以 Observation 形式反馈给模型，让它修正格式后重试。
        返回 None = 无需纠错（Final Answer / 纯文本中间回复）。
        """
        content = response_message.get("content", "") or ""
        if "Final Answer:" in content:
            return None
        if not re.search(r"(?mi)^\s*Action\s*:", content):
            return None
        if self.extract_action(response_message) is not None:
            return None  # 本轮已正常解析，无需纠错

        if not re.search(r"(?mi)^\s*Action\s*Input\s*:", content):
            return ("Observation: 输出格式错误 —— 缺少 Action Input。请严格按格式重新输出：\n"
                    "Action: <工具名>\n"
                    "Action Input: <JSON 对象>")
        return ("Observation: 输出格式错误 —— Action Input 不是合法 JSON 对象。"
                "请输出单个 JSON 对象（支持嵌套），不要附加多余文本，然后重新输出。")

    def should_stop(self, response_message: dict, step: int, max_steps: int) -> bool:
        # 注意：超步判断由 AgentLoop 负责（步数耗尽 = 失败），此处不返回 True 兜底
        content = response_message.get("content", "") or ""
        return "Final Answer:" in content
