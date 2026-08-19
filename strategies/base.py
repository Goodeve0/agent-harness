"""
Strategy 基类 + 注册表

Strategy 定义了 Agent 如何规划和执行。
AgentLoop 调用 build_messages() 构建每轮输入，strategy 不感知具体 LLM 调用。
支持 function_calling / react / swe_bench 等策略热插拔，AgentLoop 主体不变。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from harness.sandbox import MockSandbox

_STRATEGY_REGISTRY: dict[str, type["BaseStrategy"]] = {}


def register_strategy(name: str):
    """装饰器：将 Strategy 类注册到全局注册表"""
    def decorator(cls: type):
        _STRATEGY_REGISTRY[name] = cls
        return cls
    return decorator


def get_strategy(name: str) -> type["BaseStrategy"]:
    if name not in _STRATEGY_REGISTRY:
        raise ValueError(f"Strategy '{name}' not found. Available: {list(_STRATEGY_REGISTRY)}")
    return _STRATEGY_REGISTRY[name]


class BaseStrategy(ABC):
    """
    Strategy 定义了 Agent 如何规划和执行。
    AgentLoop 调用 build_messages() 构建每轮输入，strategy 不感知具体 LLM 调用。
    """

    @abstractmethod
    def system_prompt(self) -> str:
        """返回该策略的系统提示词"""

    @abstractmethod
    def build_messages(self, task_prompt: str, history: list[dict]) -> list[dict]:
        """基于 task_prompt 和历史消息，构建本轮 LLM 输入 messages"""

    @abstractmethod
    def should_stop(self, response_message: dict, step: int, max_steps: int) -> bool:
        """判断是否结束 AgentLoop"""

    # 以下三个钩子提供默认实现，新增策略只在需要改变执行协议时覆写它们。
    # Function Calling 使用 OpenAI 原生 tool_calls；ReAct 使用文本 Action/Observation。
    def prepare_prompt(self, task_prompt: str, tools: list[dict]) -> str:
        return task_prompt

    def uses_native_tools(self) -> bool:
        return True

    def extract_action(self, response_message: dict) -> tuple[str, dict[str, Any]] | None:
        return None
