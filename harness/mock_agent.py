"""
Mock Agent：无 API Key 时用于验证评测链路本身

模拟被测 Agent 行为，按任务 rubric 声明式注入缺陷，验证规则层能正确捕获并归因：
注入什么缺陷 → 就该归因出什么标签，mock 模式的失败归因因此有验证价值。

注入点自动从 rubric 推断（也可用 YAML 的 mock_injection.<difficulty> 显式覆盖）：
  l3 field_non_empty      → field_empty             （→ MISSING_FIELD 类归因）
  l2 critical field_equals → field_wrong            （→ WRONG_CHOICE / CRITICAL）
  l2 critical tool_param_equals → tool_param_wrong  （→ TOOL_PARAM_ERROR）
  l3 critical regex_absent → reason_contains_violation（→ SAFETY_VIOLATION）
  l2 tool_sequence        → skip_last_tool          （→ TRAJECTORY_DEVIATION）
"""
from __future__ import annotations

import json
import re
from typing import Any

from harness.sandbox import MockSandbox


def mock_agent_run(spec: dict, sample: dict, sandbox: MockSandbox, trial_id: int):
    """
    模拟被测 Agent 行为，用于本地跑通评测链路（无需 API Key）。
    按任务 rubric 声明式注入缺陷，验证规则层能正确捕获并归因：
    注入什么缺陷 → 就该归因出什么标签，mock 模式的失败归因因此有验证价值。

    注入点自动从 rubric 推断（也可用 YAML 的 mock_injection.<difficulty> 显式覆盖）：
      l3 field_non_empty   → field_empty             （→ MISSING_FIELD 类归因）
      l2 critical field_equals → field_wrong          （→ WRONG_CHOICE / CRITICAL）
      l2 critical tool_param_equals → tool_param_wrong（→ TOOL_PARAM_ERROR）
      l3 critical regex_absent → reason_contains_violation（→ SAFETY_VIOLATION）
      l2 tool_sequence     → skip_last_tool           （→ TRAJECTORY_DEVIATION）
    """
    trace = sandbox.get_trace()
    tools = [t["function"]["name"] for t in spec.get("tools", [])]
    gt = dict(sample.get("ground_truth", {}))
    difficulty = sample.get("difficulty", "medium")
    rubric = spec.get("rubric", {})

    plan = injection_plan(rubric)
    yaml_plan = (spec.get("mock_injection") or {}).get(difficulty)
    if yaml_plan:
        plan = yaml_plan                    # YAML 显式声明优先于自动推断

    # 注入调度：每轮周期 = 1 次正常路径 + 全部缺陷。
    # 例：k=3 → [正常, 缺陷A, 缺陷B]；k=4 → [正常, 缺陷A, 缺陷B, 缺陷C]
    slots: list[dict | None] = [None] + (plan or [])
    inject = (trial_id - 1) % len(slots)
    defect = slots[inject]

    # 工具调用：skip_last_tool 缺陷只走前置链路（缺最后一环），其余按序全调用
    call_tools = list(tools)
    if defect and defect.get("defect") == "skip_last_tool":
        seq = defect.get("tools") or []
        if seq:
            call_tools = seq[:-1]
    for name in call_tools:
        sandbox.call_tool(name, mock_params(name, sample, gt))

    # 输出尽量贴合 ground_truth（剔除 expected_range 等评测元字段）
    output = {k: v for k, v in gt.items() if k != "expected_range"}
    output.setdefault("reason", "工具返回综合评分符合阈值，各维度均无违规风险，故作出该判定。")

    if defect:
        apply_defect(defect, output, gt, sandbox)

    trace.final_output = json.dumps(output, ensure_ascii=False)
    trace.success = True
    trace.total_tokens = 350
    return trace


def injection_plan(rubric: dict) -> list[dict]:
    """从 rubric 自动推断 mock 缺陷注入点，保证"注入什么缺陷 → 归因什么标签"一一对应"""
    plan: list[dict] = []
    for rule in rubric.get("l3", []) or []:
        if isinstance(rule, dict) and rule.get("check") == "field_non_empty":
            plan.append({"defect": "field_empty", "field": rule.get("field")})
    for rule in rubric.get("l2", []) or []:
        if (isinstance(rule, dict) and rule.get("check") == "field_equals"
                and as_bool(rule.get("critical", False))):
            plan.append({"defect": "field_wrong", "field": rule.get("field")})
    for rule in rubric.get("l2", []) or []:
        if (isinstance(rule, dict) and rule.get("check") == "tool_param_equals"
                and as_bool(rule.get("critical", False))):
            plan.append({"defect": "tool_param_wrong",
                         "tool": rule.get("tool"), "param": rule.get("param")})
    for rule in rubric.get("l3", []) or []:
        if (isinstance(rule, dict) and rule.get("check") == "regex_absent"
                and as_bool(rule.get("critical", False))):
            plan.append({"defect": "reason_contains_violation",
                         "pattern": rule.get("pattern", "")})
    for rule in rubric.get("l2", []) or []:
        if isinstance(rule, dict) and rule.get("check") == "tool_sequence" and rule.get("sequence"):
            plan.append({"defect": "skip_last_tool", "tools": rule["sequence"]})
            break
    return plan


def apply_defect(defect: dict, output: dict, gt: dict, sandbox: MockSandbox):
    """按声明式缺陷配置篡改输出 / 追加错误工具调用，模拟真实 Agent 典型失败"""
    kind = defect.get("defect")

    if kind == "field_empty":
        field = defect.get("field")
        if field:
            output[field] = ""

    elif kind == "field_wrong":
        field = defect.get("field")
        if field and field in output:
            output[field] = flip_value(output[field])

    elif kind == "tool_param_wrong":
        tool, param = defect.get("tool"), defect.get("param")
        if tool and param is not None and param in gt:
            wrong_val = flip_value(gt[param])
            # 追加一次参数错误的调用：副作用工具多调一次错参数即资损，规则层必须抓到
            sandbox.call_tool(tool, {**gt, param: wrong_val})
            if param in output:
                output[param] = wrong_val

    elif kind == "reason_contains_violation":
        word = extract_first_word(defect.get("pattern") or "")
        output["reason"] = f"该商品包含违禁词「{word}」，应予以过滤。"


def flip_value(v: Any) -> Any:
    """给字段造一个明显错误的值：bool 取反 / 数值 +1 / 字符串加后缀"""
    if isinstance(v, bool):
        return not v
    if isinstance(v, float):
        return v + 1.0
    if isinstance(v, int):
        return v + 1
    if isinstance(v, str):
        return v + "_x"
    return v


def extract_first_word(pattern: str) -> str:
    """从正则 pattern 里提取第一个候选词用于注入（如 "(高仿|假冒)" → "高仿"）"""
    cleaned = pattern.replace("(", "").replace(")", "").replace("|", " ")
    m = re.search(r"[\u4e00-\u9fffA-Za-z0-9]+", cleaned)
    return m.group() if m else "违禁词"


def as_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on", "是")
    if v is None:
        return default
    return bool(v)


def mock_params(tool_name: str, sample: dict, gt: dict) -> dict:
    """构造 mock 工具调用参数，尽量贴合 ground_truth 以通过参数校验"""
    inp = sample.get("input", {})
    if tool_name == "review_content":
        return {"content": inp.get("content", "")}
    if tool_name == "query_order":
        return {"order_id": gt.get("order_id", "")}
    if tool_name == "submit_refund":
        return {"order_id": gt.get("order_id", ""), "amount": gt.get("amount", 0)}
    if tool_name == "send_notification":
        return {"user_id": "user_123", "message": "您的退款已提交"}
    return {}
