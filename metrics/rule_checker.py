"""
规则层评分器（确定性 / Deterministic Layer）

规则层覆盖：
  1. 格式合规      —— JSON 可解析、必要字段齐全（L1）
  2. 工具调用链路  —— 是否调用、调用顺序、参数正确性（L2 轨迹评测）
  3. 结果精确匹配  —— 字段值 / 数值区间 / 正则（L2 结果评测）

所有 check 以声明式 YAML 配置，无需改代码即可新增规则。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from harness.sandbox import AgentTrace


# ─────────────────────────────────────────────────────────────────────────────
#  Check 结果
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    """单条规则检查结果。score 使用三值：1 通过 / 0 不通过 / -1 无法判定"""
    name: str
    layer: str            # "l1" | "l2" | "l3"
    score: int            # 1 | 0 | -1
    detail: str = ""
    weight: float = 1.0   # 权重：重要 check 可放大影响
    critical: bool = False  # 一票否决：失败则整条样本直接判 0

    @property
    def passed(self) -> bool:
        return self.score == 1


@dataclass
class RuleScore:
    """规则层整体评分结果"""
    checks: list[CheckResult] = field(default_factory=list)
    layer_weights: dict[str, float] | None = None   # {"l1": 0.2, "l2": 0.6, "l3": 0.2}

    def add(self, name: str, layer: str, score: int, detail: str = "",
            weight: float = 1.0, critical: bool = False):
        self.checks.append(CheckResult(name, layer, score, detail, weight, critical))

    def by_layer(self, layer: str) -> dict[str, int]:
        return {c.name: c.score for c in self.checks if c.layer == layer}

    def critical_failed(self) -> list[CheckResult]:
        """失败的关键 check —— 存在即触发一票否决"""
        return [c for c in self.checks if c.critical and c.score == 0]

    def score(self) -> float:
        """
        规则层总分：
        - 存在失败的 critical check 时直接返回 0（一票否决），
          避免资损级错误被其他通过项均值稀释。
        - 配置 layer_weights 时按层加权（层内再按 check 权重加权），
          默认（None）为全 check 加权平均，保持向后兼容。
        """
        if self.critical_failed():
            return 0.0
        valid = [c for c in self.checks if c.score >= 0]
        if not valid:
            return 1.0        # 无规则可判 → 不扣分，完全交给 Judge 层
        if self.layer_weights:
            layers = sorted({c.layer for c in valid})
            lw = {l: self.layer_weights.get(l, 0.0) for l in layers}
            total_lw = sum(lw.values())
            if total_lw <= 0:                # 配置缺失 → 层间等权
                lw = {l: 1.0 for l in layers}
                total_lw = len(layers)
            layer_scores: dict[str, float] = {}
            for l in layers:
                cs = [c for c in valid if c.layer == l]
                tw = sum(c.weight for c in cs)
                layer_scores[l] = sum(c.score * c.weight for c in cs) / tw if tw else 1.0
            return round(sum(lw[l] * layer_scores[l] for l in layers) / total_lw, 4)
        total_w = sum(c.weight for c in valid)
        return sum(c.score * c.weight for c in valid) / total_w if total_w else 1.0

    def failed_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if c.score == 0]

    def to_dict(self) -> dict:
        return {
            "l1": self.by_layer("l1"),
            "l2": self.by_layer("l2"),
            "l3": self.by_layer("l3"),
            "rule_score": round(self.score(), 4),
            "critical_failed": [c.name for c in self.critical_failed()],
            "failed": [{"name": c.name, "detail": c.detail, "critical": c.critical}
                       for c in self.failed_checks()],
        }


# ─────────────────────────────────────────────────────────────────────────────
#  RuleChecker
# ─────────────────────────────────────────────────────────────────────────────

class RuleChecker:
    """
    声明式规则检查器。规则全部来自 Task Spec YAML 的 rubric 段，
    新增一条规则 = YAML 里加 4 行，无需改 Python 代码。

    支持的 check type：
      required_fields   必要字段齐全（L1）
      json_format       输出可 JSON 解析（L1）
      tool_called       某工具被调用过（L2 轨迹）
      tool_sequence     工具调用顺序符合预期（L2 轨迹）
      tool_param_equals 某次工具调用的参数值正确（L2 轨迹）
      field_equals      输出字段值精确匹配 ground_truth（L2 结果）
      field_in_range    输出数值字段落在区间内（L2 结果）
      regex_match       输出匹配正则（L3）
      regex_absent      输出不含违禁模式（L3 安全）
      max_steps         步数不超上限（L2 效率）
    """

    def __init__(self, rubric: dict, layer_weights: dict[str, float] | None = None):
        self.rubric = rubric or {}
        self.layer_weights = layer_weights

    # ── 主入口 ────────────────────────────────────────────────────────────────
    def check(self, trace: AgentTrace, ground_truth: dict | None = None) -> RuleScore:
        gt = ground_truth or {}
        rs = RuleScore(layer_weights=self.layer_weights)

        # L1：格式层（内置，所有 Agent 必查，天然 critical）
        l1_cfg = self.rubric.get("l1", {}) or {}
        parsed = self._parse_output(trace.final_output, strict=bool(l1_cfg.get("strict_json", False)))
        if l1_cfg.get("format") == "json":
            rs.add("json_format", "l1", 1 if parsed is not None else 0,
                   "" if parsed is not None else "final_output 不是合法 JSON",
                   critical=True)
        required = l1_cfg.get("required_fields") or []
        if required:
            if parsed is None:
                rs.add("required_fields", "l1", 0, "输出无法解析，字段检查跳过", critical=True)
            else:
                missing = [f for f in required if f not in parsed]
                rs.add("required_fields", "l1", 0 if missing else 1,
                       f"缺失字段: {missing}" if missing else "", critical=True)

        if l1_cfg.get("tool_calls_valid"):
            invalid = [s for s in trace.steps if s.error]
            rs.add("tool_calls_valid", "l1", 0 if invalid else 1,
                   "; ".join(f"{s.tool_name}: {s.error}" for s in invalid),
                   critical=True)

        # L2 / L3：声明式规则
        for layer in ("l2", "l3"):
            for rule in self.rubric.get(layer, []) or []:
                if not isinstance(rule, dict):
                    continue
                self._apply_rule(rs, rule, layer, trace, parsed, gt)

        return rs

    # ── 单条规则分发 ──────────────────────────────────────────────────────────
    def _apply_rule(self, rs: RuleScore, rule: dict, layer: str,
                    trace: AgentTrace, parsed: dict | None, gt: dict):
        name = rule.get("name", "unnamed")
        rtype = rule.get("check") or rule.get("type")
        tools_called = [s.tool_name for s in trace.steps]
        # YAML 声明式配置权重与一票否决（critical 可能是字符串 "false"，需安全解析）
        kw = {"weight": float(rule.get("weight", 1.0)),
              "critical": _as_bool(rule.get("critical", False))}

        # ── 轨迹类规则（评的是"怎么做"，不是"做出了什么"）───────────────────
        if rtype == "tool_called":
            target = rule["tool"]
            ok = target in tools_called
            rs.add(name, layer, 1 if ok else 0,
                   "" if ok else f"未调用 {target}，实际调用: {tools_called}", **kw)

        elif rtype == "tool_sequence":
            expected = rule["sequence"]
            ok = self._is_subsequence(expected, tools_called)
            rs.add(name, layer, 1 if ok else 0,
                   "" if ok else f"期望顺序 {expected}，实际 {tools_called}", **kw)

        elif rtype == "tool_param_equals":
            target, key = rule["tool"], rule["param"]
            expect = self._resolve(rule["expect"], gt)
            hits = [s for s in trace.steps if s.tool_name == target]
            if not hits:
                rs.add(name, layer, 0, f"未调用 {target}，无法校验参数", **kw)
            else:
                # 副作用工具（退款/下单）重复调用即风险：所有调用都必须正确，任一参数错即失败
                bad = [s for s in hits if not self._loose_eq(s.params.get(key), expect)]
                ok = not bad
                rs.add(name, layer, 1 if ok else 0,
                       "" if ok else
                       f"{target}.{key} 期望 {expect}，{len(bad)}/{len(hits)} 次调用参数错误", **kw)

        elif rtype == "max_steps":
            limit = rule["limit"]
            ok = len(trace.steps) <= limit
            rs.add(name, layer, 1 if ok else 0,
                   "" if ok else f"步数 {len(trace.steps)} 超出上限 {limit}", **kw)

        # ── 结果类规则 ────────────────────────────────────────────────────────
        elif rtype == "field_equals":
            key = rule["field"]
            expect = self._resolve(rule.get("expect", f"$gt.{key}"), gt)
            if parsed is None:
                rs.add(name, layer, 0, "输出无法解析", **kw)
            elif expect is None:
                rs.add(name, layer, -1, "ground_truth 缺少该字段，无法判定", **kw)
            else:
                ok = self._loose_eq(parsed.get(key), expect)
                rs.add(name, layer, 1 if ok else 0,
                       "" if ok else f"{key} 期望 {expect}，实际 {parsed.get(key)}", **kw)

        elif rtype == "field_in_range":
            key = rule["field"]
            rng = self._resolve(rule.get("range", "$gt.expected_range"), gt)
            val = (parsed or {}).get(key)
            if parsed is None or val is None:
                rs.add(name, layer, 0, f"输出缺少字段 {key}", **kw)
            elif not rng:
                rs.add(name, layer, -1, "未提供区间，无法判定", **kw)
            else:
                try:
                    ok = float(rng[0]) <= float(val) <= float(rng[1])
                    rs.add(name, layer, 1 if ok else 0,
                           "" if ok else f"{key}={val} 不在区间 {rng}", **kw)
                except (TypeError, ValueError):
                    rs.add(name, layer, 0, f"{key}={val} 非数值", **kw)

        # ── 安全 / 文本类规则 ─────────────────────────────────────────────────
        elif rtype == "regex_match":
            ok = bool(re.search(rule["pattern"], trace.final_output or ""))
            rs.add(name, layer, 1 if ok else 0,
                   "" if ok else f"输出未匹配 {rule['pattern']}", **kw)

        elif rtype == "regex_absent":
            hit = re.search(rule["pattern"], trace.final_output or "")
            rs.add(name, layer, 0 if hit else 1,
                   f"命中违禁模式: {hit.group()}" if hit else "", **kw)

        elif rtype == "field_non_empty":
            key = rule["field"]
            val = (parsed or {}).get(key)
            ok = bool(val) and str(val).strip() != ""
            rs.add(name, layer, 1 if ok else 0,
                   "" if ok else f"{key} 为空", **kw)

        # ── 交给 Judge 层的语义规则，规则层不判定 ─────────────────────────────
        elif rtype in ("llm_judge", "semantic"):
            rs.add(name, layer, -1, "语义指标，交由 Judge 层评分")

        else:
            rs.add(name, layer, -1, f"未知 check 类型: {rtype}")

    # ── 工具函数 ──────────────────────────────────────────────────────────────
    @staticmethod
    def _parse_output(output: str, strict: bool = False) -> dict | None:
        """strict 时仅接受完整 JSON 对象；宽松模式兼容 Markdown 包裹。"""
        if not output:
            return None
        text = output.strip()
        if strict:
            try:
                data = json.loads(text)
                return data if isinstance(data, dict) else None
            except (json.JSONDecodeError, TypeError):
                return None
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
        if fence:
            text = fence.group(1)
        else:
            brace = re.search(r"\{.*\}", text, re.S)
            if brace:
                text = brace.group()
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else None
        except (json.JSONDecodeError, TypeError):
            return None

    @staticmethod
    def _resolve(expr: Any, gt: dict) -> Any:
        """支持 "$gt.field" 语法从 ground_truth 取值，避免规则里硬编码期望值"""
        if isinstance(expr, str) and expr.startswith("$gt."):
            return gt.get(expr[4:])
        return expr

    @staticmethod
    def _loose_eq(a: Any, b: Any) -> bool:
        """宽松相等：299 == 299.0 == "299"，True == "true" """
        if a is None or b is None:
            return a is b
        if isinstance(a, bool) or isinstance(b, bool):
            return str(a).lower() == str(b).lower()
        try:
            return abs(float(a) - float(b)) < 1e-6
        except (TypeError, ValueError):
            return str(a).strip().lower() == str(b).strip().lower()

    @staticmethod
    def _is_subsequence(expected: list, actual: list) -> bool:
        """expected 是否按序出现在 actual 中（允许中间插入其他调用）"""
        it = iter(actual)
        return all(e in it for e in expected)


def _as_bool(v: Any, default: bool = False) -> bool:
    """
    YAML 安全布尔解析：critical: "false" 是字符串，bool("false") 会得到 True。
    兼容字符串 "false"/"0"/"no" 与真正的 bool 值。
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on", "是")
    if v is None:
        return default
    return bool(v)
