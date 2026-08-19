"""
HybridScorer —— 双层评分架构（规则层 + Judge 层加权合并）

核心设计动机：
  纯 LLM Judge 方案有三个致命问题：
    1. 不确定性 —— 同一条 trace 两次评分不一致，回归对比失去意义
    2. 成本高   —— 每条样本一次 LLM 调用，50 task × 3 trial = 150 次
    3. 不可解释 —— 只给分数，无法定位"哪一步错了"

  双层方案：
    规则层（确定性）：格式合规 / 工具调用链路 / 精确匹配 → 可复现、零成本、可定位
    Judge 层（语义） ：仅对规则层无法判定的语义质量评分 → pattern / numeric 双模式
    加权合并        ：final = w_rule × rule_score + w_judge × judge_score

  收益：
    - Judge 层调用量下降（规则能判的不走 LLM）
    - 评分方差下降（确定性部分占比越高越稳）
    - 失败可归因（规则层直接给出 failed check 名称）

短路策略（fail-fast）：
  规则层 L1 格式检查未通过 → 直接判 0 分，不再浪费 Judge 调用。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from harness.sandbox import AgentTrace
from metrics.rule_checker import RuleChecker, RuleScore


# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HybridResult:
    """双层评分最终结果"""
    rule_score: float
    judge_score: float
    final_score: float
    rule_detail: dict = field(default_factory=dict)
    judge_detail: dict = field(default_factory=dict)
    failure_reason: str | None = None
    suggestion: str = ""
    judge_skipped: bool = False        # 是否因短路跳过了 Judge 调用

    def to_dict(self) -> dict:
        return {
            "rule_score": round(self.rule_score, 4),
            "judge_score": round(self.judge_score, 4),
            "overall_score": round(self.final_score, 4),
            "rule_detail": self.rule_detail,
            "judge_detail": self.judge_detail,
            "failure_reason": self.failure_reason,
            "suggestion": self.suggestion,
            "judge_skipped": self.judge_skipped,
            # 兼容旧版报告字段
            "l1": self.rule_detail.get("l1", {}),
            "l2": self.rule_detail.get("l2", {}),
            "l3": self.rule_detail.get("l3", {}),
        }


# ─── 失败归因标签 ─────────────────────────────────────────────────────────────

class FailureReason:
    FORMAT_ERROR = "FORMAT_ERROR"                  # 输出格式不合规
    MISSING_FIELD = "MISSING_FIELD"                # 必要字段缺失
    TRAJECTORY_DEVIATION = "TRAJECTORY_DEVIATION"  # 工具调用链路偏离
    TOOL_PARAM_ERROR = "TOOL_PARAM_ERROR"          # 工具参数错误
    WRONG_CHOICE = "WRONG_CHOICE"                  # 结果决策错误
    SAFETY_VIOLATION = "SAFETY_VIOLATION"          # 安全规则违反
    LLM_JUDGE_FAIL = "LLM_JUDGE_FAIL"              # 语义质量不达标
    MAX_STEPS_EXCEEDED = "MAX_STEPS_EXCEEDED"      # 超出步数上限
    EXECUTION_ERROR = "EXECUTION_ERROR"            # 执行异常
    CRITICAL_FAILURE = "CRITICAL_FAILURE"          # 关键项失败（资损级，一票否决）


# check 名称 → 归因标签的映射规则（按优先级从高到低匹配）
_REASON_RULES = [
    (("json_format",), FailureReason.FORMAT_ERROR),
    (("required_fields",), FailureReason.MISSING_FIELD),
    (("tool_param",), FailureReason.TOOL_PARAM_ERROR),
    (("tool_called", "tool_sequence", "max_steps"), FailureReason.TRAJECTORY_DEVIATION),
    (("no_violation", "safety", "regex_absent"), FailureReason.SAFETY_VIOLATION),
]


# ─────────────────────────────────────────────────────────────────────────────

class HybridScorer:
    """
    参数：
        rubric:       Task Spec 的 rubric 段
        judge:        LLMJudge 实例（可为 None，则退化为纯规则评测）
        weights:      {"rule": 0.6, "judge": 0.4}
        judge_rubric: 传给 Judge 层的语义评分标准（自然语言）
        pass_threshold: 判定通过的阈值
    """

    def __init__(
        self,
        rubric: dict,
        judge=None,
        weights: dict | None = None,
        judge_rubric: str = "",
        pass_threshold: float = 0.6,
        layer_weights: dict[str, float] | None = None,
    ):
        self.rule_checker = RuleChecker(rubric, layer_weights=layer_weights)
        self.judge = judge
        w = weights or {"rule": 0.6, "judge": 0.4}
        # 无 judge 时权重全给规则层
        self.w_rule = w.get("rule", 0.6) if judge else 1.0
        self.w_judge = w.get("judge", 0.4) if judge else 0.0
        self.judge_rubric = judge_rubric
        self.pass_threshold = pass_threshold

    def score(self, trace: AgentTrace, ground_truth: dict | None = None,
              expected_behavior: str = "") -> HybridResult:
        # ── 第一层：规则层（确定性，零 LLM 成本）──────────────────────────────
        rule_result: RuleScore = self.rule_checker.check(trace, ground_truth)
        rule_score = rule_result.score()
        rule_detail = rule_result.to_dict()

        # ── 短路：规则层决定性失败，不再花钱调 Judge ──────────────────────────
        # 两类短路共用一条返回路径：
        #   1. critical 一票否决（资损级）→ 规则分归零；
        #   2. L1 格式/必填字段失败 → 保留规则层得分但总分判 0。
        # 差异只在 rule_score 与 suggestion 上，归因/跳过标记一致。
        crit = rule_result.critical_failed()
        l1 = rule_detail.get("l1", {})
        l1_failed = l1.get("json_format") == 0 or l1.get("required_fields") == 0
        if crit or l1_failed:
            return HybridResult(
                rule_score=0.0 if crit else rule_score,
                judge_score=0.0,
                final_score=0.0,
                rule_detail=rule_detail,
                failure_reason=self._infer_reason(rule_result, 0.0),
                suggestion=("关键项失败（一票否决）：" + "；".join(
                    f"{c.name} - {c.detail}" for c in crit))
                if crit else self._build_suggestion(rule_result, 0.0),
                judge_skipped=True,
            )

        # ── 第二层：Judge 层（仅评语义质量）──────────────────────────────────
        judge_score, judge_detail, skipped = 0.0, {}, False
        if self.judge and self.w_judge > 0:
            try:
                judge_score = self.judge.judge(
                    text=trace.final_output,
                    rubric=self.judge_rubric or expected_behavior,
                    reference=str(ground_truth or ""),
                )
                if judge_score < 0:          # -1 表示 Judge 无法判定
                    judge_score, skipped = 0.0, True
                judge_detail = {"mode": self.judge.mode, "score": judge_score}
            except Exception as e:
                judge_score, skipped = 0.0, True
                judge_detail = {"error": str(e)}
        else:
            skipped = True

        # ── 加权合并 ──────────────────────────────────────────────────────────
        if skipped:
            # Judge 不可用时权重回落到规则层，避免无辜扣分
            final = rule_score
        else:
            final = self.w_rule * rule_score + self.w_judge * judge_score

        return HybridResult(
            rule_score=rule_score,
            judge_score=judge_score,
            final_score=round(final, 4),
            rule_detail=rule_detail,
            judge_detail=judge_detail,
            failure_reason=self._infer_reason(rule_result, final),
            suggestion=self._build_suggestion(rule_result, judge_score),
            judge_skipped=skipped,
        )

    # ── 失败归因：不只说"分低"，而是定位到具体 check ────────────────────────
    def _infer_reason(self, rule_result: RuleScore, final: float) -> str | None:
        crit = rule_result.critical_failed()
        pool = crit or rule_result.failed_checks()
        for keys, reason in _REASON_RULES:
            for c in pool:
                if any(k in c.name for k in keys):
                    return reason
        if crit:
            return FailureReason.CRITICAL_FAILURE
        if pool:
            return FailureReason.WRONG_CHOICE
        if final < self.pass_threshold:
            return FailureReason.LLM_JUDGE_FAIL
        return None

    # ── 可操作建议：给出改 Prompt 的具体方向 ────────────────────────────────
    def _build_suggestion(self, rule_result: RuleScore, judge_score: float) -> str:
        failed = rule_result.failed_checks()
        if not failed:
            if judge_score < self.pass_threshold:
                return "规则层全通过但语义质量不足，建议在 Prompt 中补充输出内容的质量要求与示例。"
            return ""
        tips = []
        for c in failed[:3]:
            if "json_format" in c.name:
                tips.append("Prompt 中明确要求「只输出 JSON，不要任何解释文字」并给出示例")
            elif "required_fields" in c.name:
                tips.append(f"Prompt 中列出必填字段清单（{c.detail}）")
            elif "tool_called" in c.name or "tool_sequence" in c.name:
                tips.append(f"Prompt 中强调工具调用顺序约束（{c.detail}）")
            elif "tool_param" in c.name:
                tips.append(f"Prompt 中说明参数取值依据（{c.detail}）")
            else:
                tips.append(f"{c.name} 未通过：{c.detail}")
        return "；".join(tips)
