"""
双模型交叉校验（Cross-Model Validation）

单一 LLM Judge 存在系统性偏见：
  · 同源偏好：用同一厂商模型评测自家输出，倾向给高分
  · 风格偏好：偏爱冗长、结构化的回答，与业务真实质量未必正相关
  · 静默漂移：模型版本更新后评分标准悄悄变化，历史数据失去可比性

用两个异构模型独立对同一条 trace 评分，统计二者的一致率（Agreement Rate）：
  · 一致率高  → Judge 结论可信，采信均值
  · 一致率低  → 该样本标记为 DISPUTED，进入人工复核队列

一致率本身是"评测系统可信度"的量化指标，把评测器本身也纳入评测。

指标：
  agreement_rate     两模型判定（pass/fail）一致的样本占比
  mean_abs_diff      两模型分差绝对值均值（分歧强度）
  disputed_samples   分歧样本列表 → 直接产出人工标注队列
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean


# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CrossCheckRecord:
    """单条样本的双模型评分记录"""
    sample_id: str
    score_a: float
    score_b: float
    model_a: str
    model_b: str
    threshold: float = 0.6

    @property
    def valid(self) -> bool:
        """两侧都给出了有效分（>=0）；-1 表示该模型评分异常/无法判定"""
        return self.score_a >= 0 and self.score_b >= 0

    @property
    def verdict_a(self) -> bool:
        return self.score_a >= self.threshold

    @property
    def verdict_b(self) -> bool:
        return self.score_b >= self.threshold

    @property
    def agreed(self) -> bool:
        """判定一致 = 两模型对 pass/fail 的结论相同（任一侧无法判定不算一致）"""
        if not self.valid:
            return False
        return self.verdict_a == self.verdict_b

    @property
    def abs_diff(self) -> float:
        if not self.valid:
            return 0.0
        return abs(self.score_a - self.score_b)

    @property
    def merged_score(self) -> float:
        """一致时取均值；分歧时取保守值（较低分），避免高估 Agent 能力。
        单侧异常时只采信有效一侧；双侧均异常按 0 处理。"""
        vals = [s for s in (self.score_a, self.score_b) if s >= 0]
        if len(vals) == 2:
            return mean(vals) if self.agreed else min(vals)
        return vals[0] if vals else 0.0

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            self.model_a: round(self.score_a, 4),
            self.model_b: round(self.score_b, 4),
            "agreed": self.agreed,
            "abs_diff": round(self.abs_diff, 4),
            "merged_score": round(self.merged_score, 4),
        }


# ─────────────────────────────────────────────────────────────────────────────

class CrossValidator:
    """
    双模型交叉校验器。

    用法：
        cv = CrossValidator(judge_a=LLMJudge(judge_model="claude-code"),
                            judge_b=LLMJudge(judge_model="codex"))
        cv.validate(sample_id, text, rubric, reference)
        report = cv.report()      # → agreement_rate / disputed_samples
    """

    def __init__(self, judge_a, judge_b, threshold: float = 0.6,
                 dispute_delta: float = 0.3):
        """
        Args:
            judge_a / judge_b: 两个异构 LLMJudge 实例（不同厂商模型）
            threshold:      pass/fail 判定阈值
            dispute_delta:  分差超过此值即使结论一致也标记为争议
        """
        self.judge_a = judge_a
        self.judge_b = judge_b
        self.threshold = threshold
        self.dispute_delta = dispute_delta
        self.records: list[CrossCheckRecord] = []

    def validate(self, sample_id: str, text: str, rubric: str,
                 reference: str = "") -> CrossCheckRecord:
        """对同一条输出用两个模型独立评分"""
        score_a = self._safe_judge(self.judge_a, text, rubric, reference)
        score_b = self._safe_judge(self.judge_b, text, rubric, reference)

        record = CrossCheckRecord(
            sample_id=sample_id,
            score_a=score_a,
            score_b=score_b,
            model_a=getattr(self.judge_a, "judge_model", "model_a"),
            model_b=getattr(self.judge_b, "judge_model", "model_b"),
            threshold=self.threshold,
        )
        self.records.append(record)
        return record

    @staticmethod
    def _safe_judge(judge, text: str, rubric: str, reference: str) -> float:
        try:
            s = judge.judge(text=text, rubric=rubric, reference=reference)
            # 保留 -1 语义：judge 显式返回 -1（无法判定 / CLI 不可用 / 解析失败）
            # 与抛异常等价，都是"该侧无有效评分"，不能转成 0 分拉低仲裁分。
            # CrossCheckRecord.valid 以 score >= 0 判定有效性，-1 自然落到无效侧。
            return float(s)
        except Exception:
            # 评分异常不等于打 0 分：返回 -1 标记无效，避免拉低仲裁分
            return -1.0

    # ── 一致性报告 ────────────────────────────────────────────────────────────
    def report(self) -> dict:
        valid_recs = [r for r in self.records if r.valid]
        # fail-open：两侧 CLI 均不可用时（-1/-1），报告也必须返回结构完整的
        # dict，调用方（run_eval 等）按协议索引不崩。与 judge 侧 -1 语义一致：
        # 不可用 ≠ 0 分，指标全部归零但键齐全，verdict 走"一致率偏低"路径提示。
        empty = {
            "total": 0,
            "model_a": getattr(self.judge_a, "judge_model", "model_a"),
            "model_b": getattr(self.judge_b, "judge_model", "model_b"),
            "agreement_rate": 0.0,
            "mean_abs_diff": 0.0,
            "mean_score_a": 0.0,
            "mean_score_b": 0.0,
            "disputed_count": 0,
            "disputed_samples": [],
            "verdict": self._verdict(0.0),
        }
        if not valid_recs:
            return empty

        n = len(valid_recs)
        agreed = sum(1 for r in valid_recs if r.agreed)
        disputed = [
            r for r in valid_recs
            if (not r.agreed) or r.abs_diff > self.dispute_delta
        ]

        return {
            "total": n,
            "model_a": valid_recs[0].model_a,
            "model_b": valid_recs[0].model_b,
            "agreement_rate": round(agreed / n, 4),
            "mean_abs_diff": round(mean(r.abs_diff for r in valid_recs), 4),
            "mean_score_a": round(mean(r.score_a for r in valid_recs), 4),
            "mean_score_b": round(mean(r.score_b for r in valid_recs), 4),
            "disputed_count": len(disputed),
            "disputed_samples": [r.to_dict() for r in disputed],
            # 一致率低于 0.8 说明评测标准本身模糊，需要细化 Rubric
            "verdict": self._verdict(agreed / n),
        }

    @staticmethod
    def _verdict(rate: float) -> str:
        if rate >= 0.9:
            return "评测标准清晰，Judge 结论高度可信"
        if rate >= 0.8:
            return "评测标准基本可靠，建议复核争议样本"
        return "一致率偏低，Rubric 定义模糊，建议将模糊指标拆解为二元 check"

    def merged_scores(self) -> dict[str, float]:
        """返回 {sample_id: merged_score}，供主流程采信"""
        return {r.sample_id: r.merged_score for r in self.records}
