"""
聚合器：跨样本聚合的 Pass@k / Vote@k / Pass^k + Wilson 置信区间 + 难度分层

核心设计：
  Pass@k / Pass^k 的统计意义建立在「跨样本」聚合上，而非单样本内。
  例：50 个样本各跑 k=3 次 → Pass@3 = 至少 1 次通过的样本数 / 50
                   Pass^3 = 3 次全通过的样本数 / 50（鲁棒性）

  按 difficulty（easy / medium / hard）分层统计，暴露能力短板。
  Wilson score 95% 置信区间量化小样本下的不确定性。
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────────────
#  单样本聚合：每个样本的多次 trial 聚合成一个 pass/consistency 结论
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SampleAgg:
    """单个样本在 k 次采样下的聚合结果"""
    sample_id: str
    difficulty: str = "medium"
    k: int = 0
    n_pass: int = 0                 # k 次中通过几次
    pass_at_k: float = 0.0          # 至少 1 次通过 → 1.0
    pass_hat_k: float = 0.0         # k 次全通过 → 1.0
    vote_at_k: float = 0.0          # 多数票通过 → 1.0
    avg_score: float = 0.0
    pass_rate: float = 0.0          # n_pass / k
    trial_passed: list[bool] = field(default_factory=list)   # 每次 trial 是否通过（供报告展示）

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "difficulty": self.difficulty,
            "k": self.k,
            "n_pass": self.n_pass,
            "pass_rate": round(self.pass_rate, 4),
            "pass@k": int(self.pass_at_k),
            "vote@k": int(self.vote_at_k),
            "pass^k": int(self.pass_hat_k),
            "avg_score": round(self.avg_score, 4),
            "trial_passed": self.trial_passed,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  聚合器
# ─────────────────────────────────────────────────────────────────────────────

class MetricAggregator:
    """
    按 (sample_id) 聚合多次 trial，再跨样本统计得到整体指标。

    用法：
        agg = MetricAggregator(pass_threshold=0.6)
        for sample_id, trial_id, score, difficulty in trials:
            agg.add(sample_id, trial_id, score, difficulty)
        report = agg.aggregate()
    """

    def __init__(self, pass_threshold: float = 0.6):
        self.pass_threshold = pass_threshold
        # key = sample_id, value = list of (trial_id, score)
        self._trials: dict[str, list[tuple[int, float]]] = defaultdict(list)
        self._difficulty: dict[str, str] = {}

    def add(self, sample_id: str, trial_id: int,
            overall_score: float, difficulty: str = "medium"):
        self._trials[sample_id].append((trial_id, overall_score))
        self._difficulty[sample_id] = difficulty

    # ── 单样本聚合 ────────────────────────────────────────────────────────────
    def _aggregate_sample(self, sample_id: str) -> SampleAgg:
        trials = self._trials[sample_id]
        k = len(trials)
        passed = [s >= self.pass_threshold for _, s in trials]
        n_pass = sum(passed)
        scores = [s for _, s in trials]

        return SampleAgg(
            sample_id=sample_id,
            difficulty=self._difficulty.get(sample_id, "medium"),
            k=k,
            n_pass=n_pass,
            pass_at_k=1.0 if n_pass >= 1 else 0.0,
            pass_hat_k=1.0 if n_pass == k else 0.0,
            vote_at_k=1.0 if n_pass > k / 2 else 0.0,
            avg_score=sum(scores) / k if k else 0.0,
            pass_rate=n_pass / k if k else 0.0,
            trial_passed=passed,
        )

    # ── 单样本聚合结果列表 ──────────────────────────────────────────────────
    def aggregate(self) -> list[SampleAgg]:
        """返回每个样本的聚合结果（跨样本统计请用 summary()）。"""
        return [self._aggregate_sample(sid) for sid in self._trials]

    # ── 跨样本整体聚合（扁平 dict，供报告 / CI 消费）─────────────────────────
    def summary(self) -> dict:
        """
        跨样本聚合指标，返回扁平 dict。

        键名对齐 Reporter.print_summary 期望的 mean_pass@k / mean_vote@k / mean_pass^k。
        """
        samples = self.aggregate()
        if not samples:
            return {
                "total_samples": 0, "k": 0,
                "mean_pass@k": 0.0, "mean_vote@k": 0.0, "mean_pass^k": 0.0,
                "mean_pass_rate": 0.0, "mean_avg_score": 0.0,
                "ci_95": [0.0, 0.0], "by_difficulty": {},
            }
        n = len(samples)
        pass_at_k = sum(s.pass_at_k for s in samples) / n
        pass_hat_k = sum(s.pass_hat_k for s in samples) / n
        vote_at_k = sum(s.vote_at_k for s in samples) / n
        mean_pass_rate = sum(s.pass_rate for s in samples) / n
        mean_avg_score = sum(s.avg_score for s in samples) / n
        ci_low, ci_high = wilson_interval(pass_at_k, n)

        by_diff: dict[str, list[SampleAgg]] = defaultdict(list)
        for s in samples:
            by_diff[s.difficulty].append(s)
        by_difficulty = {}
        for diff, ds in by_diff.items():
            dn = len(ds)
            by_difficulty[diff] = {
                "total": dn,
                "mean_pass@k": round(sum(s.pass_at_k for s in ds) / dn, 4),
                "mean_pass^k": round(sum(s.pass_hat_k for s in ds) / dn, 4),
                "mean_vote@k": round(sum(s.vote_at_k for s in ds) / dn, 4),
                "mean_pass_rate": round(sum(s.pass_rate for s in ds) / dn, 4),
            }

        return {
            "total_samples": n,
            "k": samples[0].k,
            "mean_pass@k": round(pass_at_k, 4),
            "mean_vote@k": round(vote_at_k, 4),
            "mean_pass^k": round(pass_hat_k, 4),
            "mean_pass_rate": round(mean_pass_rate, 4),
            "mean_avg_score": round(mean_avg_score, 4),
            "ci_95": [round(ci_low, 4), round(ci_high, 4)],
            "by_difficulty": by_difficulty,
        }

    # ── Pass@k 曲线：不同采样预算下的能力上界 ────────────────────────────────
    def pass_at_k_curve(self, max_k: int = 5) -> list[dict]:
        """
        对每个 k ∈ [1, max_k]，计算"在该样本上 k 次 trial 中至少 1 次通过"的样本占比。
        用于展示采样预算 vs 能力上界的关系。
        """
        if not self._trials:
            return []

        result = []
        for k in range(1, max_k + 1):
            n_samples = 0
            n_have_pass = 0
            for sid, trials in self._trials.items():
                if len(trials) < k:
                    continue
                # 取前 k 次 trial
                first_k = trials[:k]
                if any(s >= self.pass_threshold for _, s in first_k):
                    n_have_pass += 1
                n_samples += 1
            if n_samples:
                rate = n_have_pass / n_samples
                ci_low, ci_high = wilson_interval(rate, n_samples)
                result.append({
                    "k": k,
                    "pass_at_k": round(rate, 4),
                    "ci_95": [round(ci_low, 4), round(ci_high, 4)],
                    "n_samples": n_samples,
                })
        return result


# ─────────────────────────────────────────────────────────────────────────────
#  Wilson score 置信区间
# ─────────────────────────────────────────────────────────────────────────────

def wilson_interval(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson score 95% 置信区间。
    比 normal approximation 更适合小样本与极端概率（接近 0 或 1）的场景。

    Args:
        p: 样本成功率
        n: 样本量
        z: 1.96 (95%) / 2.576 (99%)
    """
    if n == 0:
        return 0.0, 0.0
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, center - spread), min(1.0, center + spread)


# ─────────────────────────────────────────────────────────────────────────────
#  兼容旧接口
# ─────────────────────────────────────────────────────────────────────────────

def _mean_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Pass@k 无偏估计（HumanEval 论文公式）。
    当 n > k 时才有效；当 n == k 时退化为 0/1，应使用跨样本聚合的 pass_at_k_rate。
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def _mean_pass_hat_k(flags: list[bool]) -> float:
    """Pass^k 单样本鲁棒性：k 次全通过 → 1.0"""
    return 1.0 if all(flags) else 0.0


def _mean_vote_at_k(flags: list[bool]) -> float:
    """Vote@k 单样本多数票：> k/2 通过 → 1.0"""
    return 1.0 if sum(flags) > len(flags) / 2 else 0.0
