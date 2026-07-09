"""
Statistical comparison layer.

Two different questions need two different tests:

1. Shadow vs primary (no ground truth needed): do the two models AGREE on
   the same inputs? This is an agreement-rate question, not an accuracy
   question, because shadow mode is often run before ground truth exists.

2. Canary vs control (ground truth required, eventually): is the canary's
   ACCURACY actually different from control's, or is the observed gap
   noise? This needs a proper hypothesis test, not a raw percentage
   comparison, because raw percentages ignore sample size.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats


@dataclass
class AgreementReport:
    n: int
    agreement_rate: float
    primary_only_correct: Optional[int]  # requires ground truth
    shadow_only_correct: Optional[int]
    both_correct: Optional[int]
    both_wrong: Optional[int]
    mean_confidence_delta: float  # shadow_conf - primary_conf, signed mean
    mcnemar_stat: Optional[float]
    mcnemar_p_value: Optional[float]


def compare_shadow_to_primary(
    primary_preds: np.ndarray,
    shadow_preds: np.ndarray,
    primary_conf: np.ndarray,
    shadow_conf: np.ndarray,
    true_classes: Optional[np.ndarray] = None,
) -> AgreementReport:
    n = len(primary_preds)
    agreement_rate = float(np.mean(primary_preds == shadow_preds))
    mean_conf_delta = float(np.mean(shadow_conf - primary_conf))

    primary_only_correct = shadow_only_correct = both_correct = both_wrong = None
    mcnemar_stat = mcnemar_p = None

    if true_classes is not None:
        primary_correct = primary_preds == true_classes
        shadow_correct = shadow_preds == true_classes
        both_correct = int(np.sum(primary_correct & shadow_correct))
        both_wrong = int(np.sum(~primary_correct & ~shadow_correct))
        primary_only_correct = int(np.sum(primary_correct & ~shadow_correct))
        shadow_only_correct = int(np.sum(~primary_correct & shadow_correct))

        # McNemar's test on the discordant pairs. This tests whether the
        # two models disagree on ground-truth-correctness asymmetrically,
        # which is the right question when comparing paired predictions
        # on the identical inputs (not independent samples, so a plain
        # two-proportion z-test would be invalid here).
        b, c = primary_only_correct, shadow_only_correct
        if b + c > 0:
            # Exact binomial McNemar for small discordant counts, chi-square
            # with continuity correction for larger ones.
            if b + c < 25:
                mcnemar_p = float(2 * stats.binom.cdf(min(b, c), b + c, 0.5))
                mcnemar_p = min(mcnemar_p, 1.0)
                mcnemar_stat = None
            else:
                mcnemar_stat = float((abs(b - c) - 1) ** 2 / (b + c))
                mcnemar_p = float(1 - stats.chi2.cdf(mcnemar_stat, df=1))

    return AgreementReport(
        n=n, agreement_rate=agreement_rate,
        primary_only_correct=primary_only_correct,
        shadow_only_correct=shadow_only_correct,
        both_correct=both_correct, both_wrong=both_wrong,
        mean_confidence_delta=mean_conf_delta,
        mcnemar_stat=mcnemar_stat, mcnemar_p_value=mcnemar_p,
    )


@dataclass
class PromotionDecision:
    n_control: int
    n_canary: int
    acc_control: float
    acc_canary: float
    acc_delta: float
    z_stat: float
    p_value: float
    min_sample_met: bool
    significant: bool
    recommendation: str


def evaluate_promotion(
    control_correct: np.ndarray,
    canary_correct: np.ndarray,
    min_sample_per_arm: int = 200,
    alpha: float = 0.05,
    min_practical_delta: float = 0.0,
) -> PromotionDecision:
    """
    Two-proportion z-test comparing canary accuracy to control accuracy.

    This assumes control and canary are independent samples (true for
    canary routing, unlike the paired shadow comparison above, because
    canary and control serve disjoint sets of real requests).

    min_sample_met gates the recommendation on sample size BEFORE looking
    at the p-value. A significant result on 40 canary requests is not
    trustworthy regardless of what the math returns; the gate exists so a
    tiny early sample with a lucky p-value can't trigger a promotion.
    """
    n_c, n_t = len(control_correct), len(canary_correct)
    min_sample_met = min(n_c, n_t) >= min_sample_per_arm

    acc_c = float(np.mean(control_correct)) if n_c > 0 else float("nan")
    acc_t = float(np.mean(canary_correct)) if n_t > 0 else float("nan")
    delta = acc_t - acc_c

    if n_c == 0 or n_t == 0:
        return PromotionDecision(n_c, n_t, acc_c, acc_t, delta, float("nan"),
                                  float("nan"), min_sample_met, False,
                                  "insufficient_data")

    p_pool = (control_correct.sum() + canary_correct.sum()) / (n_c + n_t)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
    z = (acc_t - acc_c) / se if se > 0 else 0.0
    p_value = float(2 * (1 - stats.norm.cdf(abs(z))))
    significant = p_value < alpha

    if not min_sample_met:
        recommendation = "hold_insufficient_sample"
    elif significant and delta > min_practical_delta:
        recommendation = "promote"
    elif significant and delta < -min_practical_delta:
        recommendation = "rollback"
    else:
        recommendation = "hold_no_significant_difference"

    return PromotionDecision(
        n_control=n_c, n_canary=n_t, acc_control=acc_c, acc_canary=acc_t,
        acc_delta=delta, z_stat=float(z), p_value=p_value,
        min_sample_met=min_sample_met, significant=significant,
        recommendation=recommendation,
    )
