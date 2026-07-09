import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stats.compare import compare_shadow_to_primary, evaluate_promotion


def test_agreement_rate_perfect_agreement():
    preds = np.array([0, 1, 2, 0, 1])
    report = compare_shadow_to_primary(preds, preds, np.ones(5) * 0.9, np.ones(5) * 0.9)
    assert report.agreement_rate == 1.0


def test_agreement_rate_zero_agreement():
    p = np.array([0, 0, 0])
    s = np.array([1, 1, 1])
    report = compare_shadow_to_primary(p, s, np.ones(3) * 0.9, np.ones(3) * 0.9)
    assert report.agreement_rate == 0.0


def test_mcnemar_none_when_no_discordant_pairs():
    p = np.array([0, 1, 0, 1])
    s = np.array([0, 1, 0, 1])
    true_c = np.array([0, 1, 0, 1])
    report = compare_shadow_to_primary(p, s, np.ones(4) * 0.9, np.ones(4) * 0.9, true_c)
    assert report.primary_only_correct == 0
    assert report.shadow_only_correct == 0
    assert report.mcnemar_p_value is None


def test_mcnemar_significant_when_clearly_asymmetric():
    rng = np.random.RandomState(0)
    n = 200
    true_c = rng.randint(0, 2, n)
    primary_preds = true_c.copy()  # primary always correct
    shadow_preds = 1 - true_c  # shadow always wrong
    report = compare_shadow_to_primary(
        primary_preds, shadow_preds, np.ones(n) * 0.9, np.ones(n) * 0.9, true_c
    )
    assert report.primary_only_correct == n
    assert report.shadow_only_correct == 0
    assert report.mcnemar_p_value < 0.01


def test_promotion_holds_below_min_sample():
    control = np.array([True] * 40 + [False] * 10)
    canary = np.array([True] * 45 + [False] * 5)
    decision = evaluate_promotion(control, canary, min_sample_per_arm=200)
    assert decision.min_sample_met is False
    assert decision.recommendation == "hold_insufficient_sample"


def test_promotion_recommends_promote_when_canary_clearly_better():
    rng = np.random.RandomState(1)
    control = rng.random(500) < 0.70
    canary = rng.random(500) < 0.90
    decision = evaluate_promotion(control, canary, min_sample_per_arm=200)
    assert decision.min_sample_met is True
    assert decision.recommendation == "promote"


def test_promotion_recommends_rollback_when_canary_clearly_worse():
    rng = np.random.RandomState(2)
    control = rng.random(500) < 0.70
    canary = rng.random(500) < 0.30
    decision = evaluate_promotion(control, canary, min_sample_per_arm=200)
    assert decision.min_sample_met is True
    assert decision.recommendation == "rollback"


def test_promotion_holds_when_no_significant_difference():
    rng = np.random.RandomState(3)
    control = rng.random(500) < 0.70
    canary = rng.random(500) < 0.705
    decision = evaluate_promotion(control, canary, min_sample_per_arm=200)
    assert decision.recommendation == "hold_no_significant_difference"


def test_promotion_handles_empty_arm():
    control = np.array([True, False, True])
    canary = np.array([])
    decision = evaluate_promotion(control, canary, min_sample_per_arm=200)
    assert decision.recommendation == "insufficient_data"
