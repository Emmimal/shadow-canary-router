import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from routing.canary import CanaryRouter, assign_arm, load_canary_log


class DummyModel:
    def __init__(self, fixed_class=0, n_classes=3):
        self.fixed_class = fixed_class
        self.n_classes = n_classes

    def predict_proba(self, X):
        out = np.full((X.shape[0], self.n_classes), 0.1 / (self.n_classes - 1))
        out[:, self.fixed_class] = 0.9
        return out


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "canary_test.db")


def test_assignment_is_deterministic_per_entity():
    for _ in range(3):
        assert assign_arm("user_123", 0.20) == assign_arm("user_123", 0.20)


def test_assignment_ratio_within_tolerance():
    entities = [f"u{i}" for i in range(50000)]
    for pct in [0.05, 0.30, 0.50, 0.90]:
        observed = sum(assign_arm(e, pct) == "canary" for e in entities) / len(entities)
        assert abs(observed - pct) < 0.01


def test_zero_percent_routes_everyone_to_control():
    entities = [f"u{i}" for i in range(1000)]
    assert all(assign_arm(e, 0.0) == "control" for e in entities)


def test_hundred_percent_routes_everyone_to_canary():
    entities = [f"u{i}" for i in range(1000)]
    assert all(assign_arm(e, 1.0) == "canary" for e in entities)


def test_invalid_pct_raises(db_path):
    with pytest.raises(ValueError):
        CanaryRouter(DummyModel(0).predict_proba, DummyModel(1).predict_proba, db_path, canary_pct=1.5)


def test_changing_salt_reassigns_entities():
    pct = 0.5
    entities = [f"u{i}" for i in range(2000)]
    arms_v1 = [assign_arm(e, pct, salt="v1") for e in entities]
    arms_v2 = [assign_arm(e, pct, salt="v2") for e in entities]
    disagreement = sum(a != b for a, b in zip(arms_v1, arms_v2)) / len(entities)
    # A new salt should reshuffle assignments substantially, not leave them
    # identical (which would defeat the purpose of re-randomizing).
    assert disagreement > 0.3


def test_router_logs_correct_arm(db_path):
    router = CanaryRouter(DummyModel(0).predict_proba, DummyModel(1).predict_proba, db_path, canary_pct=1.0)
    router.predict("req-1", "user-a", np.zeros(5), true_class=1)
    log = load_canary_log(db_path)
    assert log[0]["arm"] == "canary"
    assert log[0]["predicted_class"] == 1


def test_same_entity_always_same_arm_across_calls(db_path):
    router = CanaryRouter(DummyModel(0).predict_proba, DummyModel(1).predict_proba, db_path, canary_pct=0.5)
    arms = set()
    for _ in range(10):
        r = router.predict("req", "same-user", np.zeros(5))
        arms.add(r.arm)
    assert len(arms) == 1
