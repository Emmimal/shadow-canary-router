import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from routing.shadow import ShadowRouter, load_shadow_log


class DummyModel:
    def __init__(self, fixed_class=0, n_classes=3):
        self.fixed_class = fixed_class
        self.n_classes = n_classes

    def predict_proba(self, X):
        out = np.full((X.shape[0], self.n_classes), 0.1 / (self.n_classes - 1))
        out[:, self.fixed_class] = 0.9
        return out


class RaisingModel:
    def predict_proba(self, X):
        raise RuntimeError("shadow model exploded")


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "shadow_test.db")


def test_shadow_never_affects_returned_prediction(db_path):
    router = ShadowRouter(DummyModel(fixed_class=0).predict_proba, DummyModel(fixed_class=1).predict_proba, db_path)
    result = router.predict("req-1", np.zeros(5), true_class=0)
    assert result.primary_pred == 0
    assert result.shadow_pred == 1


def test_shadow_failure_does_not_raise(db_path):
    router = ShadowRouter(DummyModel(fixed_class=0).predict_proba, RaisingModel().predict_proba, db_path)
    result = router.predict("req-2", np.zeros(5), true_class=0)
    assert result.primary_pred == 0
    assert result.shadow_pred is None
    assert result.shadow_error is not None
    assert "RuntimeError" in result.shadow_error


def test_shadow_failure_is_logged(db_path):
    router = ShadowRouter(DummyModel(fixed_class=0).predict_proba, RaisingModel().predict_proba, db_path)
    router.predict("req-3", np.zeros(5), true_class=0)
    log = load_shadow_log(db_path)
    assert len(log) == 1
    assert log[0]["shadow_error"] is not None
    assert log[0]["primary_pred"] == 0


def test_shadow_log_persists_true_class(db_path):
    router = ShadowRouter(DummyModel(0).predict_proba, DummyModel(0).predict_proba, db_path)
    router.predict("req-4", np.zeros(5), true_class=2)
    log = load_shadow_log(db_path)
    assert log[0]["true_class"] == 2


def test_shadow_log_batch_filtering(db_path):
    router1 = ShadowRouter(DummyModel(0).predict_proba, DummyModel(0).predict_proba, db_path, batch_id=1)
    router2 = ShadowRouter(DummyModel(0).predict_proba, DummyModel(0).predict_proba, db_path, batch_id=2)
    router1.predict("req-5", np.zeros(5))
    router2.predict("req-6", np.zeros(5))
    assert len(load_shadow_log(db_path, batch_id=1)) == 1
    assert len(load_shadow_log(db_path, batch_id=2)) == 1
    assert len(load_shadow_log(db_path)) == 2


def test_numpy_scalar_serialization_does_not_crash(db_path):
    # Regression test: np.float64 / np.bool_ payloads must not raise
    # TypeError on json.dumps, the same bug documented in Article 10.
    router = ShadowRouter(DummyModel(0).predict_proba, DummyModel(1).predict_proba, db_path)
    features = np.array([np.float64(1.5), np.float64(2.5), np.bool_(True), 0.0, 0.0])
    result = router.predict("req-7", features.astype(float), true_class=0)
    assert result.primary_pred == 0
