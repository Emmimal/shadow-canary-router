"""
Shadow deployment router.

The primary model's prediction is always the one returned to the caller.
The shadow model runs on the identical input, on the same request, but its
output never reaches the caller. It is logged for offline comparison only.
"""
from __future__ import annotations

import json
import time
import sqlite3
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np


SCHEMA = """
CREATE TABLE IF NOT EXISTS shadow_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    batch_id INTEGER NOT NULL,
    features TEXT NOT NULL,
    primary_pred INTEGER NOT NULL,
    primary_conf REAL NOT NULL,
    primary_scores TEXT NOT NULL,
    shadow_pred INTEGER,
    shadow_conf REAL,
    shadow_scores TEXT,
    primary_latency_ms REAL NOT NULL,
    shadow_latency_ms REAL,
    shadow_error TEXT,
    true_class INTEGER,
    logged_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_shadow_batch ON shadow_log(batch_id);
"""


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str) -> None:
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA)


@dataclass
class ShadowResult:
    request_id: str
    primary_pred: int
    primary_conf: float
    primary_latency_ms: float
    shadow_pred: Optional[int]
    shadow_conf: Optional[float]
    shadow_latency_ms: Optional[float]
    shadow_error: Optional[str]


class ShadowRouter:
    """
    Wraps a primary model and a shadow model behind one predict() call.

    The caller only ever receives the primary model's output. The shadow
    model's predict is invoked on the same input, timed, caught if it
    raises, and logged. It never affects what the caller gets back and a
    shadow failure never surfaces as a caller-visible error.
    """

    def __init__(
        self,
        primary_predict_proba: Callable[[np.ndarray], np.ndarray],
        shadow_predict_proba: Callable[[np.ndarray], np.ndarray],
        db_path: str,
        batch_id: int = 0,
    ):
        self.primary_predict_proba = primary_predict_proba
        self.shadow_predict_proba = shadow_predict_proba
        self.db_path = db_path
        self.batch_id = batch_id
        init_db(db_path)

    def predict(self, request_id: str, features: np.ndarray, true_class: Optional[int] = None) -> ShadowResult:
        x = features.reshape(1, -1)

        t0 = time.perf_counter()
        primary_scores = self.primary_predict_proba(x)[0]
        primary_latency_ms = (time.perf_counter() - t0) * 1000
        primary_pred = int(np.argmax(primary_scores))
        primary_conf = float(primary_scores[primary_pred])

        shadow_pred = shadow_conf = shadow_latency_ms = None
        shadow_scores = None
        shadow_error = None
        try:
            t0 = time.perf_counter()
            shadow_scores = self.shadow_predict_proba(x)[0]
            shadow_latency_ms = (time.perf_counter() - t0) * 1000
            shadow_pred = int(np.argmax(shadow_scores))
            shadow_conf = float(shadow_scores[shadow_pred])
        except Exception as e:
            # A shadow failure must never propagate to the caller. Log it
            # and move on; primary_pred is already computed and returned.
            shadow_error = f"{type(e).__name__}: {e}"

        self._log(
            request_id, features, primary_pred, primary_conf, primary_scores,
            primary_latency_ms, shadow_pred, shadow_conf, shadow_scores,
            shadow_latency_ms, shadow_error, true_class,
        )

        return ShadowResult(
            request_id=request_id,
            primary_pred=primary_pred,
            primary_conf=primary_conf,
            primary_latency_ms=primary_latency_ms,
            shadow_pred=shadow_pred,
            shadow_conf=shadow_conf,
            shadow_latency_ms=shadow_latency_ms,
            shadow_error=shadow_error,
        )

    def _log(self, request_id, features, primary_pred, primary_conf, primary_scores,
              primary_latency_ms, shadow_pred, shadow_conf, shadow_scores,
              shadow_latency_ms, shadow_error, true_class):
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO shadow_log
                    (request_id, batch_id, features, primary_pred, primary_conf,
                     primary_scores, shadow_pred, shadow_conf, shadow_scores,
                     primary_latency_ms, shadow_latency_ms, shadow_error,
                     true_class, logged_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id, self.batch_id,
                    json.dumps(features.tolist(), cls=_NumpyEncoder),
                    primary_pred, primary_conf,
                    json.dumps(primary_scores, cls=_NumpyEncoder),
                    shadow_pred, shadow_conf,
                    json.dumps(shadow_scores, cls=_NumpyEncoder) if shadow_scores is not None else None,
                    primary_latency_ms, shadow_latency_ms, shadow_error,
                    int(true_class) if true_class is not None else None,
                    time.time(),
                ),
            )


def load_shadow_log(db_path: str, batch_id: Optional[int] = None) -> list[dict]:
    with get_connection(db_path) as conn:
        if batch_id is not None:
            rows = conn.execute(
                "SELECT * FROM shadow_log WHERE batch_id = ? ORDER BY id", (batch_id,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM shadow_log ORDER BY id").fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["features"] = json.loads(d["features"])
        d["primary_scores"] = json.loads(d["primary_scores"])
        d["shadow_scores"] = json.loads(d["shadow_scores"]) if d["shadow_scores"] else None
        out.append(d)
    return out
