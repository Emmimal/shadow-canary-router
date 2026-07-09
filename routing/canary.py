"""
Canary traffic router.

Unlike shadow deployment, the canary model's prediction IS served to a
subset of real traffic. Routing must be deterministic per entity (the same
user should not flip between canary and control on every request) and the
split ratio must be adjustable without redeploying.
"""
from __future__ import annotations

import hashlib
import json
import time
import sqlite3
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


SCHEMA = """
CREATE TABLE IF NOT EXISTS canary_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    batch_id INTEGER NOT NULL,
    arm TEXT NOT NULL,           -- 'control' | 'canary'
    canary_pct REAL NOT NULL,
    predicted_class INTEGER NOT NULL,
    confidence REAL NOT NULL,
    true_class INTEGER,
    logged_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_canary_batch ON canary_log(batch_id);
CREATE INDEX IF NOT EXISTS idx_canary_arm ON canary_log(arm);
"""


def get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str) -> None:
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA)


def assign_arm(entity_id: str, canary_pct: float, salt: str = "canary-v1") -> str:
    """
    Deterministic assignment: the same entity_id always lands in the same
    arm for a given canary_pct and salt. Hashing instead of a random draw
    means a user does not flip between control and canary on every request,
    which would make their individual experience incoherent and would
    correlate errors with request timing instead of with the arm itself.

    Changing `salt` re-randomizes every entity's assignment. Use this when
    starting a new canary test so the previous test's assignment doesn't
    bias the new one.
    """
    digest = hashlib.sha256(f"{salt}:{entity_id}".encode()).hexdigest()
    # Take the first 8 hex chars as an integer, map to [0, 1)
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "canary" if bucket < canary_pct else "control"


@dataclass
class CanaryResult:
    request_id: str
    entity_id: str
    arm: str
    predicted_class: int
    confidence: float


class CanaryRouter:
    def __init__(
        self,
        control_predict_proba: Callable[[np.ndarray], np.ndarray],
        canary_predict_proba: Callable[[np.ndarray], np.ndarray],
        db_path: str,
        canary_pct: float,
        batch_id: int = 0,
        salt: str = "canary-v1",
    ):
        if not 0.0 <= canary_pct <= 1.0:
            raise ValueError(f"canary_pct must be in [0, 1], got {canary_pct}")
        self.control_predict_proba = control_predict_proba
        self.canary_predict_proba = canary_predict_proba
        self.db_path = db_path
        self.canary_pct = canary_pct
        self.batch_id = batch_id
        self.salt = salt
        init_db(db_path)

    def predict(self, request_id: str, entity_id: str, features: np.ndarray,
                true_class: Optional[int] = None) -> CanaryResult:
        arm = assign_arm(entity_id, self.canary_pct, self.salt)
        x = features.reshape(1, -1)
        fn = self.canary_predict_proba if arm == "canary" else self.control_predict_proba
        scores = fn(x)[0]
        pred = int(np.argmax(scores))
        conf = float(scores[pred])

        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO canary_log
                    (request_id, entity_id, batch_id, arm, canary_pct,
                     predicted_class, confidence, true_class, logged_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (request_id, entity_id, self.batch_id, arm, self.canary_pct,
                 pred, conf, int(true_class) if true_class is not None else None,
                 time.time()),
            )

        return CanaryResult(request_id=request_id, entity_id=entity_id, arm=arm,
                             predicted_class=pred, confidence=conf)


def load_canary_log(db_path: str, batch_id: Optional[int] = None) -> list[dict]:
    with get_connection(db_path) as conn:
        if batch_id is not None:
            rows = conn.execute(
                "SELECT * FROM canary_log WHERE batch_id = ? ORDER BY id", (batch_id,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM canary_log ORDER BY id").fetchall()
    return [dict(r) for r in rows]
