
# shadow-canary-router

A pure-Python shadow deployment and canary release router for ML models — failure-isolated shadow calls, deterministic hash-based traffic splitting, and statistical promotion gates in one library.

![Python Version](https://img.shields.io/badge/python-3.12-blue) ![License](https://img.shields.io/badge/license-MIT-green)

Most deployment tutorials stop at: replace the model, watch the dashboard, hope. This library handles what comes before that — running a candidate against real traffic without risking it, then deciding, with an actual statistical test and a sample-size gate, whether it's earned the right to take over.

Read the full write-up on EmiTechLogic → [Shadow Deployment and Canary Testing for Machine Learning Models: A Practical Guide](https://emitechlogic.com/shadow-deployment-and-canary-testing-for-machine-learning-models-a-practical-guide/)

## What It Does

```
                    ┌──────────────┐
  Request ────────▶ │ ShadowRouter │ ────▶ primary_pred (served to caller)
                    │              │
                    │  primary ────┼────▶ always served
                    │  shadow ─────┼────▶ logged only, isolated from failures
                    └──────────────┘
                           │
                           ▼
                    shadow_log.db ────▶ compare_shadow_to_primary() ─▶ McNemar's test


                    ┌──────────────┐
  Request ────────▶ │ CanaryRouter │ ────▶ assign_arm(entity_id) ─▶ control OR canary
                    └──────────────┘              (deterministic hash)
                           │
                           ▼
                    canary_log.db ────▶ evaluate_promotion() ─▶ promote / rollback / hold
```

Two routers, one statistics layer:

| Component | Job |
|---|---|
| `ShadowRouter` | Runs a candidate model on every request alongside the primary; primary always serves, shadow failures never propagate to the caller |
| `CanaryRouter` | Routes a deterministic, adjustable percentage of real traffic to a candidate by hashing an entity ID |
| `compare_shadow_to_primary()` | Agreement rate + McNemar's test on paired shadow-vs-primary predictions |
| `evaluate_promotion()` | Two-proportion z-test with a hard minimum-sample gate, returns promote / rollback / hold |

## Installation

```bash
git clone https://github.com/Emmimal/shadow-canary-router.git
cd shadow-canary-router
pip install numpy scipy scikit-learn      # required
pip install pytest                        # optional — to run the test suite
```

No other dependencies. Everything runs on NumPy, SciPy, and scikit-learn.

## Quick Start

```python
from routing.shadow import ShadowRouter
from routing.canary import CanaryRouter
from stats.compare import compare_shadow_to_primary, evaluate_promotion
import numpy as np

# --- Shadow mode: candidate runs on every request, never served ---
router = ShadowRouter(
    primary_predict_proba=primary_model.predict_proba,
    shadow_predict_proba=candidate_model.predict_proba,
    db_path="shadow.db",
)
result = router.predict(request_id="req-1", features=np.array(feature_vector))
print(result.primary_pred)   # what the caller gets
print(result.shadow_pred)    # logged only; None if the shadow model raised

# --- Canary: candidate serves a real, deterministic slice of traffic ---
canary_router = CanaryRouter(
    control_predict_proba=primary_model.predict_proba,
    canary_predict_proba=candidate_model.predict_proba,
    db_path="canary.db",
    canary_pct=0.10,
)
canary_result = canary_router.predict("req-2", entity_id="user-42", features=np.array(feature_vector))
print(canary_result.arm)     # "control" or "canary" — stable for this entity_id

# --- Decide whether to promote, once ground truth lands ---
decision = evaluate_promotion(control_correct, canary_correct, min_sample_per_arm=200)
print(decision.recommendation)   # "promote" | "rollback" | "hold_insufficient_sample" | "hold_no_significant_difference"
```

## Running the Benchmarks

Two runnable scripts reproduce every number in the write-up, seed 42 throughout:

```bash
python3 run_benchmark.py           # shadow agreement stats, traffic-split check, canary ramp + rollback progression
python3 run_latency_benchmark.py   # synchronous shadow-call latency overhead, 5 trials, median reported
```

| Script | What It Shows |
|---|---|
| `run_benchmark.py` | Shadow scenario for a genuinely-better candidate vs. a candidate with a silent column-mismatch defect; traffic-split ratio verification at four percentages; a three-stage canary ramp (5% → 25% → 50%); a five-checkpoint rollback progression |
| `run_latency_benchmark.py` | Per-request latency with and without a synchronous shadow call, median of 5 trials × 500 requests |

## Configuration Reference

```python
ShadowRouter(
    primary_predict_proba,   # callable, always served to the caller
    shadow_predict_proba,    # callable, logged only — exceptions are caught, never raised
    db_path,                 # SQLite path for the shadow log
    batch_id=0,              # groups requests for later analysis
)

CanaryRouter(
    control_predict_proba,
    canary_predict_proba,
    db_path,
    canary_pct,               # float in [0, 1] — fraction of traffic routed to canary
    batch_id=0,
    salt="canary-v1",         # change this to re-randomize entity assignment for a new test
)

evaluate_promotion(
    control_correct,          # bool array — per-request correctness in the control arm
    canary_correct,           # bool array — per-request correctness in the canary arm
    min_sample_per_arm=200,   # gate: hold regardless of p-value below this
    alpha=0.05,
    min_practical_delta=0.0,  # minimum accuracy delta to act on, even if significant
)
```

Choosing `min_sample_per_arm`: 200 is a reasonable default for a binary/multiclass accuracy comparison via a normal approximation. Lower it and you'll promote or roll back faster, at the cost of trusting a shakier approximation; raise it if a wrong early decision is expensive.

## Project Structure

```
shadow-canary-router/
├── routing/
│   ├── shadow.py              # ShadowRouter, failure-isolated shadow calls, SQLite log schema
│   └── canary.py               # CanaryRouter, assign_arm() hash-based traffic split
├── stats/
│   └── compare.py               # compare_shadow_to_primary() [McNemar], evaluate_promotion() [z-test]
├── data/
│   └── generators.py            # synthetic primary/good-candidate/buggy-candidate models
├── tests/
│   ├── test_shadow.py           # 6 tests, incl. the failure-isolation guarantee
│   ├── test_canary.py           # 8 tests, incl. split-ratio + per-entity determinism
│   └── test_compare.py          # 9 tests, incl. promote/rollback/hold decision logic
├── run_benchmark.py             # full pipeline: shadow scenarios, traffic split, canary ramp + rollback
├── run_latency_benchmark.py     # dedicated synchronous shadow-call latency benchmark
└── LICENSE
```

## Performance (CPU only, single-row inference)

| Configuration | Median latency per request |
|---|---|
| Primary only (logistic regression) | 0.077 ms |
| Primary + synchronous shadow call (150-tree random forest) | 8.556 ms |

A synchronous shadow call is roughly 111x slower per request in this benchmark, entirely from the shadow model's own inference cost. `ShadowRouter.predict()` is written synchronously for clarity; a production deployment should queue the shadow call to run asynchronously so it never adds latency to the caller's response.

| Metric | Genuinely-better candidate | Column-mismatch candidate |
|---|---|---|
| Offline accuracy | 87.2% (vs. 71.6% primary) | 38.5% (vs. 71.6% primary) |
| Shadow agreement rate | 80.0% | 41.1% |
| McNemar p-value | < 0.001 | < 0.001 |

## When to Use This

Worth it when you have:
- A candidate model you don't yet trust with real outcomes, and a primary you can't afford to destabilize
- A rollout process where "we eyeballed the dashboard" isn't a good enough promotion criterion
- Multiple candidates in flight where you need a repeatable, code-reviewable decision rule instead of a judgment call each time

Skip it when you have:
- A managed serving platform (SageMaker, Seldon Core) that already provides shadow/canary primitives natively — use those instead of re-implementing this layer
- Traffic volume too low to reach a meaningful sample size in a reasonable time window; the promotion gate will sit on `hold` indefinitely
- A change so small (a minor retrain on the same architecture) that a full shadow-then-canary cycle is more process than the risk justifies

## Known Limitations

- `evaluate_promotion()` uses a two-proportion z-test, which assumes a normal approximation. It's gated on a 200-sample minimum per arm for this reason, but that's a default, not a guarantee — very imbalanced canary percentages can still leave the canary arm thin even at high total traffic.
- `assign_arm()` re-hashes on every call; at very high request volumes, cache the arm assignment per entity instead of re-hashing per request if hashing itself becomes measurable overhead.
- `ShadowRouter` and `CanaryRouter` are both synchronous. Neither queues work to a background thread or process; wrap the shadow/canary call in your own async layer for production use, per the latency benchmark above.
- SQLite is used for both logs, same as Article 10's monitoring store. It comfortably handles the request volumes benchmarked here; move to Postgres for concurrent multi-process writers.
- The "buggy candidate" in `generators.py` simulates one specific defect class (column-order mismatch). It's a stand-in for the broader category of silent feature-pipeline bugs, not an exhaustive test of every way a candidate can fail.
