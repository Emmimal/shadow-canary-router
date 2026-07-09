import json
import time
import numpy as np

from data.generators import make_dataset, train_primary_model, train_candidate_model

SEED = 42
np.random.seed(SEED)

X, y = make_dataset(n_samples=6000, seed=SEED)
X_train, y_train = X[:3000], y[:3000]
X_stream = X[3000:]

primary = train_primary_model(X_train, y_train, seed=SEED)
candidate = train_candidate_model(X_train, y_train, seed=SEED)

N_REQUESTS = 500
N_TRIALS = 5

primary_only_trials = []
sync_shadow_trials = []

for trial in range(N_TRIALS):
    t0 = time.perf_counter()
    for i in range(N_REQUESTS):
        x = X_stream[i % len(X_stream)].reshape(1, -1)
        _ = primary.predict_proba(x)
    primary_only_trials.append((time.perf_counter() - t0) * 1000 / N_REQUESTS)

    t0 = time.perf_counter()
    for i in range(N_REQUESTS):
        x = X_stream[i % len(X_stream)].reshape(1, -1)
        _ = primary.predict_proba(x)
        _ = candidate.predict_proba(x)  # synchronous shadow call, blocking
    sync_shadow_trials.append((time.perf_counter() - t0) * 1000 / N_REQUESTS)

result = {
    "n_requests_per_trial": N_REQUESTS,
    "n_trials": N_TRIALS,
    "primary_only_ms": {
        "median": float(np.median(primary_only_trials)),
        "min": float(np.min(primary_only_trials)),
        "max": float(np.max(primary_only_trials)),
    },
    "sync_with_shadow_ms": {
        "median": float(np.median(sync_shadow_trials)),
        "min": float(np.min(sync_shadow_trials)),
        "max": float(np.max(sync_shadow_trials)),
    },
}
result["overhead_pct_median"] = (
    (result["sync_with_shadow_ms"]["median"] - result["primary_only_ms"]["median"])
    / result["primary_only_ms"]["median"] * 100
)

print(json.dumps(result, indent=2))
with open("latency_results.json", "w") as f:
    json.dump(result, f, indent=2)
