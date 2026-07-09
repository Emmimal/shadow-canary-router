import json
import time
import uuid
import numpy as np

from data.generators import (
    make_dataset, train_primary_model, train_candidate_model,
    train_buggy_candidate_model,
)
from routing.shadow import ShadowRouter, load_shadow_log, init_db as init_shadow_db
from routing.canary import CanaryRouter, load_canary_log, assign_arm, init_db as init_canary_db
from stats.compare import compare_shadow_to_primary, evaluate_promotion

SEED = 42
np.random.seed(SEED)

RESULTS = {}

# ---------------------------------------------------------------------------
# Setup: train primary, a genuinely-better candidate, and a buggy candidate
# ---------------------------------------------------------------------------
X, y = make_dataset(n_samples=6000, seed=SEED)
X_train, y_train = X[:3000], y[:3000]
X_stream, y_stream = X[3000:], y[3000:]  # simulated live traffic, 3000 rows

primary = train_primary_model(X_train, y_train, seed=SEED)
good_candidate = train_candidate_model(X_train, y_train, seed=SEED)
buggy_candidate = train_buggy_candidate_model(X_train, y_train, seed=SEED)

primary_acc = float(np.mean(primary.predict(X_stream) == y_stream))
good_acc = float(np.mean(good_candidate.predict(X_stream) == y_stream))
buggy_acc = float(np.mean(buggy_candidate.predict_proba(X_stream).argmax(axis=1) == y_stream))

print(f"Offline accuracy — primary: {primary_acc:.4f}  good candidate: {good_acc:.4f}  buggy candidate: {buggy_acc:.4f}")
RESULTS["offline_accuracy"] = {"primary": primary_acc, "good_candidate": good_acc, "buggy_candidate": buggy_acc}

# ---------------------------------------------------------------------------
# Scenario 1: Shadow deployment — genuinely better candidate
# ---------------------------------------------------------------------------
db_path_good = "data/shadow_good.db"
import os
if os.path.exists(db_path_good):
    os.remove(db_path_good)
init_shadow_db(db_path_good)

router = ShadowRouter(
    primary_predict_proba=primary.predict_proba,
    shadow_predict_proba=good_candidate.predict_proba,
    db_path=db_path_good, batch_id=1,
)

n_shadow_requests = 1500
latencies_primary_only = []
latencies_with_shadow = []

for i in range(n_shadow_requests):
    rid = str(uuid.uuid4())
    feat = X_stream[i]
    true_c = int(y_stream[i])

    # measure primary-alone latency for comparison (separate timing pass)
    t0 = time.perf_counter()
    _ = primary.predict_proba(feat.reshape(1, -1))
    latencies_primary_only.append((time.perf_counter() - t0) * 1000)

    result = router.predict(rid, feat, true_class=true_c)
    latencies_with_shadow.append(result.primary_latency_ms + (result.shadow_latency_ms or 0))

log_good = load_shadow_log(db_path_good, batch_id=1)
primary_preds = np.array([r["primary_pred"] for r in log_good])
shadow_preds = np.array([r["shadow_pred"] for r in log_good])
primary_conf = np.array([r["primary_conf"] for r in log_good])
shadow_conf = np.array([r["shadow_conf"] for r in log_good])
true_classes = np.array([r["true_class"] for r in log_good])

report_good = compare_shadow_to_primary(primary_preds, shadow_preds, primary_conf, shadow_conf, true_classes)
print("\n--- Shadow scenario: GOOD candidate ---")
print(report_good)
RESULTS["shadow_good_candidate"] = {
    "n": report_good.n,
    "agreement_rate": report_good.agreement_rate,
    "primary_only_correct": report_good.primary_only_correct,
    "shadow_only_correct": report_good.shadow_only_correct,
    "both_correct": report_good.both_correct,
    "both_wrong": report_good.both_wrong,
    "mean_confidence_delta": report_good.mean_confidence_delta,
    "mcnemar_p_value": report_good.mcnemar_p_value,
}

mean_lat_primary = float(np.mean(latencies_primary_only))
mean_lat_shadow = float(np.mean(latencies_with_shadow))
overhead_pct = (mean_lat_shadow - mean_lat_primary) / mean_lat_primary * 100
print(f"Mean latency primary-only: {mean_lat_primary:.4f} ms | with shadow logged: {mean_lat_shadow:.4f} ms | overhead: {overhead_pct:.1f}%")
RESULTS["latency_overhead"] = {
    "mean_ms_primary_only": mean_lat_primary,
    "mean_ms_with_shadow_call": mean_lat_shadow,
    "overhead_pct": overhead_pct,
}

# ---------------------------------------------------------------------------
# Scenario 2: Shadow deployment — buggy candidate (should show up as
# disagreement + shadow-only-wrong, NOT primary errors, since primary
# still serves)
# ---------------------------------------------------------------------------
db_path_buggy = "data/shadow_buggy.db"
if os.path.exists(db_path_buggy):
    os.remove(db_path_buggy)
init_shadow_db(db_path_buggy)

router_buggy = ShadowRouter(
    primary_predict_proba=primary.predict_proba,
    shadow_predict_proba=buggy_candidate.predict_proba,
    db_path=db_path_buggy, batch_id=1,
)
for i in range(n_shadow_requests):
    rid = str(uuid.uuid4())
    feat = X_stream[i]
    true_c = int(y_stream[i])
    router_buggy.predict(rid, feat, true_class=true_c)

log_buggy = load_shadow_log(db_path_buggy, batch_id=1)
primary_preds_b = np.array([r["primary_pred"] for r in log_buggy])
shadow_preds_b = np.array([r["shadow_pred"] for r in log_buggy])
primary_conf_b = np.array([r["primary_conf"] for r in log_buggy])
shadow_conf_b = np.array([r["shadow_conf"] for r in log_buggy])
true_classes_b = np.array([r["true_class"] for r in log_buggy])

report_buggy = compare_shadow_to_primary(primary_preds_b, shadow_preds_b, primary_conf_b, shadow_conf_b, true_classes_b)
print("\n--- Shadow scenario: BUGGY candidate ---")
print(report_buggy)
RESULTS["shadow_buggy_candidate"] = {
    "n": report_buggy.n,
    "agreement_rate": report_buggy.agreement_rate,
    "primary_only_correct": report_buggy.primary_only_correct,
    "shadow_only_correct": report_buggy.shadow_only_correct,
    "both_correct": report_buggy.both_correct,
    "both_wrong": report_buggy.both_wrong,
    "mean_confidence_delta": report_buggy.mean_confidence_delta,
    "mcnemar_p_value": report_buggy.mcnemar_p_value,
}

# ---------------------------------------------------------------------------
# Scenario 3: Traffic split verification — does assign_arm actually produce
# the requested ratio, and is it stable per entity across repeated calls?
# ---------------------------------------------------------------------------
entity_ids = [f"user_{i}" for i in range(20000)]
for pct in [0.05, 0.10, 0.25, 0.50]:
    arms = [assign_arm(eid, pct) for eid in entity_ids]
    observed = arms.count("canary") / len(arms)
    # stability check: same entity, same pct, called twice -> same arm
    stable = all(assign_arm(eid, pct) == assign_arm(eid, pct) for eid in entity_ids[:500])
    print(f"Requested canary pct: {pct:.2f} | observed: {observed:.4f} | stable: {stable}")
    RESULTS.setdefault("traffic_split_check", {})[str(pct)] = {"observed": observed, "stable": stable}

# ---------------------------------------------------------------------------
# Scenario 4: Canary promotion decision — good candidate, ramping 5% -> 25% -> 50%
# ---------------------------------------------------------------------------
db_path_canary_good = "data/canary_good.db"
if os.path.exists(db_path_canary_good):
    os.remove(db_path_canary_good)
init_canary_db(db_path_canary_good)

ramp_stages = [0.05, 0.25, 0.50]
canary_results = []
offset = 0
for stage_idx, pct in enumerate(ramp_stages):
    router_c = CanaryRouter(
        control_predict_proba=primary.predict_proba,
        canary_predict_proba=good_candidate.predict_proba,
        db_path=db_path_canary_good, canary_pct=pct, batch_id=stage_idx + 1,
    )
    n_this_stage = 800
    for i in range(n_this_stage):
        idx = offset + i
        eid = f"user_{idx}"
        feat = X_stream[idx % len(X_stream)]
        true_c = int(y_stream[idx % len(y_stream)])
        router_c.predict(str(uuid.uuid4()), eid, feat, true_class=true_c)
    offset += n_this_stage

    log = load_canary_log(db_path_canary_good, batch_id=stage_idx + 1)
    control_rows = [r for r in log if r["arm"] == "control"]
    canary_rows = [r for r in log if r["arm"] == "canary"]
    control_correct = np.array([r["predicted_class"] == r["true_class"] for r in control_rows])
    canary_correct = np.array([r["predicted_class"] == r["true_class"] for r in canary_rows])

    decision = evaluate_promotion(control_correct, canary_correct, min_sample_per_arm=200)
    print(f"\nStage {stage_idx+1} (canary_pct={pct}): n_control={decision.n_control} n_canary={decision.n_canary} "
          f"acc_control={decision.acc_control:.4f} acc_canary={decision.acc_canary:.4f} "
          f"delta={decision.acc_delta:+.4f} p={decision.p_value:.4f} -> {decision.recommendation}")
    canary_results.append({
        "stage": stage_idx + 1, "canary_pct": pct,
        "n_control": decision.n_control, "n_canary": decision.n_canary,
        "acc_control": decision.acc_control, "acc_canary": decision.acc_canary,
        "acc_delta": decision.acc_delta, "p_value": decision.p_value,
        "recommendation": decision.recommendation,
    })
RESULTS["canary_ramp_good_candidate"] = canary_results

# ---------------------------------------------------------------------------
# Scenario 5: Canary promotion decision — buggy candidate, single stage,
# should trigger rollback once enough samples accumulate
# ---------------------------------------------------------------------------
db_path_canary_buggy = "data/canary_buggy.db"
if os.path.exists(db_path_canary_buggy):
    os.remove(db_path_canary_buggy)
init_canary_db(db_path_canary_buggy)


def buggy_predict_proba(X):
    return buggy_candidate.predict_proba(X)


router_cb = CanaryRouter(
    control_predict_proba=primary.predict_proba,
    canary_predict_proba=buggy_predict_proba,
    db_path=db_path_canary_buggy, canary_pct=0.10, batch_id=1,
)

# check decision at small sample (should hold) and at full sample (should catch it)
checkpoint_sizes = [50, 150, 400, 1000, 2500]
buggy_canary_progression = []
n_total = max(checkpoint_sizes)
rows_so_far = []
for i in range(n_total):
    idx = i
    eid = f"buyer_{idx}"
    feat = X_stream[idx % len(X_stream)]
    true_c = int(y_stream[idx % len(y_stream)])
    router_cb.predict(str(uuid.uuid4()), eid, feat, true_class=true_c)
    if (i + 1) in checkpoint_sizes:
        log = load_canary_log(db_path_canary_buggy, batch_id=1)
        control_rows = [r for r in log if r["arm"] == "control"]
        canary_rows = [r for r in log if r["arm"] == "canary"]
        control_correct = np.array([r["predicted_class"] == r["true_class"] for r in control_rows])
        canary_correct = np.array([r["predicted_class"] == r["true_class"] for r in canary_rows])
        decision = evaluate_promotion(control_correct, canary_correct, min_sample_per_arm=200)
        print(f"\nTotal requests={i+1}: n_control={decision.n_control} n_canary={decision.n_canary} "
              f"acc_control={decision.acc_control:.4f} acc_canary={decision.acc_canary:.4f} "
              f"delta={decision.acc_delta:+.4f} p={decision.p_value:.4f} min_sample_met={decision.min_sample_met} "
              f"-> {decision.recommendation}")
        buggy_canary_progression.append({
            "total_requests": i + 1, "n_control": decision.n_control, "n_canary": decision.n_canary,
            "acc_control": decision.acc_control, "acc_canary": decision.acc_canary,
            "acc_delta": decision.acc_delta, "p_value": decision.p_value,
            "min_sample_met": decision.min_sample_met, "recommendation": decision.recommendation,
        })
RESULTS["canary_buggy_progression"] = buggy_canary_progression

with open("results.json", "w") as f:
    json.dump(RESULTS, f, indent=2)

print("\n\nAll results saved to results.json")
