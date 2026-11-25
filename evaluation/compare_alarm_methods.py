#!/usr/bin/env python3
"""
Compare two alarm methods on simulated ICU telemetry.

Ground truth comes from the simulator's own state, not from clinicians:
  real      - non-sinus rhythm (the simulator's rhythm event) and no technical artifact
  artifact  - sinus rhythm with a technical artifact (noise)
  quiet     - sinus rhythm, no artifact
Seconds where a rhythm event and an artifact overlap are not scored.

Neither method sees the rhythm or artifact labels. Both see only vital signs:
  heart rate, SpO2, systolic BP and respiratory rate.

Methods:
  baseline - threshold alarm: fires when every one of the last PERSIST samples has
             HR < 50 or HR > 110 or SpO2 < 92.
  model    - logistic regression on a WINDOW-second feature window, trained on
             patients 1-6 and tested on patients 7-8 (split by patient, not by second).

Metrics (test patients only):
  recall             - share of real-event seconds that the method alarms on
  false alarms       - alarm episodes that start on an artifact or quiet second
  false alarms/hour  - false alarm episodes per patient-hour of monitoring

Usage:
  python evaluation/compare_alarm_methods.py
"""

from __future__ import annotations

import os
import random
import sys
from collections import defaultdict, deque

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from icu_telemetry_simulator import (  # noqa: E402
    CardiacRhythm,
    ICUTelemetrySimulator,
    TechnicalArtifact,
)

WINDOW = 10          # seconds of history used by the model
PERSIST = 5          # seconds the baseline condition must hold before alarming
STEPS = 20000        # simulated seconds per run
SEEDS = [42, 7, 2026]
TRAIN_PATIENTS = {1, 2, 3, 4, 5, 6}
TEST_PATIENTS = {7, 8}
CHANNELS = 4         # HR, SpO2, SBP, RR


def label_for(patient) -> str | None:
    nonsinus = patient.current_rhythm != CardiacRhythm.NORMAL_SINUS
    artifact = patient.current_artifact != TechnicalArtifact.NONE
    if nonsinus and artifact:
        return None  # overlap: not scored
    if nonsinus:
        return "real"
    if artifact:
        return "artifact"
    return "quiet"


def window_features(window: np.ndarray) -> np.ndarray:
    """window: (WINDOW, CHANNELS) array. Returns a flat feature vector."""
    last = window[-1]
    mean = window.mean(axis=0)
    std = window.std(axis=0)
    lo = window.min(axis=0)
    hi = window.max(axis=0)
    t = np.arange(window.shape[0])
    hr_slope = np.polyfit(t, window[:, 0], 1)[0]
    return np.concatenate([last, mean, std, lo, hi, [hr_slope]])


def run_seed(seed: int) -> dict:
    random.seed(seed)
    sim = ICUTelemetrySimulator(num_patients=8)
    windows = {p.patient_id: deque(maxlen=WINDOW) for p in sim.patients}
    flag_history = {p.patient_id: deque(maxlen=PERSIST) for p in sim.patients}
    rows = []  # one dict per patient-second, in time order

    for t in range(STEPS):
        sim.update_step()
        for p in sim.patients:
            v = p.vitals
            sample = np.array([v.heart_rate, v.o2_saturation,
                               v.systolic_bp, v.respiratory_rate], dtype=float)
            windows[p.patient_id].append(sample)
            abnormal = (sample[0] < 50) or (sample[0] > 110) or (sample[1] < 92)
            flag_history[p.patient_id].append(abnormal)

            ready = len(windows[p.patient_id]) == WINDOW
            base_alarm = (len(flag_history[p.patient_id]) == PERSIST
                          and all(flag_history[p.patient_id]))
            rows.append({
                "pid": p.patient_id,
                "t": t,
                "label": label_for(p) if ready else None,
                "features": window_features(np.array(windows[p.patient_id]))
                            if ready else None,
                "baseline": bool(base_alarm) if ready else False,
            })

    # Train the model on training patients only.
    train = [r for r in rows if r["pid"] in TRAIN_PATIENTS and r["label"] in ("real", "artifact", "quiet")]
    X_train = np.vstack([r["features"] for r in train])
    y_train = np.array([1 if r["label"] == "real" else 0 for r in train])
    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(max_iter=1000).fit(scaler.transform(X_train), y_train)

    # Score every ready row so alarm episodes can be counted in time order.
    ready_rows = [r for r in rows if r["features"] is not None]
    probs = model.predict_proba(scaler.transform(np.vstack([r["features"] for r in ready_rows])))[:, 1]
    for r, pr in zip(ready_rows, probs):
        r["model"] = bool(pr >= 0.5)

    test_rows = [r for r in ready_rows if r["pid"] in TEST_PATIENTS]
    return summarize(test_rows, n_patients=len(TEST_PATIENTS))


def summarize(test_rows: list[dict], n_patients: int) -> dict:
    result = {}
    seconds = len(test_rows)
    hours = seconds / n_patients / 3600
    for method in ("baseline", "model"):
        real_total = sum(1 for r in test_rows if r["label"] == "real")
        real_alarmed = sum(1 for r in test_rows if r["label"] == "real" and r[method])

        # Count alarm episodes: a False->True edge on each patient's timeline.
        by_patient = defaultdict(list)
        for r in test_rows:
            by_patient[r["pid"]].append(r)
        false_eps = 0
        true_eps = 0
        for pid, seq in by_patient.items():
            prev = False
            for r in seq:
                if r[method] and not prev and r["label"] is not None:
                    if r["label"] == "real":
                        true_eps += 1
                    else:
                        false_eps += 1
                prev = r[method]

        result[method] = {
            "recall": real_alarmed / real_total if real_total else 0.0,
            "false_episodes": false_eps,
            "true_episodes": true_eps,
            "false_per_patient_hour": false_eps / hours if hours else 0.0,
        }
    result["label_counts"] = {
        k: sum(1 for r in test_rows if r["label"] == k) for k in ("real", "artifact", "quiet")
    }
    return result


def main() -> None:
    all_results = []
    for seed in SEEDS:
        res = run_seed(seed)
        all_results.append(res)
        print(f"seed {seed}: labels {res['label_counts']}")
        for m in ("baseline", "model"):
            r = res[m]
            print(f"  {m:9s} recall {r['recall']:.1%}  "
                  f"false alarm episodes {r['false_episodes']}  "
                  f"(per patient-hour {r['false_per_patient_hour']:.2f})  "
                  f"true episodes {r['true_episodes']}")

    print("\nAverage over seeds:")
    for m in ("baseline", "model"):
        rec = np.mean([r[m]["recall"] for r in all_results])
        fp = np.mean([r[m]["false_per_patient_hour"] for r in all_results])
        print(f"  {m:9s} recall {rec:.1%}  false alarms per patient-hour {fp:.2f}")


if __name__ == "__main__":
    main()
