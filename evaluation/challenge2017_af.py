#!/usr/bin/env python3
"""
Atrial fibrillation (AF) vs normal sinus rhythm on the PhysioNet/CinC Challenge 2017
single-lead ECG data (AliveCor recordings, 300 Hz).

Download the training set on your own computer from:
  https://physionet.org/content/challenge-2017/1.0.0/
Unzip it so the folder looks like:
  data/challenge2017/training2017/REFERENCE.csv
  data/challenge2017/training2017/A00001.mat, A00002.mat, ...

Labels used: N (normal) and A (AF). Records labeled O (other rhythm) or ~ (too noisy)
are left out, so the task is binary.

Methods (both see only the ECG signal, through R-peak timing):
  baseline - one feature: coefficient of variation of R-R intervals. A single cutoff,
             chosen on the training split.
  model    - logistic regression on several R-R features (CV, RMSSD, share of large
             successive changes, mean heart rate), trained on the training split.

Caveat: this dataset does not give patient IDs, so the split is by recording. The same
person may appear in both splits, which can make results look better than they are.

Usage:
  python evaluation/challenge2017_af.py [path/to/training2017]
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FS = 300  # Hz
DEFAULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", "challenge2017", "training2017")
SEED = 42


def load_labels(folder: str) -> list[tuple[str, str]]:
    path = os.path.join(folder, "REFERENCE.csv")
    with open(path, newline="") as fh:
        return [(rid, lab) for rid, lab in csv.reader(fh) if lab in ("N", "A")]


def r_peak_times(signal: np.ndarray) -> np.ndarray:
    """Return R-peak times in seconds using a band-pass filter and a peak finder."""
    b, a = butter(2, [5 / (FS / 2), 15 / (FS / 2)], btype="band")
    filtered = filtfilt(b, a, signal)
    height = 0.5 * np.max(np.abs(filtered))
    peaks, _ = find_peaks(np.abs(filtered), height=height, distance=int(0.25 * FS))
    return peaks / FS


def rr_features(times: np.ndarray) -> np.ndarray | None:
    if len(times) < 5:
        return None
    rr = np.diff(times)
    rr = rr[(rr > 0.3) & (rr < 2.0)]  # keep physiologic intervals
    if len(rr) < 4:
        return None
    mean_rr = rr.mean()
    cv = rr.std() / mean_rr
    diffs = np.diff(rr)
    rmssd = np.sqrt(np.mean(diffs ** 2)) / mean_rr
    large_change = np.mean(np.abs(diffs) > 0.05)
    mean_hr = 60.0 / mean_rr / 100.0  # scaled so it sits near the other features
    return np.array([cv, rmssd, large_change, mean_hr])


def load_dataset(folder: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X, y, ids = [], [], []
    for rid, lab in load_labels(folder):
        mat_path = os.path.join(folder, rid + ".mat")
        if not os.path.exists(mat_path):
            continue
        signal = loadmat(mat_path)["val"].ravel().astype(float)
        feats = rr_features(r_peak_times(signal))
        if feats is None:
            continue
        X.append(feats)
        y.append(1 if lab == "A" else 0)
        ids.append(rid)
    return np.vstack(X), np.array(y), ids


def metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    tp = int(np.sum((pred == 1) & (truth == 1)))
    tn = int(np.sum((pred == 0) & (truth == 0)))
    fp = int(np.sum((pred == 1) & (truth == 0)))
    fn = int(np.sum((pred == 0) & (truth == 1)))
    return {
        "accuracy": (tp + tn) / len(truth),
        "sensitivity": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "false_alarms": fp,
    }


def main() -> None:
    folder = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    if not os.path.exists(os.path.join(folder, "REFERENCE.csv")):
        print(f"Dataset not found at {folder}.\n"
              "Download training2017.zip from https://physionet.org/content/challenge-2017/1.0.0/ "
              "and unzip it there.")
        return

    X, y, ids = load_dataset(folder)
    print(f"Usable records: {len(y)} (AF: {y.sum()}, normal: {(y == 0).sum()})")

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(y))
    cut = int(0.7 * len(y))
    tr, te = idx[:cut], idx[cut:]

    # Baseline: single CV cutoff chosen on the training split.
    cvs = X[tr, 0]
    best_t, best_acc = None, -1
    for t in np.linspace(0.02, 0.5, 97):
        acc = np.mean(((cvs > t).astype(int)) == y[tr])
        if acc > best_acc:
            best_t, best_acc = t, acc
    base_pred = (X[te, 0] > best_t).astype(int)
    base = metrics(base_pred, y[te])

    # Model: logistic regression on all features.
    scaler = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=1000).fit(scaler.transform(X[tr]), y[tr])
    model_pred = clf.predict(scaler.transform(X[te]))
    model = metrics(model_pred, y[te])

    print(f"Baseline cutoff on CV of R-R intervals: {best_t:.3f}")
    for name, m in (("baseline", base), ("model", model)):
        print(f"  {name:9s} accuracy {m['accuracy']:.1%}  sensitivity {m['sensitivity']:.1%}  "
              f"specificity {m['specificity']:.1%}  false alarms {m['false_alarms']}")


if __name__ == "__main__":
    main()
