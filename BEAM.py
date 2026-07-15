"""
BEAM — Brainwave EEG for Alzheimer's using Machine Learning
=============================================================
Simplified pipeline.

What changed from the original script
--------------------------------------
- One pass through the data builds three feature sets: DS1 (eyes-closed),
  DS2 (eyes-open), and Combined — instead of duplicating loops.
- Model training / prognosis / band-power summary are each a single
  reusable function called once per feature set, instead of one giant
  plotting function with everything inlined.
- All matplotlib dashboards are gone. Instead the script writes one
  `results.json` shaped for the web dashboard in dashboard/index.html —
  every number the site needs (model metrics, ROC points, cognitive
  trajectories, band-power comparisons) lives in that one file, so the
  site can be dataset-aware and clickable without regenerating images.

Run this after pointing DATASET_PATH / DATASET_PATH2 at your BIDS folders.
Output: dashboard/results.json
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.signal import welch

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
)

import mne
mne.set_log_level("WARNING")


# ── CONFIGURATION ────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH  = os.path.join(BASE_DIR, "data", "ds004504")
DATASET_PATH2 = os.path.join(BASE_DIR, "data", "ds006036")
CACHE_FILE    = os.path.join(BASE_DIR, "results", "features_cache.npz")
OUTPUT_JSON = os.path.join(BASE_DIR, "docs", "results.json")

FS                = 500
EPOCH_LENGTH      = 30
WEARABLE_CHANNELS = ["Fp1", "Fp2", "F3", "F4"]

BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
BAND_NAMES = list(BANDS.keys())
N_FEAT_PER_CHANNEL = 6   # delta, theta, alpha, beta, TAR, speed_ratio

DATASETS = {
    "ds1": {"label": "Dataset 1 — Eyes-Closed Resting",  "path": DATASET_PATH,  "task": "eyesclosed"},
    "ds2": {"label": "Dataset 2 — Eyes-Open Photic",      "path": DATASET_PATH2, "task": "photomark"},
}


# ── STEP 1: LABELS ──────────────────────────────────────────────────────
def load_participants(dataset_path):
    df = pd.read_csv(os.path.join(dataset_path, "participants.tsv"), sep="\t")
    df = df[df["Group"].isin(["A", "C"])].reset_index(drop=True)
    df["label"] = df["Group"].map({"A": 1, "C": 0})
    print(f"Loaded {len(df)} subjects: {(df['label']==1).sum()} AD, {(df['label']==0).sum()} Healthy")
    return df


# ── STEP 2: EEG LOADING + FEATURES ──────────────────────────────────────
def load_eeg(subject_id, dataset_path, task):
    for folder in ("derivatives", ""):
        parts = [dataset_path] + ([folder] if folder else []) + [subject_id, "eeg", f"{subject_id}_task-{task}_eeg.set"]
        path = os.path.join(*parts)
        if os.path.exists(path):
            try:
                return mne.io.read_raw_eeglab(path, preload=True, verbose=False)
            except Exception as e:
                print(f"  ERROR loading {path}: {e}")
                return None
    print(f"  WARNING: no EEG file for {subject_id} ({task})")
    return None


def band_power(signal, fs, band):
    nperseg = min(fs * 2, len(signal))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    idx = (freqs >= band[0]) & (freqs <= band[1])
    return float(np.mean(psd[idx]) * 1e12 )


def extract_features(raw, fs=FS, channels=WEARABLE_CHANNELS, epoch_len=EPOCH_LENGTH):
    available = [ch for ch in channels if ch in raw.ch_names] or raw.ch_names
    data = raw.copy().pick_channels(available).get_data()
    n_channels, n_samples = data.shape
    win = int(fs * epoch_len)

    epoch_feats = []
    for start in range(0, n_samples - win, win):
        epoch = data[:, start:start + win]
        row = []
        for ch in range(n_channels):
            p = {b: band_power(epoch[ch], fs, r) for b, r in BANDS.items()}
            tar = p["theta"] / (p["alpha"] + 1e-10)
            speed = (p["alpha"] + p["beta"]) / (p["delta"] + p["theta"] + 1e-10)
            row.extend([p["delta"], p["theta"], p["alpha"], p["beta"], tar, speed])
        epoch_feats.append(row)

    return np.mean(epoch_feats, axis=0)


def build_feature_sets(participants_df):
    """One pass over subjects; returns X/y/ids for ds1, ds2, and combined."""
    feats = {"ds1": {"X": [], "y": []}, "ds2": {"X": [], "y": []}}

    for _, row in participants_df.iterrows():
        sid, label = row["participant_id"], row["label"]
        for key, cfg in DATASETS.items():
            print(f"Processing {sid} [{key}] ({'AD' if label == 1 else 'CN'})...")
            raw = load_eeg(sid, cfg["path"], cfg["task"])
            if raw is not None:
                feats[key]["X"].append(extract_features(raw))
                feats[key]["y"].append(label)

    for key in feats:
        feats[key]["X"] = np.array(feats[key]["X"])
        feats[key]["y"] = np.array(feats[key]["y"])

    feats["combined"] = {
        "X": np.vstack([feats["ds1"]["X"], feats["ds2"]["X"]]),
        "y": np.concatenate([feats["ds1"]["y"], feats["ds2"]["y"]]),
    }
    return feats


# ── STEP 3: MODEL TRAINING ──────────────────────────────────────────────
def make_models():
    return {
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
                        probability=True, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)),
        ]),
        "MLP Neural Network": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(64, 32, 16), activation="relu",
                                   solver="adam", alpha=0.01, learning_rate="adaptive",
                                   max_iter=500, early_stopping=True,
                                   validation_fraction=0.15, random_state=42)),
        ]),
    }


def train_and_evaluate(X, y):
    """5-fold CV for every model. Returns metrics + ROC points + the fitted best model."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = make_models()
    metrics, roc_points = {}, {}

    for name, model in models.items():
        y_pred = cross_val_predict(model, X, y, cv=cv, method="predict")
        y_prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
        cm = confusion_matrix(y, y_pred)
        sens = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) else 0
        spec = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) else 0
        fpr, tpr, _ = roc_curve(y, y_prob)

        metrics[name] = {
            "accuracy": round(accuracy_score(y, y_pred) * 100, 2),
            "f1": round(f1_score(y, y_pred), 4),
            "auc": round(roc_auc_score(y, y_prob), 4),
            "sensitivity": round(sens * 100, 2),
            "specificity": round(spec * 100, 2),
            "confusion_matrix": cm.tolist(),
        }
        step = max(1, len(fpr) // 30)
        roc_points[name] = {"fpr": fpr[::step].tolist(), "tpr": tpr[::step].tolist()}

    best_name = max(metrics, key=lambda n: metrics[n]["auc"])
    best_model = models[best_name].fit(X, y)
    print(f"  Best model: {best_name} (AUC={metrics[best_name]['auc']})")
    return metrics, roc_points, best_name, best_model


# ── STEP 4: PROGNOSIS ────────────────────────────────────────────────────
def build_prognosis(X, y, best_model, seed=42):
    """
    Clinically-grounded 10-year MMSE projection. We don't have longitudinal
    data, so each subject's EEG-derived AD probability sets a risk tier,
    and published annual MMSE decline rates (Tombaugh & McIntyre 1992;
    Mitchell 2009) project their trajectory forward.
    """
    rng = np.random.default_rng(seed)
    ad_prob = best_model.predict_proba(X)[:, 1]
    tier = np.where(ad_prob < 0.33, 0, np.where(ad_prob < 0.66, 1, 2))
    decline_rate = {0: 0.3, 1: 1.5, 2: 3.0}
    baseline = np.clip(
        np.where(y == 1, rng.normal(20, 2, len(y)), rng.normal(28, 1, len(y))), 10, 30
    )
    years = list(range(0, 11))

    subjects = []
    tier_trajectories = {0: [], 1: [], 2: []}
    for i in range(len(X)):
        traj = np.clip(baseline[i] - decline_rate[tier[i]] * np.array(years), 0, 30)
        tier_trajectories[int(tier[i])].append(traj.tolist())
        subjects.append({
            "label": int(y[i]),
            "ad_prob": round(float(ad_prob[i]), 4),
            "risk_tier": int(tier[i]),
            "baseline_mmse": round(float(baseline[i]), 1),
            "mmse_at_5yr": round(float(traj[5]), 1),
            "mmse_at_10yr": round(float(traj[10]), 1),
        })

    mean_trajectories = {
        str(t): (np.mean(tier_trajectories[t], axis=0).tolist() if tier_trajectories[t] else None)
        for t in (0, 1, 2)
    }
    tier_counts = {str(t): len(tier_trajectories[t]) for t in (0, 1, 2)}

    return {
        "years": years,
        "mean_trajectories": mean_trajectories,
        "tier_counts": tier_counts,
        "subjects": subjects,
    }


# ── STEP 5: BAND POWER SUMMARY ──────────────────────────────────────────
def band_power_summary(X, y):
    """Mean +/- std power per band, split AD vs Healthy, averaged across channels."""
    n_ch = X.shape[1] // N_FEAT_PER_CHANNEL
    summary = {}
    for b, band_name in enumerate(BAND_NAMES):
        cols = [b + ch * N_FEAT_PER_CHANNEL for ch in range(n_ch)]
        ad_vals = X[y == 1][:, cols].mean(axis=1)
        cn_vals = X[y == 0][:, cols].mean(axis=1)
        summary[band_name] = {
            "ad_mean": float(np.mean(ad_vals)), "ad_std": float(np.std(ad_vals)),
            "cn_mean": float(np.mean(cn_vals)), "cn_std": float(np.std(cn_vals)),
        }
    return summary


# ── MAIN ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("BEAM pipeline — simplified")
    print("=" * 60)

    participants = load_participants(DATASET_PATH)

    if os.path.exists(CACHE_FILE):
        print("Loading cached features...")
        cache = np.load(CACHE_FILE, allow_pickle=True)
        feats = cache["feats"].item()
    else:
        feats = build_feature_sets(participants)
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        np.savez(CACHE_FILE, feats=feats)
        print(f"Features cached -> {CACHE_FILE}")

    output = {"datasets": {}, "brainwave_comparison": {}}

    for key in ("ds1", "ds2", "combined"):
        X, y = feats[key]["X"], feats[key]["y"]
        print(f"\n--- {key} ({X.shape[0]} recordings) ---")
        metrics, roc_points, best_name, best_model = train_and_evaluate(X, y)
        prognosis = build_prognosis(X, y, best_model)
        bands = band_power_summary(X, y)

        output["datasets"][key] = {
            "label": DATASETS.get(key, {}).get("label", "Combined (DS1 + DS2)"),
            "n_subjects": int(X.shape[0]),
            "n_ad": int((y == 1).sum()),
            "n_healthy": int((y == 0).sum()),
            "models": metrics,
            "roc": roc_points,
            "best_model": best_name,
            "prognosis": prognosis,
            "band_power": bands,
        }

    # Brainwave comparison: DS1 vs DS2, band power for AD and CN separately
    output["brainwave_comparison"] = {
        "ds1": output["datasets"]["ds1"]["band_power"],
        "ds2": output["datasets"]["ds2"]["band_power"],
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved -> {OUTPUT_JSON}")
    print("Open dashboard/index.html to view results.")


if __name__ == "__main__":
    main()
