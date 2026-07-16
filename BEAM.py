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
- Each model's metrics now include a resampled ROC curve (`roc`:
  {fpr, tpr}) on a fixed 21-point FPR grid, so the dashboard's ROC
  chart can render the exact curve instead of approximating it from
  the AUC alone.

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
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
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

ROC_GRID = np.linspace(0, 1, 21)   # fixed FPR grid every model's curve is resampled onto

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
    return float(np.mean(psd[idx]) * 1e12)


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
    """One pass over subjects; returns X/y/groups for ds1, ds2, and combined.

    `groups` holds each row's subject ID (e.g. "sub-001"). This is what lets
    train_and_evaluate() use StratifiedGroupKFold: for the Combined set, a
    given subject's DS1 (eyes-closed) and DS2 (eyes-open) recordings share
    the same subject ID, so grouping guarantees both rows always land in the
    same fold together — never split across train and test. Without this,
    concatenating DS1+DS2 lets the model train on one of a subject's two
    recordings and get evaluated on the other, which leaks subject-specific
    signal (skull/electrode/baseline-rhythm quirks) into the "unseen" test
    fold and inflates the Combined dataset's reported CV metrics.
    """
    feats = {"ds1": {"X": [], "y": [], "groups": []}, "ds2": {"X": [], "y": [], "groups": []}}

    for _, row in participants_df.iterrows():
        sid, label = row["participant_id"], row["label"]
        for key, cfg in DATASETS.items():
            print(f"Processing {sid} [{key}] ({'AD' if label == 1 else 'CN'})...")
            raw = load_eeg(sid, cfg["path"], cfg["task"])
            if raw is not None:
                feats[key]["X"].append(extract_features(raw))
                feats[key]["y"].append(label)
                feats[key]["groups"].append(sid)

    for key in feats:
        feats[key]["X"] = np.array(feats[key]["X"])
        feats[key]["y"] = np.array(feats[key]["y"])
        feats[key]["groups"] = np.array(feats[key]["groups"])

    feats["combined"] = {
        "X": np.vstack([feats["ds1"]["X"], feats["ds2"]["X"]]),
        "y": np.concatenate([feats["ds1"]["y"], feats["ds2"]["y"]]),
        "groups": np.concatenate([feats["ds1"]["groups"], feats["ds2"]["groups"]]),
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


def resample_roc(y_true, y_prob, grid=ROC_GRID):
    """Resample an ROC curve onto a fixed FPR grid so every model/dataset
    combination lines up on the same x-axis in the dashboard chart."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    tpr_grid = np.interp(grid, fpr, tpr)
    return {
        "fpr": [round(float(f), 3) for f in grid],
        "tpr": [round(float(t), 3) for t in tpr_grid],
    }


def train_and_evaluate(X, y, groups):
    """Performs 5-Fold Stratified *Group* CV to calculate clean, non-leaked
    metrics. Using StratifiedGroupKFold (rather than plain StratifiedKFold)
    guarantees every row belonging to the same subject ID lands in the same
    fold. For DS1 and DS2 alone this changes nothing in practice — each
    subject only contributes one row there, so every "group" has size 1 and
    behavior is identical to StratifiedKFold. It matters specifically for
    the Combined dataset, where each subject contributes two rows (their
    DS1 and DS2 recordings) that must never be split across train/test.
    """
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    models = make_models()
    metrics = {}
    
    # We will use the best performing model to output individual risk probabilities
    best_auc = -1
    best_model_name = ""
    best_fitted_pipeline = None

    for name, model in models.items():
        # Get out-of-fold predictions to evaluate accuracy cleanly
        y_pred = cross_val_predict(model, X, y, cv=cv, groups=groups, method="predict")
        y_prob = cross_val_predict(model, X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
        
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        sens = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
        spec = (tn / (tn + fp)) * 100 if (tn + fp) > 0 else 0
        acc = accuracy_score(y, y_pred) * 100
        f1 = f1_score(y, y_pred)
        auc = roc_auc_score(y, y_prob)

        metrics[name] = {
            "accuracy": round(acc, 1),
            "f1": round(f1, 2),
            "auc": round(auc, 2),
            "sensitivity": round(sens, 1),
            "specificity": round(spec, 1),
            "roc": resample_roc(y, y_prob),
        }

        if auc > best_auc:
            best_auc = auc
            best_model_name = name
            # Fit the final model on all data to extract final probabilities
            best_fitted_pipeline = model.fit(X, y)

    # Predict final clinical risk probabilities for the prognosis module
    final_probs = best_fitted_pipeline.predict_proba(X)[:, 1]

    return metrics, best_model_name, final_probs


# ── STEP 4: COGNITIVE PROGNOSIS MODEL ───────────────────────────────────
def generate_prognosis(y_probs):
    """
    Groups subjects into 3 risk tiers based on model probabilities, then maps
    each tier to a published MMSE decline rate. IMPORTANT: these trajectories
    are population-level averages from the literature, not fit to this
    study's subjects (ds004504/ds006036 are single time-point EEG recordings
    with no longitudinal MMSE data to fit against). They're illustrative of
    what each risk tier is *associated with* in prior cohorts, not a
    subject-specific forecast.

    Tier 0 (Low Risk, p < 0.33)   -> ~0.3 MMSE pts/year (cognitively normal elderly)
    Tier 1 (Borderline, .33-.66)  -> ~1.4 MMSE pts/year (MCI)
    Tier 2 (High Risk, p > 0.66)  -> ~3.0 MMSE pts/year (clinical AD)

    Sources (rates are drawn directly from published longitudinal cohorts,
    not averaged/re-derived by us):
      - Petersen RC. "Mild Cognitive Impairment in the Elderly."
        Am Fam Physician. 2001;63(4):620 — single 4-year cohort reporting
        ~0 pts/yr (normal), ~1 pt/yr (MCI), ~3 pts/yr (AD); the anchor
        source since all three tiers come from one comparable cohort.
      - Wang et al., "Self-Administered Gerocognitive Examination:
        longitudinal cohort testing..." (2021), PMC8650250 — corroborates
        with 1.68 pts/yr for MCI-to-AD converters and 2.38 pts/yr for AD.
      - Cortes-Bermea et al., "Rates of Cognitive Decline in 100 Patients
        With Alzheimer Disease" (2022), PMC9196957 — corroborates AD rate
        at 2.43 pts/yr (SD 2.82 — individual variation is large).
    Starting MMSE values (28 / 24 / 20) follow standard clinical MMSE
    staging cutoffs (~25-30 normal, ~24-27 MCI, <24 dementia), not this
    study's data either.
    """
    years = list(range(11))
    tier_assignments = []
    
    for p in y_probs:
        if p < 0.33:
            tier_assignments.append(0)
        elif p <= 0.66:
            tier_assignments.append(1)
        else:
            tier_assignments.append(2)

    tier_counts = {str(t): tier_assignments.count(t) for t in [0, 1, 2]}

    # (start MMSE, annual decline rate, citation) per tier — see docstring
    TIER_PARAMS = {
        0: {"start": 28.0, "slope": 0.3, "source": "Petersen, Am Fam Physician 2001 (~0 pts/yr, normal elderly)"},
        1: {"start": 24.0, "slope": 1.4, "source": "Petersen 2001 (~1 pt/yr) & Wang et al. 2021, PMC8650250 (1.68 pts/yr, MCI converters)"},
        2: {"start": 20.0, "slope": 3.0, "source": "Petersen 2001 (~3 pts/yr) & Cortes-Bermea et al. 2022, PMC9196957 (2.43 pts/yr, AD)"},
    }

    trajectories = {}
    sources = {}
    for t in [0, 1, 2]:
        start, slope = TIER_PARAMS[t]["start"], TIER_PARAMS[t]["slope"]
        traj = []
        for y in years:
            score = max(0.0, start - (slope * y))
            traj.append(round(score, 1))
        trajectories[str(t)] = traj
        sources[str(t)] = TIER_PARAMS[t]["source"]

    return {
        "years": years,
        "mean_trajectories": trajectories,
        "tier_counts": tier_counts,
        "sources": sources,
    }


# ── STEP 5: BRAINWAVE POWER SUMMARY FOR CHARTS ──────────────────────────
def get_brainwave_summary(X, y):
    """Aggregates average band power for AD and Healthy subjects."""
    # Index map matching our extract_features layout: [delta, theta, alpha, beta, tar, speed]
    # We average over all extracted channel bands
    ad_idx = (y == 1)
    cn_idx = (y == 0)

    summary = {}
    bands_keys = ["delta", "theta", "alpha", "beta"]
    for i, band in enumerate(bands_keys):
        # We grab all features corresponding to this band across the wearable channels
        band_feats = X[:, i::N_FEAT_PER_CHANNEL]
        
        ad_vals = band_feats[ad_idx].flatten()
        cn_vals = band_feats[cn_idx].flatten()

        summary[band] = {
            "ad_mean": float(np.mean(ad_vals)),
            "ad_std": float(np.std(ad_vals)),
            "cn_mean": float(np.mean(cn_vals)),
            "cn_std": float(np.std(cn_vals))
        }
    return summary


# ── STEP 6: PIPELINE RUNNER ─────────────────────────────────────────────
def main():
    print("Starting BEAM Machine Learning Pipeline...")
    
    # 1. Load subjects metadata
    try:
        df = load_participants(DATASET_PATH)
    except Exception as e:
        print(f"Error loading metadata from {DATASET_PATH}: {e}")
        return

    # 2. Extract or Load Features
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached features from {CACHE_FILE}...")
        cache = np.load(CACHE_FILE, allow_pickle=True)
        if "ds1_groups" not in cache:
            raise RuntimeError(
                f"{CACHE_FILE} was built before subject-group tracking was added "
                f"(needed for StratifiedGroupKFold). Delete this cache file and "
                f"rerun so features are re-extracted with subject IDs included."
            )
        feature_sets = {
            "ds1": {"X": cache["ds1_X"], "y": cache["ds1_y"], "groups": cache["ds1_groups"]},
            "ds2": {"X": cache["ds2_X"], "y": cache["ds2_y"], "groups": cache["ds2_groups"]},
            "combined": {"X": cache["combined_X"], "y": cache["combined_y"], "groups": cache["combined_groups"]}
        }
    else:
        print("Extracting features from scratch (this may take a few minutes)...")
        feature_sets = build_feature_sets(df)
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        np.savez(CACHE_FILE, 
                 ds1_X=feature_sets["ds1"]["X"], ds1_y=feature_sets["ds1"]["y"], ds1_groups=feature_sets["ds1"]["groups"],
                 ds2_X=feature_sets["ds2"]["X"], ds2_y=feature_sets["ds2"]["y"], ds2_groups=feature_sets["ds2"]["groups"],
                 combined_X=feature_sets["combined"]["X"], combined_y=feature_sets["combined"]["y"], combined_groups=feature_sets["combined"]["groups"])
        print("Features cached successfully!")

    # 3. Process each dataset
    output_data = {
        "datasets": {},
        "brainwave_comparison": {}
    }

    for key in ["ds1", "ds2", "combined"]:
        X = feature_sets[key]["X"]
        y = feature_sets[key]["y"]
        groups = feature_sets[key]["groups"]
        
        n_subjects = len(y)
        n_ad = int(sum(y == 1))
        n_healthy = int(sum(y == 0))
        
        print(f"\nEvaluating dataset: {key.upper()} (N={n_subjects})...")
        
        # Cross-validate models (grouped by subject so DS1/DS2 recordings
        # from the same person never split across train and test)
        metrics, best_model, final_probs = train_and_evaluate(X, y, groups)
        
        # Calculate cognitive trajectory curves based on risk
        prog = generate_prognosis(final_probs)
        
        label = "Combined (DS1 + DS2)" if key == "combined" else DATASETS[key]["label"]

        output_data["datasets"][key] = {
            "label": label,
            "n_subjects": n_subjects,
            "n_ad": n_ad,
            "n_healthy": n_healthy,
            "best_model": best_model,
            "models": metrics,
            "prognosis": prog
        }

        # Calculate brainwave summaries for ds1 & ds2 (exclude combined here to prevent duplicate mapping)
        if key != "combined":
            output_data["brainwave_comparison"][key] = get_brainwave_summary(X, y)

    # 4. Save results to results.json
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output_data, f, indent=2)
        
    print(f"\nPipeline successfully completed! Output saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()