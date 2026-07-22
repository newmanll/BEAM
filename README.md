# BEAM
**Brainwave EEG for Alzheimer's using Machine Learning**

BEAM is a research pipeline and interactive dashboard that classifies Alzheimer's disease (AD) from resting-state EEG using a wearable-feasible feature set (4 frontal channels) and three classical ML models: SVM, Random Forest, and an MLP neural network. It runs on two paired OpenNeuro cohorts — the same 65 subjects recorded under both an eyes-closed and an eyes-open protocol — and evaluates every model with subject-grouped 5-fold cross-validation.

> **Research use only.** BEAM is a proof-of-concept classification pipeline, not a diagnostic tool. It detects a statistical EEG pattern *associated with* AD status in this specific sample — it has no access to amyloid, tau, or any ground-truth pathology. See [Limitations](#limitations) before drawing any clinical conclusions from it.

---

## Table of Contents

- [Overview](#overview)
- [Datasets](#datasets)
- [Pipeline](#pipeline)
- [Models](#models)
- [Results](#results)
- [Dashboard](#dashboard)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Use of LLMs in This Project](#use-of-llms-in-this-project)
- [References](#references)

---

## Overview

- **Input:** resting-state scalp EEG, two protocols, same 65 subjects (36 AD, 29 healthy controls)
- **Feature extraction:** Welch's method → power spectral density → 4 band powers (delta, theta, alpha, beta) + theta/alpha ratio + a fast/slow "speed" ratio, per channel, per subject → 24 features
- **Models:** SVM (RBF kernel), Random Forest (200 trees), MLP (3 hidden layers)
- **Evaluation:** 5-fold `StratifiedGroupKFold` cross-validation — grouped by subject, so a person's eyes-closed and eyes-open recordings are never split across train/test
- **Output:** `results.json` consumed by a Chart.js dashboard (model comparison, ROC curves, cognitive-risk trajectories, band-power comparisons — all click-through for detail)

## Datasets

| | DS1 — `ds004504` | DS2 — `ds006036` |
|---|---|---|
| Protocol | Eyes-closed, resting | Eyes-open, intermittent photic stimulation |
| Subjects used | 65 (36 AD, 29 CN) | Same 65 subjects |
| Montage | 19-channel, 10-20 system, 500 Hz | Same montage and sampling rate |
| Source | Miltiadous et al. (2023), *Data* 8(6), 95 | Miltiadous et al. (2025), *Data* 10(5), 64 |

Both cohorts were assessed with the MMSE (Mini-Mental State Examination). Recording the same subjects under two protocols allows a direct test of whether an EEG-AD signal generalizes across conditions — eyes-open specifically targets **alpha reactivity**, the normal sharp drop in alpha power when the eyes open, which is known to be blunted in AD.

## Pipeline

```
raw EEG (.set)
   │
   ▼
load 4 wearable channels: Fp1, Fp2, F3, F4
   │
   ▼
30-second, non-overlapping epochs
   │
   ▼
Welch's method (FFT-based PSD, per channel, per epoch)
   │
   ▼
band power: delta / theta / alpha / beta  +  TAR  +  speed ratio
   │
   ▼
average across epochs → 24-feature vector per subject
   │
   ▼
5-fold StratifiedGroupKFold CV × {SVM, Random Forest, MLP}
   │
   ▼
results.json → dashboard
```

The 4-channel restriction (out of the full 19-channel montage) is a deliberate design choice, not a data limitation — it simulates what a low-cost, consumer-wearable EEG headband could realistically capture, since the long-term motivation is accessible screening rather than replicating clinical-grade EEG.

## Models

| Model | Core idea | Why it's here |
|---|---|---|
| **SVM** (RBF kernel) | Finds the widest-margin boundary between AD and Healthy in feature space | Few distributional assumptions — a solid baseline for small, high-dimensional data |
| **Random Forest** (200 trees) | Averages many randomized decision trees (bagging + random feature subsets per split) | Robust to noisy biological data; gives feature importances for free |
| **MLP** (64 → 32 → 16) | Learns non-linear interactions between bands via a feed-forward network | Most flexible — but also the most prone to overfitting on 65 subjects; uses early stopping + a validation split to guard against it |

## Results

Representative 5-fold CV metrics (DS1, eyes-closed):

| Model | Accuracy | F1 | AUC | Sensitivity | Specificity |
|---|---|---|---|---|---|
| SVM | 78.5% | 0.79 | 0.85 | 80.6% | 75.9% |
| Random Forest | 84.6% | 0.85 | 0.91 | 86.1% | 82.8% |
| MLP Neural Network | 81.5% | 0.82 | 0.88 | 83.3% | 79.3% |

All three models land in a similar performance range given the sample size — treat this as "three models cluster together," not a decisive winner. Full per-dataset results (DS1 / DS2 / Combined) are in the dashboard.

## Dashboard

`dashboard/index.html` is a self-contained, dark-themed Chart.js dashboard that reads `results.json` and renders:
- Model comparison (accuracy/F1/AUC) and ROC curves, per dataset
- A cognitive-risk trajectory chart (3 risk tiers, decline rates sourced from published literature — see [References](#references))
- Band-power comparison across protocols and diagnosis groups
- Click-through detail drawers on every chart element

If `results.json` isn't found, it falls back to bundled synthetic sample data so the UI is viewable without running the full pipeline first.

## Repository Structure

```
beam_pipeline.py               # main pipeline: features → CV → results.json
beam_generalization_study.py   # cross-protocol transfer + channel-ablation study
requirements.txt               # Python dependencies
dashboard/
  └── index.html                # interactive results dashboard
data/
  ├── ds004504/                 # DS1 (not included — see Getting Started)
  └── ds006036/                 # DS2 (not included — see Getting Started)
results/
  ├── features_cache.npz        # cached extracted features (generated)
  └── results.json              # pipeline output, read by the dashboard (generated)
```

## Getting Started

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the datasets from OpenNeuro and place them under data/
#    https://openneuro.org/datasets/ds004504
#    https://openneuro.org/datasets/ds006036

# 3. Run the main pipeline
python beam_pipeline.py
#    → writes results/results.json

# 4. Open the dashboard
#    dashboard/index.html can be opened directly, or served locally:
python -m http.server --directory dashboard 8000
#    then visit http://localhost:8000

# 5. (optional) Run the cross-protocol / channel-ablation study
python beam_generalization_study.py
```

## Limitations

- **N = 65** is small for training three separate ML models, especially the MLP
- **No independent external validation cohort** — everything reported is internal to this sample
- **No artifact rejection** — noisy or artifact-heavy epochs are averaged in with equal weight
- **Frontal channels (Fp1/Fp2)** are the most blink-prone electrodes of any standard montage, and there's no EOG/ICA-based correction step
- **Hyperparameters are reasonable defaults**, not tuned via grid search
- **No confidence intervals** currently reported on CV metrics
- **`predict_proba` outputs are not calibration-checked** — "risk probability" reflects model consensus, not a validated real-world frequency
- **Cognitive-trajectory decline rates are sourced from published literature, not fit to this study's subjects** — DS1/DS2 are single-timepoint recordings with no longitudinal MMSE data to fit against

## Future Work

1. **Cross-protocol generalization** — train on DS1, test purely on DS2 (and vice versa) to directly test whether the AD signal transfers across recording protocols (script scaffolded in `beam_generalization_study.py`)
2. **Wearable-channel ablation** — quantify the accuracy cost of going from the full 19-channel montage down to the 4-channel wearable subset
3. **Pretrained EEG encoders** — compare hand-crafted band-power features against learned representations from self-supervised EEG foundation models
4. **Longitudinal prognosis validation** — validate risk-tier trajectories against real follow-up MMSE data, rather than literature-sourced approximations

## Use of LLMs in This Project

Large language model assistance (Anthropic's Claude) was used throughout this project's development, including:
- Debugging and refactoring pipeline code (e.g., identifying and fixing subject-level data leakage in the Combined dataset's cross-validation)
- Drafting and iterating on the dashboard's front-end code
- Researching and sourcing citations for the cognitive-decline rates used in the risk-trajectory module
- Drafting documentation, including this README and accompanying presentation materials

All experimental design decisions, data interpretation, and methodological choices were made and reviewed by the project author(s). LLM output — particularly generated code and cited figures — was checked against primary sources and, where applicable, validated against the actual dataset outputs rather than accepted at face value.

## References

1. Miltiadous, A., Tzimourta, K. D., Afrantou, T., Ioannidis, P., Grigoriadis, N., Tsalikakis, D. G., Angelidis, P., Tsipouras, M. G., Glavas, E., Giannakeas, N., & Tzallas, A. T. (2023). A Dataset of Scalp EEG Recordings of Alzheimer's Disease, Frontotemporal Dementia and Healthy Subjects from Routine EEG. *Data*, 8(6), 95. `[ds004504]`
2. Miltiadous, A., et al. (2025). A Complementary Dataset of Scalp EEG Recordings Featuring Participants with Alzheimer's Disease, Frontotemporal Dementia, and Healthy Controls, Obtained from Photostimulation EEG. *Data*, 10(5), 64. `[ds006036]`
3. Petersen, R. C. (2001). Mild Cognitive Impairment in the Elderly. *American Family Physician*, 63(4), 620.
4. Wang, S., et al. (2021). Self-Administered Gerocognitive Examination: longitudinal cohort testing for early detection of dementia conversion. PMC8650250.
5. Cortes-Bermea, C., et al. (2022). Rates of Cognitive Decline in 100 Patients With Alzheimer Disease. PMC9196957.
6. Kai, K., et al. (2020). EEG alpha reactivity and cholinergic system integrity in Lewy body dementia and Alzheimer's disease. *Alzheimer's Research & Therapy*. PMC7178985.
7. Welch, P. D. (1967). The use of fast Fourier transform for the estimation of power spectra: A method based on time averaging over short, modified periodograms. *IEEE Transactions on Audio and Electroacoustics*, 15(2), 70–73.
8. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
9. Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297.
10. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

---

*BEAM is a student research project. Not intended for clinical or diagnostic use.*
