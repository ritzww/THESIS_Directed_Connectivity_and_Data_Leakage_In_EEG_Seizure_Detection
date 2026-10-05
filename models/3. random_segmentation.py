# %% [markdown]
# -----------------------------

# # Cross-Validated Segment Model with Random Splitting
# -----------------------------

# 
# Fixed configuration (`C=1, gamma=0.001, k_feat=100`), stratified 5-fold CV on randomly assigned windows.

# %%
from pathlib import Path
import numpy as np
import pandas as pd
import h5py

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

# %% [markdown]
# ## Load one patient (seizure files only)

# %%
N_FEATURES = 1530
IMAG_TOL   = 1e-6   

SEIZURE_DIR = Path("Features/Seizure")   


def _read_feats(h, key="all_feats"):
    """Read a feature array: real part (rows = windows) and a mask of invalid windows."""
    X = h[key][:]
    if X.dtype.names and "imag" in X.dtype.names:
        real, imag = X["real"], X["imag"]
    else:
        real, imag = X, np.zeros_like(X)
    if real.shape[0] == N_FEATURES:          # rows = windows
        real, imag = real.T, imag.T
    bad = np.abs(imag).max(axis=1) > IMAG_TOL
    return np.ascontiguousarray(real, dtype=np.float64), bad


def load_patient(patient):
    """Seizure records for one patient as {record_name: (X, y)}."""
    records = {}
    for f in sorted(SEIZURE_DIR.glob(f"{patient}*_seizure.mat")):
        with h5py.File(f, "r") as h:
            X, bad = _read_feats(h, "all_feats")
            yv = h["all_labels"][:].squeeze()
        y = (yv if yv.shape[0] == X.shape[0] else yv[: X.shape[0]]).astype(int)
        if bad.any():
            X, y = X[~bad], y[~bad]
        records[f.stem.replace("_seizure", "")] = (X, y)
        print("[SEIZURE]", f.stem, X.shape, "positives =", int(y.sum()))
    return {k: v for k, v in records.items() if v[1].sum() > 0}


def pool_windows(seizure_records):
    """Stack every window from a patient's seizure files into one (X, y) pool."""
    X = np.vstack([X for X, _ in seizure_records.values()])
    y = np.hstack([y for _, y in seizure_records.values()]).astype(int)
    return X, y

# %% [markdown]
# ## Feature selection (FCQ-mRMR), class weighting and pipeline

# %%
def damped_class_weight(y, power=1):
    """Inverse-frequency class weights raised to `power`, renormalised to mean 1."""
    classes, counts = np.unique(y, return_counts=True)
    n = len(y)
    inv = n / (len(classes) * counts)        # inverse-frequency weights
    weights = inv ** power                    # damping
    weights *= n / np.sum(counts * weights)   # mean sample weight = 1
    return {int(c): float(w) for c, w in zip(classes, weights)}


# FCQ-mRMR feature selection
# relevance = ANOVA F-statistic, redundancy = mean |Pearson corr|
def mrmr_rank(X, y, k, floor=1e-3):
    """Feature indices ordered by greedy FCQ-mRMR (at most k)."""
    X = np.asarray(X, float); y = np.asarray(y)
    n, p = X.shape
    with np.errstate(all="ignore"):
        F, _ = f_classif(X, y)
    F = np.nan_to_num(np.asarray(F, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    F = np.maximum(F, 0.0)
    pool = F > 0.0                                      # relevance > 0 only
    k = int(min(k, int(pool.sum())))
    if k <= 0:
        return np.array([], dtype=int)
    # z-score so dot products are correlations
    sd = X.std(0); sd = np.where(sd < 1e-12, 1.0, sd)
    Z = (X - X.mean(0)) / sd
    remaining = pool.copy()
    first = int(np.argmax(np.where(pool, F, -np.inf)))  # most relevant first
    selected = [first]; remaining[first] = False
    accum = np.zeros(p)                                 # sum of |corr| to selected
    for _ in range(1, k):
        corr = np.abs((Z.T @ Z[:, selected[-1]]) / n)   # |corr| vs last pick
        corr = np.clip(np.nan_to_num(corr, nan=floor), floor, None)   # per-pair floor
        accum += corr
        den = accum / len(selected)                     # mean |corr| to selected
        den = np.where(np.isclose(den, 1.0), np.inf, den)   # skip perfectly redundant
        score = np.where(remaining, F / den, -np.inf)
        nxt = int(np.argmax(score))
        if not np.isfinite(score[nxt]):
            break
        selected.append(nxt); remaining[nxt] = False
    return np.array(selected, dtype=int)


class MRMRSelector(BaseEstimator, TransformerMixin):
    """Top-k FCQ-mRMR selector, fitted on the training data only."""
    def __init__(self, k=20):
        self.k = k
    def fit(self, X, y=None):
        self.support_idx_ = mrmr_rank(X, y, self.k)
        return self
    def transform(self, X):
        return np.asarray(X)[:, self.support_idx_]


def make_pipe(cfg, class_weight):
    return Pipeline([
        ("scale",  StandardScaler()),
        ("select", MRMRSelector(k=cfg["k_feat"])),
        ("svm",    SVC(kernel="rbf", C=cfg["C"], gamma=cfg["gamma"],
                       class_weight=class_weight, cache_size=1024)),
    ])

# %% [markdown]
# ## Segment metrics

# %%
def _safe_div(a, b):
    return a / b if b else float("nan")


def _segment_confusion(y, sc, thr=0.0):
    """Segment-level metrics at threshold `thr`: acc, spec, sens, prec, f1, tp, fp, fn."""
    y = np.asarray(y).astype(int)
    pred = (sc >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    acc  = _safe_div(tp + tn, tp + tn + fp + fn)
    spec = _safe_div(tn, tn + fp)
    sens = _safe_div(tp, tp + fn)
    prec = _safe_div(tp, tp + fp)
    f1 = (_safe_div(2 * prec * sens, prec + sens)
          if (prec == prec and sens == sens) else float("nan"))
    return acc, spec, sens, prec, f1, tp, fp, fn


def diag_scores(y, sc):
    """ROC-AUC and average precision; NaN if the fold is single-class."""
    y = np.asarray(y).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan"), float("nan")
    return roc_auc_score(y, sc), average_precision_score(y, sc)


def segment_row(patient, record, cfg, y, sc):
    """One row of the segment CSV."""
    acc, spec, sens, prec, f1, tp, fp, fn = _segment_confusion(y, sc, 0.0)
    roc, ap = diag_scores(y, sc)
    return {
        "C": cfg["C"], "gamma": cfg["gamma"], "k_feat": cfg["k_feat"],
        "patient": patient, "record": record,
        "n_windows": int(len(y)), "n_seizure_windows": int(np.sum(y)),
        "record_length": float("nan"), "threshold": 0.0, "k": float("nan"),
        "roc_auc": roc, "ap": ap,
        "seg_accuracy": acc, "seg_specificity": spec,
        "seg_sensitivity": sens, "seg_precision": prec, "seg_f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
    }

# %% [markdown]
# -----------------------------

# ## Configuration

# -----------------------------

# %%
SEED     = 42
PATIENTS = [f"chb{i:02d}" for i in range(1, 25)]   # chb01 .. chb24
CV_FOLDS = 5
CLASS_WEIGHT_POWER = 0.5

# Fixed hyperparameters
FIXED_CFG = {"C": 1, "gamma": 0.001, "k_feat": 100}

# Output
MODEL_A_CSV = "CV_FIXED_segment.csv"


def cv_splits(y, n_splits, seed):
    """Stratified K-fold; fewer folds if the patient has few seizure windows."""
    n_pos = int(np.sum(y))
    k = min(n_splits, n_pos, len(y) - n_pos)
    if k < 2:
        return None
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

# %% [markdown]
# ## Fixed hyperparameters, 5-fold CV

# %%
def run_model_a(patient, cfg):
    seizure_records = load_patient(patient)
    if not seizure_records:
        print(f"  [{patient}] no seizure records — skipping")
        return []
    X, y = pool_windows(seizure_records)
    skf = cv_splits(y, CV_FOLDS, SEED)
    if skf is None:
        print(f"  [{patient}] too few seizure windows for CV — skipping")
        return []
    print(f"  [{patient}] pool windows={len(y)} positives={int(y.sum())}")

    rows = []
    for i, (tr, te) in enumerate(skf.split(X, y), 1):
        cw = damped_class_weight(y[tr], CLASS_WEIGHT_POWER)
        model = make_pipe(cfg, cw).fit(X[tr], y[tr])
        sc = model.decision_function(X[te])
        row = segment_row(patient, f"fold{i}", cfg, y[te], sc)
        rows.append(row)
        print(f"    fold{i}: AUPRC={row['ap']:.3f} ROC={row['roc_auc']:.3f} "
              f"SENS={row['seg_sensitivity']:.3f} SPEC={row['seg_specificity']:.3f}")
    return rows


rows_a = []
for patient in PATIENTS:
    print(f"\n[A] {patient}")
    try:
        rows_a.extend(run_model_a(patient, FIXED_CFG))
        pd.DataFrame(rows_a).to_csv(MODEL_A_CSV, index=False)
    except Exception as e:
        print(f"  FAILED {patient}: {type(e).__name__}: {e}")

df_a = pd.DataFrame(rows_a)
df_a.to_csv(MODEL_A_CSV, index=False)
print(f"\nWrote {MODEL_A_CSV}  ({len(df_a)} rows)")
df_a


