# ////////////////////////////////////////////////////////////7
# Patient-Specific Seizure Detection - Main Model
# ////////////////////////////////////////////////////////////7

# Nested leave-one-record-out evaluation, one model per patient.

# ////////////////////////////////////////////////////////////7
# ## Imports

# %%
import os
import hashlib
from pathlib import Path

for env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[env_var] = "1"

import numpy as np
import pandas as pd
import h5py
from joblib import Parallel, delayed

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from timescoring import scoring
from timescoring.annotations import Annotation

# ////////////////////////////////////////////////////////////7
# ## Configuration

# %%
CLASS_WEIGHT_POWER = 0.5

# Patients to run
ALL_PATIENTS = [f"chb{n:02d}" for n in range(1, 25)]
shard = os.environ.get("PATIENT_SHARD", "").strip()
if shard:
    PATIENTS = [p.strip() for p in shard.split(",") if p.strip()]
else:
    PATIENTS = ALL_PATIENTS

# Paths
SEIZURE_DIR     = Path("Features/Seizure")
NON_SEIZURE_DIR = Path("Features/Non_Seizure")
ANNOTATIONS_CSV = "seizure_annotations.csv"
FEATURE_KEY     = "all_feats"    
LABEL_KEY       = "all_labels"
N_FEATURES      = 1530          

# Model config
# filled in by select_hp_and_k (C, gamma, k_feat tuned per fold)
cfg = {}

# Windowing (must match feature extraction)
WINDOW_LEN = 10.0
STEP       = 2.5
GRID_FS    = 1

# Fixed operating point (not tuned)
# threshold 0 = SVM boundary, k = 3 consecutive positive windows
FIXED_THRESHOLD = 0.0
FIXED_K         = 3
K_SWEEP         = [2, 3, 4]      

# Hyperparameter grids
C_GRID     = [0.1, 1, 10, 100, 1000]
GAMMA_GRID = [0.1, 0.01, 0.001, 0.0001]
K_GRID     = [20, 40, 60, 80, 100]

# persistence values considered (fixed at 3)
K_GRID_CAL = [3]

# SzCORE event-scoring parameters (Dan et al., 2024)
SZCORE_PARAMS = scoring.EventScoring.Parameters(
    toleranceStart=30, toleranceEnd=60, minOverlap=0,
    maxEventDuration=5 * 60, minDurationBetweenEvents=90,
)

# Training-window budget
# all seizure windows kept; interictal windows fill the rest
MAX_TRAIN_WINDOWS = 10000          # total training windows (seizure + interictal)
STRAT_BINS        = 50             # temporal segments for stratified sampling
SUBSAMPLE_SEED    = 0              # seed for interictal sampling

# Parallelism over LORO folds
OUTER_N_JOBS = 4

# Outputs
SUMMARY_CSV = "RESULTS_SUMMARY.csv"   # combined summary across patients

SEIZ = pd.read_csv(ANNOTATIONS_CSV)


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
        den = np.where(np.isclose(den, 1.0), np.inf, den)  
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

# ////////////////////////////////////////////////////////////7
# ## Load one patient

# ----------------------------------------------------------------------
def damped_class_weight(y, power=1):
    """Inverse-frequency class weights raised to `power`, renormalised to mean 1."""
    classes, counts = np.unique(y, return_counts=True)
    n = len(y)
    inv = n / (len(classes) * counts)        # inverse-frequency weights
    weights = inv ** power                    # damping
    weights *= n / np.sum(counts * weights)   # mean sample weight = 1
    return {int(c): float(w) for c, w in zip(classes, weights)}

# %%
IMAG_TOL = 1e-6  


def _read_feats(h, key):
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
    records = {}

    for f in sorted(SEIZURE_DIR.glob(f"{patient}*_seizure.mat")):
        try:
            with h5py.File(f, "r") as h:
                X, bad = _read_feats(h, FEATURE_KEY)
                yv = h[LABEL_KEY][:].squeeze()
        except OSError as e:
            print(f"  [SKIP corrupt] {f.name} — {e}")
            continue
        if yv.shape[0] == X.shape[0]:
            y = yv.astype(int)
        else:
            y = yv[: X.shape[0]].astype(int)
        if bad.any():
            print(f"  [DROP complex] {f.name}: {int(bad.sum())} window(s) "
                  f"{list(np.where(bad)[0])}")
            X, y = X[~bad], y[~bad]
        records[f.stem.replace("_seizure", "")] = (X, y)

    for f in sorted(NON_SEIZURE_DIR.glob(f"{patient}*_features.mat")):
        try:
            with h5py.File(f, "r") as h:
                X, bad = _read_feats(h, FEATURE_KEY)
        except OSError as e:
            print(f"  [SKIP corrupt] {f.name} — {e}")
            continue
        if bad.any():
            print(f"  [DROP complex] {f.name}: {int(bad.sum())} window(s) "
                  f"{list(np.where(bad)[0])}")
            X = X[~bad]
        y = np.zeros(X.shape[0], dtype=int)
        records[f.stem.replace("_features", "")] = (X, y)

    seizure_records = {k: v for k, v in records.items() if v[1].sum() > 0}
    non_records     = {k: v for k, v in records.items() if v[1].sum() == 0}
    return seizure_records, non_records

# ----------------------------------------------------------------------

# ## Event-scoring helpers
# ----------------------------------------------------------------------

def duration_of(y):
    return (len(y) - 1) * STEP + WINDOW_LEN


def windows_to_mask(scores, duration, threshold, k):
    pos = (scores >= threshold).astype(int)
    alarm = np.zeros_like(pos); run = 0
    for i, p in enumerate(pos):
        run = run + 1 if p else 0
        if run >= k:
            alarm[i - k + 1: i + 1] = 1
    n = int(round(duration * GRID_FS))
    mask = np.zeros(n, dtype=bool)
    for i, a in enumerate(alarm):
        if a:
            s = int(round(i * STEP * GRID_FS))
            e = int(round((i * STEP + WINDOW_LEN) * GRID_FS))
            mask[s:min(e, n)] = True
    return mask, n


def ref_mask_from_record(record, n, offset_sec=0.0):
    # offset_sec is 0 for full records
    mask = np.zeros(n, dtype=bool)
    for _, r in SEIZ[SEIZ["file"] == record].iterrows():
        s = int(round((r["start_sec"] - offset_sec) * GRID_FS))
        e = int(round((r["end_sec"]   - offset_sec) * GRID_FS))
        if e <= 0 or s >= n:
            continue
        mask[max(s, 0):min(e, n)] = True
    return mask


def _arming_times(sc, thr, k):
    pos = (sc >= thr).astype(int)
    arm, run = [], 0
    for i, p in enumerate(pos):
        run = run + 1 if p else 0
        if run == k:
            arm.append(i * STEP + WINDOW_LEN)
    return arm


def detection_latencies(record, sc, thr, k):
    TOL_PRE, TOL_POST = 30, 60
    arm_times = _arming_times(sc, thr, k)
    lat = []
    for _, r in SEIZ[SEIZ["file"] == record].iterrows():
        onset, offset = r["start_sec"], r["end_sec"]
        cands = [t for t in arm_times
                 if (onset - TOL_PRE) <= t <= (offset + TOL_POST)]
        if cands:
            lat.append(min(cands) - onset)
    return lat


def _safe_div(a, b):
    return a / b if b else float("nan")


def segment_metrics(y, sc, thr):
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


def event_detections(record, sc, thr, k):
    """One row per annotated seizure: detected or not, and latency."""
    TOL_PRE, TOL_POST = 30, 60
    arm_times = _arming_times(sc, thr, k)
    rows = []
    for _, r in SEIZ[SEIZ["file"] == record].iterrows():
        onset, offset = r["start_sec"], r["end_sec"]
        cands = [t for t in arm_times
                 if (onset - TOL_PRE) <= t <= (offset + TOL_POST)]
        rows.append({
            "event_start": onset,
            "event_end":   offset,
            "detected":    1 if cands else 0,
            "latency_s":   (min(cands) - onset) if cands else float("nan"),
        })
    return rows


def event_metrics_at_k(test_outputs, threshold, k):
    """Event-level TP, FP, FN and metrics over all test records at (threshold, k)."""
    TP = FP = FN = 0
    for record, sc, y, dur, *_ in test_outputs:
        hyp, n = windows_to_mask(sc, dur, threshold, k)
        ref = ref_mask_from_record(record, n)
        s = scoring.EventScoring(Annotation(ref, GRID_FS),
                                 Annotation(hyp, GRID_FS), SZCORE_PARAMS)
        TP += s.tp; FP += s.fp; FN += s.refTrue - s.tp
    total_days = sum(t[3] for t in test_outputs) / 86400
    sens = TP/(TP+FN) if (TP+FN) else 0.0
    prec = TP/(TP+FP) if (TP+FP) else 0.0
    f1   = 2*sens*prec/(sens+prec) if (sens+prec) else 0.0
    fpday = FP/total_days if total_days else float("nan")
    return dict(k=k, TP=TP, FP=FP, FN=FN, sensitivity=sens,
                precision=prec, f1=f1, fp_per_day=fpday)

# ----------------------------------------------------------------------

# ## LORO for one patient
# 
# Loop 1 holds out each seizure record; loop 2 holds out each seizure-free record. Hyperparameters are selected on the remaining seizure records with an inner LORO.
# ----------------------------------------------------------------------

def _stable_seed(*parts):
    """Deterministic seed that is the same in every process."""
    s = "|".join("" if p is None else str(p) for p in parts)
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(),
                          "little")


def _stratified_indices(idx_pool, n_keep, n_bins, rng):
    """Pick n_keep indices spread evenly over n_bins segments of idx_pool."""
    idx_pool = np.asarray(idx_pool)
    n = len(idx_pool)
    if n_keep >= n:
        return idx_pool.copy()
    if n_keep <= 0:
        return np.array([], dtype=int)
    bins = np.array_split(idx_pool, min(n_bins, n))
    per, rem = divmod(n_keep, len(bins))
    picks = []
    for bi, b in enumerate(bins):
        q = min(per + (1 if bi < rem else 0), len(b))
        if q > 0:
            picks.append(rng.choice(b, size=q, replace=False))
    return np.sort(np.concatenate(picks)) if picks else np.array([], dtype=int)


def build_train(seizure_records, non_records=None, exclude=None,
                exclude_non=None):
    """Training data for one fold: all seizure windows plus undersampled interictal windows."""
    Xs, ys, grps = [], [], []
    offsets = {}                       # unused
    # seizure records: keep all seizure windows
    interictal_pool = []               # list of (name, X, y, avail_idx)
    total_seiz_kept = 0
    for name, (X, y) in seizure_records.items():
        if name == exclude:
            continue
        keep = (y == 1)                # seizure windows only
        if keep.any():
            Xs.append(X[keep]); ys.append(y[keep])
            grps.extend([name] * int(keep.sum()))
            total_seiz_kept += int(keep.sum())
        rest = np.where(~keep)[0]      # interictal windows go to the pool
        if rest.size:
            interictal_pool.append((name, X, y, rest))
        offsets[name] = 0.0

    # seizure-free records: all windows go to the pool
    if non_records:
        for name, (X, y) in non_records.items():
            if name == exclude_non:
                continue
            interictal_pool.append((name, X, y, np.arange(len(y))))
            offsets[name] = 0.0

    # fill the remaining budget from the interictal pool
    total_avail = sum(len(idx) for _, _, _, idx in interictal_pool)
    budget = (None if MAX_TRAIN_WINDOWS is None
              else max(0, MAX_TRAIN_WINDOWS - total_seiz_kept))
    for name, X, y, idx in interictal_pool:
        if budget is None or total_avail <= budget:
            keep_idx = idx
        elif budget == 0 or total_avail == 0:
            continue
        else:
            quota = max(1, int(round(budget * len(idx) / total_avail)))
            rng = np.random.default_rng(
                _stable_seed(SUBSAMPLE_SEED, exclude, exclude_non, name))
            keep_idx = _stratified_indices(idx, quota, STRAT_BINS, rng)
        if len(keep_idx):
            Xs.append(X[keep_idx]); ys.append(y[keep_idx])
            grps.extend([name] * len(keep_idx))

    return np.vstack(Xs), np.concatenate(ys), np.array(grps), offsets


def diag_scores(y, sc):
    y = np.asarray(y).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan"), float("nan")
    return roc_auc_score(y, sc), average_precision_score(y, sc)


# Stage 1: select (C, gamma, k_feat) on the seizure records (inner LORO)
# Stage 2: refit on the undersampled training pool


def build_selection_pool(seizure_records, exclude=None):
    """Every window of the seizure records (no undersampling), grouped by record."""
    Xs, ys, grps, order = [], [], [], []
    for name, (X, y) in seizure_records.items():
        if name == exclude:
            continue
        Xs.append(X); ys.append(np.asarray(y)); grps.extend([name] * len(y))
        order.append(name)
    if not Xs:
        return None
    return np.vstack(Xs), np.concatenate(ys), np.array(grps), order


def _event_counts_full(record, sc, thr, k, dur):
    """Event-level TP, FP, FN for one record at (thr, k)."""
    hyp, n = windows_to_mask(sc, dur, thr, k)
    ref = ref_mask_from_record(record, n)
    s = scoring.EventScoring(Annotation(ref, GRID_FS),
                             Annotation(hyp, GRID_FS), SZCORE_PARAMS)
    return int(s.tp), int(s.fp), int(s.refTrue - s.tp)


def select_hp_and_k(seizure_records, cfg, exclude=None):
    """Select (C, gamma, k_feat) by pooled window-level F1 with an inner leave-one-record-out CV."""
    default = {**cfg, "C": 1, "gamma": "scale", "k_feat": 20}
    pool = build_selection_pool(seizure_records, exclude=exclude)
    if pool is None or len(np.unique(pool[1])) < 2:
        return {"cfg": default, "k": FIXED_K, "thr": FIXED_THRESHOLD,
                "ap": float("nan"), "n_records": 0, "ksel": []}
    X, y, grps, rec_order = pool
    n_feat = X.shape[1]

    # per inner fold: scaler, mRMR order and class weight (reused across the grid)
    prep = []
    for tr, va in LeaveOneGroupOut().split(X, y, groups=grps):
        if len(np.unique(y[tr])) < 2:
            continue
        scl = StandardScaler().fit(X[tr])
        Xtr_s, Xva_s = scl.transform(X[tr]), scl.transform(X[va])
        order = mrmr_rank(Xtr_s, y[tr], max(K_GRID))
        cw = damped_class_weight(y[tr], CLASS_WEIGHT_POWER)
        prep.append((y[tr], va, Xtr_s, Xva_s, order, cw))
    if not prep:
        return {"cfg": default, "k": FIXED_K, "thr": FIXED_THRESHOLD,
                "ap": float("nan"), "n_records": 0, "ksel": []}

    # pick (C, gamma, k_feat) by pooled window-level F1
    best = None   # (f1, C, gamma, k_feat, oof)

    for kf in K_GRID:
        kf_ = min(kf, n_feat)
        for C in C_GRID:
            for gval in GAMMA_GRID:
                oof = np.full(len(y), np.nan)
                for ytr, va, Xtr_s, Xva_s, order, cw in prep:
                    sel = order[:kf_]
                    svm = SVC(kernel="rbf", C=C, gamma=gval,
                              class_weight=cw, cache_size=1024)
                    svm.fit(Xtr_s[:, sel], ytr)
                    oof[va] = svm.decision_function(Xva_s[:, sel])
                m = ~np.isnan(oof)
                if m.sum() == 0 or len(np.unique(y[m])) < 2:
                    continue
                f1 = f1_score(y[m], (oof[m] > FIXED_THRESHOLD).astype(int),
                              zero_division=0)
                if best is None or f1 > best[0]:
                    best = (f1, C, gval, kf_, oof)
    if best is None:
        return {"cfg": default, "k": FIXED_K, "thr": FIXED_THRESHOLD,
                "ap": float("nan"), "n_records": len(rec_order), "ksel": []}
    win_cfg = {**cfg, "C": best[1], "gamma": best[2], "k_feat": best[3]}
    oof = best[4]


    # persistence k (threshold fixed)
    m = ~np.isnan(oof)
    thr_grid = [FIXED_THRESHOLD]     # threshold fixed at 0

    recs = []
    for name in rec_order:
        idx = np.where(grps == name)[0]
        sc_r = oof[idx]
        if np.isnan(sc_r).any():
            continue
        recs.append((name, sc_r, y[idx], duration_of(y[idx])))

    cand = []
    for thr in thr_grid:
        for k in K_GRID_CAL:
            TP = FP = FN = 0; days = 0.0
            for name, sc_r, y_r, dur in recs:
                tp, fp, fn = _event_counts_full(name, sc_r, thr, k, dur)
                TP += tp; FP += fp; FN += fn; days += dur / 86400
            sens = TP / (TP + FN) if (TP + FN) else 0.0
            prc  = TP / (TP + FP) if (TP + FP) else 0.0
            f1   = 2 * sens * prc / (sens + prc) if (sens + prc) else 0.0
            fpday = FP / days if days else float("inf")
            cand.append(dict(thr=float(thr), k=k, sens=sens, f1=f1,
                             fpday=fpday, TP=TP, FP=FP, FN=FN))

    chosen = max(cand, key=lambda c: (c["f1"], c["sens"], -c["k"])) if cand \
             else {"thr": FIXED_THRESHOLD, "k": FIXED_K}
    return {"cfg": win_cfg, "k": int(chosen["k"]), "thr": float(chosen["thr"]),
            "ap": float(best[0]), "n_records": len(rec_order), "ksel": cand}


def process_seizure_fold(seizure_records, non_records, test_record, cfg, k, thr):
    """One fold over a seizure record: refit without it and score it in full."""
    Xtr, ytr, grps, offsets = build_train(seizure_records, non_records,
                                          exclude=test_record)
    cw = damped_class_weight(ytr,CLASS_WEIGHT_POWER)
    model = make_pipe(cfg, cw).fit(Xtr, ytr)
    Xte, yte = seizure_records[test_record]
    sc = model.decision_function(Xte)
    roc, ap = diag_scores(yte, sc)
    return dict(record=test_record, sc=sc, y=yte, dur=duration_of(yte),
                thr=thr, k=k, roc=roc, ap=ap,
                C=cfg["C"], gamma=cfg["gamma"], k_feat=cfg["k_feat"])


def process_nonseizure_fold(seizure_records, non_records, test_record, cfg, k, thr):
    """One fold over a seizure-free record: refit without it and score it in full."""
    Xtr, ytr, grps, offsets = build_train(seizure_records, non_records,
                                          exclude_non=test_record)
    cw = damped_class_weight(ytr,CLASS_WEIGHT_POWER)
    model = make_pipe(cfg, cw).fit(Xtr, ytr)
    Xte, yte = non_records[test_record]
    sc = model.decision_function(Xte)
    return dict(record=test_record, sc=sc, y=yte, dur=duration_of(yte),
                thr=thr, k=k, roc=float("nan"), ap=float("nan"),
                C=cfg["C"], gamma=cfg["gamma"], k_feat=cfg["k_feat"])

# ----------------------------------------------------------------------
# ## Code to run one patient
# ----------------------------------------------------------------------

def run_patient(patient, cfg):
    seizure_records, non_records = load_patient(patient)
    if not seizure_records:
        print(f"[{patient}] no seizure records — skipping")
        return None, None, None, None
    print(f"[{patient}] seizure={len(seizure_records)} "
          f"non-seizure={len(non_records)}")

    # Stage 1: hyperparameter selection on the seizure records
    # each seizure fold excludes its own test record
    _sel_keys = list(seizure_records.keys())
    _sel_vals = Parallel(n_jobs=OUTER_N_JOBS)(
        delayed(select_hp_and_k)(seizure_records, cfg, r) for r in _sel_keys
    )
    sel = dict(zip(_sel_keys, _sel_vals))
    sel_all = select_hp_and_k(seizure_records, cfg, exclude=None)
    for r in _sel_keys:
        s = sel[r]
        print(f"  [SELECT] {r}  C={s['cfg']['C']} g={s['cfg']['gamma']} "
              f"k_feat={s['cfg']['k_feat']}  k={s['k']} thr={s['thr']:.3f} "
              f"AP={s['ap']:.3f}")
    print(f"  [SELECT-ALL] C={sel_all['cfg']['C']} g={sel_all['cfg']['gamma']} "
          f"k_feat={sel_all['cfg']['k_feat']}  k={sel_all['k']} "
          f"thr={sel_all['thr']:.3f} AP={sel_all['ap']:.3f}")

    # Loop 1: seizure records
    seiz_results = Parallel(n_jobs=OUTER_N_JOBS)(
        delayed(process_seizure_fold)(
            seizure_records, non_records, r,
            sel[r]["cfg"], sel[r]["k"], sel[r]["thr"])
        for r in seizure_records.keys()
    )

    test_outputs, meta = [], {}
    for res in seiz_results:
        test_outputs.append((res["record"], res["sc"], res["y"], res["dur"],
                             res["thr"], res["k"]))
        # store the selected hyperparameters
        meta[res["record"]] = dict(type="seizure", roc=res["roc"], ap=res["ap"],
                                   C=res["C"], gamma=res["gamma"], k_feat=res["k_feat"])
        print(f"  [SEIZ] {res['record']}  thr={res['thr']:.3f} k={res['k']}  "
              f"C={res['C']} g={res['gamma']} p={res['k_feat']}  "
              f"roc={res['roc']:.3f} ap={res['ap']:.3f}  "
              f"test_pos={int(res['y'].sum())}")

    # Loop 2: seizure-free records
    if non_records:
        non_results = Parallel(n_jobs=OUTER_N_JOBS)(
            delayed(process_nonseizure_fold)(
                seizure_records, non_records, r,
                sel_all["cfg"], sel_all["k"], sel_all["thr"])
            for r in non_records.keys()
        )
        for res in non_results:
            test_outputs.append((res["record"], res["sc"], res["y"],
                                 res["dur"], res["thr"], res["k"]))
            # store the selected hyperparameters
            meta[res["record"]] = dict(type="nonseizure", roc=float("nan"),
                                       ap=float("nan"),
                                       C=res["C"], gamma=res["gamma"], k_feat=res["k_feat"])
        print(f"  [NON] {len(non_results)} records scored (true LONO)")

    # per-record pass: segment table + event totals
    TP = FP = FN = 0
    sTP = sFP = sFN = 0   # sample-based (1 Hz) counts
    all_latencies, segment_rows, event_rows, fp_rows = [], [], [], []
    for record, sc, y, dur, thr, k in test_outputs:
        is_seiz = meta[record]["type"] == "seizure"
        hyp, n = windows_to_mask(sc, dur, thr, k)
        ref = ref_mask_from_record(record, n)
        s = scoring.EventScoring(Annotation(ref, GRID_FS),
                                 Annotation(hyp, GRID_FS), SZCORE_PARAMS)
        TP += s.tp; FP += s.fp; FN += s.refTrue - s.tp
        # sample-based scoring on the same masks
        ss = scoring.SampleScoring(Annotation(ref, GRID_FS),
                                   Annotation(hyp, GRID_FS))
        sTP += ss.tp; sFP += ss.fp; sFN += ss.refTrue - ss.tp
        # false-positive table: one row per false alarm
        for _ev in s.hyp.events:
            _a, _b = round(_ev[0] * s.fs), round(_ev[1] * s.fs)
            if np.all(~s.tpMask[_a:_b]):
                fp_rows.append({
                    "patient": patient, "record": record,
                    "fp_start": _ev[0], "fp_duration_s": _ev[1] - _ev[0],
                })
        if is_seiz:
            all_latencies.extend(detection_latencies(record, sc, thr, k))
            # event table: one row per annotated seizure
            for ev in event_detections(record, sc, thr, k):
                event_rows.append({
                    "patient": patient, "record": record,
                    "event_start": ev["event_start"],
                    "event_end":   ev["event_end"],
                    "detected":    ev["detected"],
                    "latency_s":   ev["latency_s"],
                    "threshold": thr, "k": k,
                    "k_feat": meta[record]["k_feat"],
                })

        acc, spec, sens, prec, f1, stp, sfp, sfn = segment_metrics(y, sc, 0.0)
        if not is_seiz:
            acc = spec = sens = prec = f1 = np.nan
        segment_rows.append({
            "C": meta[record]["C"],
            "gamma": meta[record]["gamma"],
            "k_feat": meta[record]["k_feat"],
            "patient": patient, "record": record,
            "n_windows": int(len(y)), "n_seizure_windows": int(np.sum(y)),
            "record_length": dur, "threshold": thr, "k": k,
            "roc_auc": meta[record]["roc"], "ap": meta[record]["ap"],
            "seg_accuracy": acc, "seg_specificity": spec,
            "seg_sensitivity": sens, "seg_precision": prec, "seg_f1": f1,
            "tp": stp, "fp": sfp, "fn": sfn,
        })
        kind = "SEIZ" if is_seiz else "NON"
        print(f"    {record} done  ({kind})", flush=True)

    # event-level totals
    sens = TP/(TP+FN) if (TP+FN) else 0.0
    prec = TP/(TP+FP) if (TP+FP) else 0.0
    f1   = 2*sens*prec/(sens+prec) if (sens+prec) else 0.0
    # sample-based metrics
    s_sens = sTP/(sTP+sFN) if (sTP+sFN) else 0.0
    s_prec = sTP/(sTP+sFP) if (sTP+sFP) else 0.0
    s_f1   = 2*s_sens*s_prec/(s_sens+s_prec) if (s_sens+s_prec) else 0.0
    total_len_sec = sum(t[3] for t in test_outputs)
    days = total_len_sec / 86400
    med_lat = float(np.median(all_latencies)) if all_latencies else float("nan")

    fp_day = FP / days if days else float("nan")
    print(f"  EVENT: TP={TP} FP={FP} FN={FN}  sens={sens:.3f} "
          f"prec={prec:.3f} f1={f1:.3f}  FP/day={fp_day:.3f}  "
          f"lat={med_lat:.1f}s")
    print(f"  SAMPLE(1Hz): TP={sTP} FP={sFP} FN={sFN}  "
          f"sens={s_sens:.3f} prec={s_prec:.3f} f1={s_f1:.3f}")

    # results at other k (reporting only)
    ksweep = [event_metrics_at_k(test_outputs, FIXED_THRESHOLD, kk) for kk in K_SWEEP]
    print("  K-SWEEP (thr=0):")
    for _m in ksweep:
        print(f"    k={_m['k']}: sens={_m['sensitivity']:.3f} "
              f"prec={_m['precision']:.3f} f1={_m['f1']:.3f} "
              f"FP/day={_m['fp_per_day']:.3f}  (TP={_m['TP']} FP={_m['FP']} FN={_m['FN']})")

    # report the most frequent C, gamma and k_feat
    _cs = [m.get("C") for m in meta.values() if m.get("C") is not None]
    _gs = [m.get("gamma") for m in meta.values() if m.get("gamma") is not None]
    mode_C     = max(set(_cs), key=_cs.count) if _cs else None
    mode_gamma = max(set(_gs), key=_gs.count) if _gs else None
    _ps = [m.get("k_feat") for m in meta.values() if m.get("k_feat") is not None]
    mode_perc  = max(set(_ps), key=_ps.count) if _ps else None

    summary_row = {
        "C": mode_C, "gamma": mode_gamma,
        "k_feat": mode_perc,
        "patient": patient,
        "total_length_hours": total_len_sec / 3600,
        "TP": TP, "FP": FP, "FN": FN,
        "sensitivity": sens, "precision": prec, "f1": f1,
        "sample_TP": sTP, "sample_FP": sFP, "sample_FN": sFN,
        "sample_sensitivity": s_sens, "sample_precision": s_prec, "sample_f1": s_f1,
        "FP/day": FP / days if days else float("nan"),
        "FP/hour": FP / (days * 24) if days else float("nan"),
        "latency_median": med_lat, "n_detected": len(all_latencies),
        "mean_fold_roc_auc": float(np.nanmean(
            [m["roc"] for m in meta.values()])),
        "mean_fold_ap": float(np.nanmean([m["ap"] for m in meta.values()])),
    }
    for _m in ksweep:
        _k=_m["k"]
        summary_row[f"sens_k{_k}"]=_m["sensitivity"]; summary_row[f"f1_k{_k}"]=_m["f1"]
        summary_row[f"fpday_k{_k}"]=_m["fp_per_day"]

    return summary_row, segment_rows, event_rows, fp_rows

# ----------------------------------------------------------------------
# # Run for one patient
# ----------------------------------------------------------------------


PATIENT = "chb01"         

print(f"C / gamma / percentile auto-tuned per fold  |  patient = {PATIENT}")

SUMMARY_CSV = f"RESULTS_SUMMARY_{PATIENT}.csv"
SEGMENT_CSV = f"SEGMENT_PER_RECORD_{PATIENT}.csv"
EVENTS_CSV  = f"EVENTS_PER_RECORD_{PATIENT}.csv"
FP_CSV      = f"FALSE_POSITIVES_PER_RECORD_{PATIENT}.csv"

all_summaries, all_segments, all_events, all_fps = [], [], [], []

for patient in [PATIENT]:
    try:
        srow, segrows, evrows, fprows = run_patient(patient, cfg)

        if srow is not None:
            all_summaries.append(srow)
            all_segments.extend(segrows)
            all_events.extend(evrows)
            all_fps.extend(fprows)

        pd.DataFrame(all_summaries).to_csv(SUMMARY_CSV, index=False)
        pd.DataFrame(all_segments).to_csv(SEGMENT_CSV, index=False)
        pd.DataFrame(all_events).to_csv(EVENTS_CSV, index=False)
        pd.DataFrame(all_fps).to_csv(FP_CSV, index=False)

        print(f"  >>> SAVED [{patient}] -> {SUMMARY_CSV}", flush=True)

    except Exception as e:
        print(f"[{patient}] FAILED: {type(e).__name__}: {e}")
        continue

# final write
summary_df = pd.DataFrame(all_summaries)
summary_df.to_csv(SUMMARY_CSV, index=False)
pd.DataFrame(all_segments).to_csv(SEGMENT_CSV, index=False)
pd.DataFrame(all_events).to_csv(EVENTS_CSV, index=False)
pd.DataFrame(all_fps).to_csv(FP_CSV, index=False)

print(f"\nWrote {SUMMARY_CSV}")

# pooled metrics
if len(summary_df):
    TP = summary_df.TP.sum()
    FP = summary_df.FP.sum()
    FN = summary_df.FN.sum()
    total_hours = summary_df.total_length_hours.sum()

    pooled = pd.DataFrame([{
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "sensitivity": TP/(TP+FN) if (TP+FN) else float("nan"),
        "precision": TP/(TP+FP) if (TP+FP) else float("nan"),
        "FP/day": FP/(total_hours/24) if total_hours else float("nan"),
    }])

    print("\nPooled event-level:")
    print(pooled.to_string(index=False))

summary_df



