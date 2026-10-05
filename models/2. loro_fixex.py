# //////////////////////////////////////////////////////77
# # Seizure-File LORO without tuning
# //////////////////////////////////////////////////////77

# -----------------------------
# Leave-one-record-out over the seizure-containing records only, with fixed hyperparameters and a fixed operating point (threshold 0, k = 3).
# -----------------------------

# -----------------------------
# ## Imports
# -----------------------------

# %%
import os
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
from sklearn.metrics import roc_auc_score, average_precision_score

from timescoring import scoring
from timescoring.annotations import Annotation

# -----------------------------

# ## Configuration
# -----------------------------


# Patients to run
ALL_PATIENTS = [f"chb{n:02d}" for n in range(1, 25)]
shard = os.environ.get("PATIENT_SHARD", "").strip()
if shard:
    PATIENTS = [p.strip() for p in shard.split(",") if p.strip()]
else:
    PATIENTS = ALL_PATIENTS

# Paths
SEIZURE_DIR     = Path("Features/Seizure")
ANNOTATIONS_CSV = "seizure_annotations.csv"
FEATURE_KEY     = "all_feats"      
LABEL_KEY       = "all_labels"
N_FEATURES      = 1530           

# Fixed model configuration
# scaler -> mRMR -> RBF-SVC with damped class weights
CLASS_WEIGHT_POWER = 0.5
cfg = {"C": 1, "gamma": 0.001, "k_feat": 100}

# Windowing 
WINDOW_LEN = 10.0
STEP       = 2.5
GRID_FS    = 1

# Fixed operating point (not tuned)
# threshold 0 = SVM boundary, k = 3 consecutive positive windows
FIXED_THRESHOLD = 0.0
FIXED_K         = 3
K_SWEEP         = [2, 3, 4]        # k values also reported

# SzCORE event-scoring parameters (Dan et al., 2024)
SZCORE_PARAMS = scoring.EventScoring.Parameters(
    toleranceStart=30, toleranceEnd=60, minOverlap=0,
    maxEventDuration=5 * 60, minDurationBetweenEvents=90,
)

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


def damped_class_weight(y, power=1):
    """Inverse-frequency class weights raised to `power`, renormalised to mean 1."""
    classes, counts = np.unique(y, return_counts=True)
    n = len(y)
    inv = n / (len(classes) * counts)        # inverse-frequency weights
    weights = inv ** power                    # damping
    weights *= n / np.sum(counts * weights)   # mean sample weight = 1
    return {int(c): float(w) for c, w in zip(classes, weights)}

# -----------------------------
# ## Load one patient
# -----------------------------
IMAG_TOL = 1e-6   # windows with a larger imaginary part are dropped


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
    """Load every seizure-containing record for `patient` as {record: (X, y)}."""
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
    return {k: v for k, v in records.items() if v[1].sum() > 0}

# -----------------------------
# ## Event-scoring helpers
# -----------------------------
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

# %% [markdown]
# ## LORO over seizure files

# %%
def build_train(seizure_records, exclude=None):
    """All windows of the seizure records except `exclude`."""
    Xs, ys = [], []
    for name, (X, y) in seizure_records.items():
        if name == exclude:
            continue
        Xs.append(X)
        ys.append(np.asarray(y))
    return np.vstack(Xs), np.concatenate(ys)


def diag_scores(y, sc):
    """ROC-AUC and average precision for one fold (reporting only)."""
    y = np.asarray(y).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan"), float("nan")
    return roc_auc_score(y, sc), average_precision_score(y, sc)


def process_seizure_fold(seizure_records, test_record, cfg, k, thr):
    """One LORO fold: fit on the other seizure records and score the held-out record."""
    Xtr, ytr = build_train(seizure_records, exclude=test_record)
    cw = damped_class_weight(ytr, CLASS_WEIGHT_POWER)
    model = make_pipe(cfg, cw).fit(Xtr, ytr)
    Xte, yte = seizure_records[test_record]
    sc = model.decision_function(Xte)
    roc, ap = diag_scores(yte, sc)
    return dict(record=test_record, sc=sc, y=yte, dur=duration_of(yte),
                thr=thr, k=k, roc=roc, ap=ap,
                C=cfg["C"], gamma=cfg["gamma"], k_feat=cfg["k_feat"])

# %% [markdown]
# ## Run one patient

# %%
def run_patient(patient, cfg):
    seizure_records = load_patient(patient)
    if not seizure_records:
        print(f"[{patient}] no seizure records — skipping")
        return None, None, None, None
    print(f"[{patient}] seizure files = {len(seizure_records)}")

    # LORO over seizure records at the fixed operating point
    results = Parallel(n_jobs=OUTER_N_JOBS)(
        delayed(process_seizure_fold)(
            seizure_records, r, cfg, FIXED_K, FIXED_THRESHOLD)
        for r in seizure_records.keys()
    )

    test_outputs, meta = [], {}
    for res in results:
        test_outputs.append((res["record"], res["sc"], res["y"], res["dur"],
                             res["thr"], res["k"]))
        meta[res["record"]] = dict(roc=res["roc"], ap=res["ap"],
                                   C=res["C"], gamma=res["gamma"],
                                   k_feat=res["k_feat"])
        print(f"  [SEIZ] {res['record']}  thr={res['thr']:.3f} k={res['k']}  "
              f"C={res['C']} g={res['gamma']} p={res['k_feat']}  "
              f"roc={res['roc']:.3f} ap={res['ap']:.3f}  "
              f"test_pos={int(res['y'].sum())}")

    # per-record: segment table + event totals
    TP = FP = FN = 0
    sTP = sFP = sFN = 0   # sample-based (1 Hz) counts
    all_latencies, segment_rows, event_rows, fp_rows = [], [], [], []
    for record, sc, y, dur, thr, k in test_outputs:
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
        for ev in s.hyp.events:
            a, b = round(ev[0] * s.fs), round(ev[1] * s.fs)
            if np.all(~s.tpMask[a:b]):
                fp_rows.append({
                    "patient": patient, "record": record,
                    "fp_start": ev[0], "fp_duration_s": ev[1] - ev[0],
                })
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
        segment_rows.append({
            "C": meta[record]["C"], "gamma": meta[record]["gamma"],
            "k_feat": meta[record]["k_feat"],
            "patient": patient, "record": record,
            "n_windows": int(len(y)), "n_seizure_windows": int(np.sum(y)),
            "record_length": dur, "threshold": thr, "k": k,
            "roc_auc": meta[record]["roc"], "ap": meta[record]["ap"],
            "seg_accuracy": acc, "seg_specificity": spec,
            "seg_sensitivity": sens, "seg_precision": prec, "seg_f1": f1,
            "tp": stp, "fp": sfp, "fn": sfn,
        })
        print(f"    {record} done  (SEIZ)", flush=True)

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
    for m in ksweep:
        print(f"    k={m['k']}: sens={m['sensitivity']:.3f} "
              f"prec={m['precision']:.3f} f1={m['f1']:.3f} "
              f"FP/day={m['fp_per_day']:.3f}  (TP={m['TP']} FP={m['FP']} FN={m['FN']})")

    summary_row = {
        "C": cfg["C"], "gamma": cfg["gamma"], "k_feat": cfg["k_feat"],
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
    for m in ksweep:
        kk = m["k"]
        summary_row[f"sens_k{kk}"] = m["sensitivity"]
        summary_row[f"f1_k{kk}"]   = m["f1"]
        summary_row[f"fpday_k{kk}"] = m["fp_per_day"]

    return summary_row, segment_rows, event_rows, fp_rows


# -----------------------------
# # Run
# -----------------------------


PATIENT = "chb01"          # patient to run

print(f"Fixed operating point (thr=0, k=3), no tuning  |  patient = {PATIENT}")

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


