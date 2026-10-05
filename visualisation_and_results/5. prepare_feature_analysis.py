# Extract Selected Features per LORO Fold


# Imports


import os
from pathlib import Path
import hashlib

# single-thread BLAS
for env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[env_var] = "1"

import numpy as np
import pandas as pd
import h5py

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif


# Configuration (same as the main model notebook)


# Paths
SEIZURE_DIR     = Path("Features/Seizure")
NON_SEIZURE_DIR = Path("Features/Non_Seizure")
FEATURE_KEY     = "all_feats"      # dataset name inside the .mat files
LABEL_KEY       = "all_labels"
N_FEATURES      = 1530             # used to detect transposed arrays
IMAG_TOL        = 1e-6             # windows with a larger imaginary part are dropped

# Training-window budget
MAX_TRAIN_WINDOWS = 10000          # total training windows (seizure + interictal)
STRAT_BINS        = 50             # temporal segments for stratified sampling
SUBSAMPLE_SEED    = 0              # seed for interictal sampling


# FCQ-mRMR feature selection (same as the main model notebook)


def mrmr_rank(X, y, k, floor=1e-3):
    """Feature indices ordered by greedy FCQ-mRMR (at most k)."""
    X = np.asarray(X, float); y = np.asarray(y)
    n, p = X.shape
    with np.errstate(all="ignore"):
        F, _ = f_classif(X, y)
    F = np.nan_to_num(np.asarray(F, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    F = np.maximum(F, 0.0)
    pool = F > 0.0                                      # keep only relevance>0
    k = int(min(k, int(pool.sum())))
    if k <= 0:
        return np.array([], dtype=int)
    sd = X.std(0); sd = np.where(sd < 1e-12, 1.0, sd)
    Z = (X - X.mean(0)) / sd
    remaining = pool.copy()
    first = int(np.argmax(np.where(pool, F, -np.inf)))  # most relevant first
    selected = [first]; remaining[first] = False
    accum = np.zeros(p)                                 # sum of |corr| to selected
    for _ in range(1, k):
        corr = np.abs((Z.T @ Z[:, selected[-1]]) / n)   # |corr| vs last pick
        corr = np.clip(np.nan_to_num(corr, nan=floor), floor, None)
        accum += corr
        den = accum / len(selected)                     # mean |corr| to selected
        den = np.where(np.isclose(den, 1.0), np.inf, den)
        score = np.where(remaining, F / den, -np.inf)
        nxt = int(np.argmax(score))
        if not np.isfinite(score[nxt]):
            break
        selected.append(nxt); remaining[nxt] = False
    return np.array(selected, dtype=int)


# Load one patient (same as the main model notebook)


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


# Build the training pool (same as the main model notebook)


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
    offsets = {}
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

    if non_records:
        for name, (X, y) in non_records.items():
            if name == exclude_non:
                continue
            interictal_pool.append((name, X, y, np.arange(len(y))))
            offsets[name] = 0.0

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


# Extraction config


import numpy as np
import pandas as pd

# Configuration

N_FEATURES = 1530

FEATURE_NAMES_PATH = "gc_feature_lookup.csv"
SEGMENT_CSV = "SEGMENTS_MAIN_MODEL.csv"

PATIENTS_TO_RUN = ['chb03','chb04','chb05','chb06','chb07','chb08','chb09','chb10']          # or None to run all patients

PATIENT_COL = "patient"
RECORD_COL = "record"
KFEAT_COL = "k_feat"

# Load feature names

def load_feature_names(path, n_expected=N_FEATURES):
    try:
        if str(path).lower().endswith(".npy"):
            names = list(np.load(path, allow_pickle=True).ravel())
        else:
            df = pd.read_csv(path)
            names = df["name"].astype(str).tolist()

    except FileNotFoundError:
        print(f"[warn] '{path}' not found -> using generic names.")
        names = [f"f{i:04d}" for i in range(n_expected)]

    if len(names) != n_expected:
        print(f"[warn] got {len(names)} names but expected {n_expected}.")

    return names


FEATURE_NAMES = load_feature_names(FEATURE_NAMES_PATH)


def name_of(i):
    return FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"f{i:04d}"


# Load combined segment results

segments = pd.read_csv(SEGMENT_CSV)
SEGMENTS = pd.read_csv("SEGMENTS_MAIN_MODEL.csv")
# Restrict to selected patients if requested
if PATIENTS_TO_RUN is not None:
    segments = segments[segments[PATIENT_COL].isin(PATIENTS_TO_RUN)].copy()

print(f"Loaded {len(segments)} outer folds.")
print(f"Patients: {sorted(segments[PATIENT_COL].unique())}")

# Iterate over patients

for patient, patient_df in segments.groupby(PATIENT_COL):

    print(f"\n===== {patient} =====")

    for _, row in patient_df.iterrows():

        record = row[RECORD_COL]
        k_feat = int(row[KFEAT_COL])

        print(f"{record}: k = {k_feat}")


# Repeat the selection for each fold and write the CSV files


PATIENTS_TO_RUN = ['chb03','chb04','chb05','chb06','chb07','chb08','chb09','chb10']


def selected_features_for_fold(record, k_feat, seizure_records, non_records):

    if record in seizure_records:
        Xtr, ytr, _, _ = build_train(seizure_records, non_records, exclude=record)

    elif record in non_records:
        Xtr, ytr, _, _ = build_train(seizure_records, non_records, exclude_non=record)

    else:
        raise KeyError(f"record '{record}' not found")

    scaler = StandardScaler().fit(Xtr)
    idx = mrmr_rank(scaler.transform(Xtr), np.asarray(ytr), int(k_feat))
    return np.asarray(idx, dtype=int)


def extract_patient(patient):
    seg = SEGMENTS[SEGMENTS["patient"] == patient].copy()

    # load the feature arrays
    seizure_records, non_records = load_patient(patient)

    print("SEGMENT records:")
    print(seg[RECORD_COL].tolist())

    print("\nSeizure records:")
    print(sorted(seizure_records))

    print("\nNon-seizure records:")
    print(sorted(non_records))

    # every fold record must be loaded
    known = set(seizure_records) | set(non_records)
    missing = [r for r in seg[RECORD_COL] if r not in known]
    if missing:
        print(f"\n[warn] {len(missing)} record(s) in the csv were not loaded "
              f"by load_patient (corrupt/skipped or path mismatch): {missing}")

    long_rows, wide_rows = [], []     # accumulators
    for _, r in seg.iterrows():
        rec    = r[RECORD_COL]
        k_feat = int(r[KFEAT_COL])
        idx    = selected_features_for_fold(rec, k_feat, seizure_records, non_records)
        names  = [name_of(i) for i in idx]
        for rank, (i, nm) in enumerate(zip(idx.tolist(), names), start=1):
            long_rows.append({"patient": patient, "record": rec, "k_feat": k_feat,
                              "rank": rank, "feature_index": i, "feature_name": nm})
        wide_rows.append({"patient": patient, "record": rec, "k_feat": k_feat,
                          "n_selected": len(idx),
                          "feature_indices": ";".join(map(str, idx.tolist())),
                          "feature_names": ";".join(names)})
        print(f"  [{patient}] {rec}: selected {len(idx)}/{k_feat} features")

    long_df = pd.DataFrame(long_rows)
    wide_df = pd.DataFrame(wide_rows)
    n_folds = seg[RECORD_COL].nunique()
    freq_df = (long_df.groupby(["feature_index", "feature_name"])
               .size().reset_index(name="times_selected")
               .sort_values("times_selected", ascending=False))
    freq_df["fold_fraction"] = freq_df["times_selected"] / max(n_folds, 1)

    long_df.to_csv(f"SELECTED_FEATURES_LONG_{patient}.csv", index=False)
    wide_df.to_csv(f"SELECTED_FEATURES_WIDE_{patient}.csv", index=False)
    freq_df.to_csv(f"SELECTED_FEATURES_FREQUENCY_{patient}.csv", index=False)
    print(f"  [{patient}] wrote LONG / WIDE / FREQUENCY csvs "
          f"({len(long_df)} fold-feature rows, {n_folds} folds)")
    return long_df, wide_df, freq_df


import traceback
all_long = []
for p in PATIENTS_TO_RUN:
    try:
        ldf, wdf, fdf = extract_patient(p)
        all_long.append(ldf)
    except Exception as e:
        print(f"[{p}] FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()     # print the traceback

if all_long:
    pooled = pd.concat(all_long, ignore_index=True)
    pooled_freq = (pooled.groupby(["feature_index", "feature_name"])
                   .size().reset_index(name="times_selected")
                   .sort_values("times_selected", ascending=False))
    pooled_freq.to_csv("SELECTED_FEATURES_FREQUENCY_ALL.csv", index=False)
    print(f"\nWrote SELECTED_FEATURES_FREQUENCY_ALL.csv "
          f"({pooled['patient'].nunique()} patients)")

all_long[0].head(20) if all_long else None


# Ictal vs. Non-Ictal Summary


"""Per-patient, per-feature ictal vs non-ictal summary (AUC and Cohen's d)."""
import numpy as np, pandas as pd, h5py
from pathlib import Path
from scipy.stats import rankdata
 
N_FEATURES  = 1530
IMAG_TOL    = 1e-6
SEIZURE_DIR = Path("Features/Seizure")
NONSEIZ_DIR = Path("Features/Non_Seizure")
SEIZ_GLOB   = "{p}*_seizure.mat"     # seizure files
NONSEIZ_GLOB= "{p}*.mat"             # seizure-free files (in Non_Seizure/)
PATIENTS    = [f"chb{p:02d}" for p in range(1, 25)]
 
# loader
def _read_feats(h, key="all_feats"):
    X = h[key][:]
    if X.dtype.names and "imag" in X.dtype.names:
        real, imag = X["real"], X["imag"]
    else:
        real, imag = X, np.zeros_like(X)
    if real.shape[0] == N_FEATURES:
        real, imag = real.T, imag.T
    bad = np.abs(imag).max(axis=1) > IMAG_TOL
    return np.ascontiguousarray(real, dtype=np.float64), bad
 
def load_seizure(patient):
    """Seizure records only. Returns list of (X, y) where y has 1=ictal, 0=background."""
    recs = []
    for f in sorted(SEIZURE_DIR.glob(SEIZ_GLOB.format(p=patient))):
        with h5py.File(f, "r") as h:
            X, bad = _read_feats(h, "all_feats")
            yv = h["all_labels"][:].squeeze()
        y = (yv if yv.shape[0] == X.shape[0] else yv[: X.shape[0]]).astype(int)
        if bad.any():
            X, y = X[~bad], y[~bad]
        if y.sum() > 0:                       # keep only records with a seizure
            recs.append((X, y))
    return recs
 
def load_nonseizure(patient):
    """Seizure-free records. Every window is non-ictal (label 0). Returns list of X."""
    if not NONSEIZ_DIR.exists():
        return []
    Xs = []
    for f in sorted(NONSEIZ_DIR.glob(NONSEIZ_GLOB.format(p=patient))):
        with h5py.File(f, "r") as h:
            X, bad = _read_feats(h, "all_feats")
        if bad.any():
            X = X[~bad]
        Xs.append(X)
    return Xs
 
# per-feature stats
def feature_stats(X, y):
    """Rank-AUC + Cohen's d for every feature. y: 1=ictal, 0=non-ictal."""
    ict, inter = X[y == 1], X[y == 0]
    n1, n0 = len(ict), len(inter)
    if n1 == 0 or n0 == 0:
        return None
    R = np.apply_along_axis(rankdata, 0, X)                       # ranks within each column
    auc = (R[y == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)      # P(ictal > non-ictal)
    m1, m0 = ict.mean(0), inter.mean(0)
    sp = np.sqrt(((n1 - 1) * ict.var(0, ddof=1) + (n0 - 1) * inter.var(0, ddof=1)) / (n1 + n0 - 2))
    d  = np.divide(m1 - m0, sp, out=np.zeros_like(m1), where=sp > 0)
    return dict(n_ictal=n1, n_interictal=n0, mean_ictal=m1, mean_interictal=m0,
                auc=auc, abs_auc=np.abs(auc - 0.5) + 0.5, cohen_d=d)
 
def to_frame(patient, stats):
    return pd.DataFrame({
        "patient": patient, "feature_index": np.arange(N_FEATURES),
        "n_ictal": stats["n_ictal"], "n_interictal": stats["n_interictal"],
        "mean_ictal": stats["mean_ictal"], "mean_interictal": stats["mean_interictal"],
        "auc": stats["auc"], "abs_auc": stats["abs_auc"], "cohen_d": stats["cohen_d"],
    })
 
# per patient, both definitions
def summarize_patient(patient):
    seiz = load_seizure(patient)
    if not seiz:
        return None, None
    Xs = np.vstack([x for x, _ in seiz])
    ys = np.hstack([yy for _, yy in seiz]).astype(int)
 
    # (A) within-record: non-ictal = background inside seizure files
    within = feature_stats(Xs, ys)
 
    # (B) all-records: add every window from seizure-free records as non-ictal (0)
    nons = load_nonseizure(patient)
    if nons:
        Xn = np.vstack(nons)
        X_all = np.vstack([Xs, Xn])
        y_all = np.concatenate([ys, np.zeros(len(Xn), dtype=int)])
    else:
        X_all, y_all = Xs, ys        # no seizure-free files found: use the within-record definition
    allrec = feature_stats(X_all, y_all)
 
    wf = to_frame(patient, within)  if within else None
    af = to_frame(patient, allrec)  if allrec else None
    return wf, af
 
if __name__ == "__main__":
    within_out, allrec_out = [], []
    for p in PATIENTS:
        wf, af = summarize_patient(p)
        if wf is None:
            continue
        within_out.append(wf); allrec_out.append(af)
        n_within = int(wf.n_interictal.iloc[0])
        n_all    = int(af.n_interictal.iloc[0])
        print(f"[{p}] ictal={int(wf.n_ictal.iloc[0]):5d}  "
              f"non-ictal within={n_within:6d}  all-records={n_all:7d}  "
              f"(+{n_all - n_within} seizure-free windows)")
 
    if not within_out:
        raise SystemExit("No patients summarized -- check SEIZURE_DIR / glob.")
 
    pd.concat(within_out, ignore_index=True).to_csv("ICTAL_VS_INTERICTAL_SUMMARY.csv", index=False)
    pd.concat(allrec_out, ignore_index=True).to_csv("ICTAL_VS_NONICTAL_ALLREC_SUMMARY.csv", index=False)
    print("\nwrote ICTAL_VS_INTERICTAL_SUMMARY.csv        (within-record non-ictal)")
    print("wrote ICTAL_VS_NONICTAL_ALLREC_SUMMARY.csv   (all-records non-ictal)")
