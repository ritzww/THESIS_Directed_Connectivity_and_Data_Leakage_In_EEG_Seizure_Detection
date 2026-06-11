"""
Event-level scoring for seizure detection.

Converts per-window (segment-level) predictions into clinically meaningful
event-level metrics:
    - sensitivity   : fraction of annotated seizures detected
    - fd_per_hour   : false detections per hour of test EEG
    - latency       : detection delay relative to annotated onset

Pipeline per record:
    binarize -> persistence rule (k consecutive) -> form detection events
    -> refractory suppression -> match events to annotated seizures.

Design choices you must FIX ON VALIDATION (not on the records you report):
    threshold, k_consecutive, refractory_sec.
Define "detected" as: a detection event overlaps the annotated [onset, offset].
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class RecordPrediction:
    """Everything needed to score one test record."""
    record_id: str
    window_starts: np.ndarray      # shape (N,), start time of each window in SECONDS, sorted
    window_len: float              # window length in seconds (e.g. 10.0)
    scores: np.ndarray             # shape (N,), model score/probability per window
    duration_sec: float            # total length of the record in seconds
    seizures: list                 # list of (onset_sec, offset_sec) tuples; [] if seizure-free


def _runs_of_true(mask):
    """Return list of (start_idx, end_idx_inclusive) for each contiguous True run."""
    runs, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def detect_events(rec, threshold, k_consecutive, refractory_sec):
    """
    Turn one record's window scores into a list of detection events.
    Each event = dict(onset_sec, offset_sec, start_idx, end_idx).
    Refractory: after an accepted detection at time t, ignore detections whose
    onset falls within [t, t + refractory_sec].
    """
    pred = rec.scores >= threshold

    # persistence: only keep runs of >= k consecutive positive windows
    events = []
    for s, e in _runs_of_true(pred):
        if (e - s + 1) >= k_consecutive:
            onset = rec.window_starts[s]
            offset = rec.window_starts[e] + rec.window_len
            events.append({"onset_sec": onset, "offset_sec": offset,
                           "start_idx": s, "end_idx": e})

    # refractory suppression (events are already time-ordered)
    accepted, last_t = [], -np.inf
    for ev in events:
        if ev["onset_sec"] >= last_t + refractory_sec:
            accepted.append(ev)
            last_t = ev["onset_sec"]
    return accepted


def score_records(records, threshold, k_consecutive=2, refractory_sec=120.0,
                  onset_tolerance_sec=0.0):
    """
    Aggregate event-level metrics across ALL test records (must include
    seizure-free records for fd_per_hour to be meaningful).

    onset_tolerance_sec : widen the match window to [onset - tol, offset + tol].

    Returns a dict of pooled metrics plus per-seizure latencies.
    """
    total_seizures = 0
    detected_seizures = 0
    false_detections = 0
    total_hours = 0.0
    latencies = []

    for rec in records:
        total_hours += rec.duration_sec / 3600.0
        events = detect_events(rec, threshold, k_consecutive, refractory_sec)

        matched_event = [False] * len(events)

        # each annotated seizure: detected if any event overlaps it
        for (onset, offset) in rec.seizures:
            total_seizures += 1
            lo, hi = onset - onset_tolerance_sec, offset + onset_tolerance_sec
            hit_latency = None
            for idx, ev in enumerate(events):
                overlaps = ev["onset_sec"] <= hi and ev["offset_sec"] >= lo
                if overlaps:
                    matched_event[idx] = True
                    lat = ev["onset_sec"] - onset   # negative => early detection
                    hit_latency = lat if hit_latency is None else min(hit_latency, lat)
            if hit_latency is not None:
                detected_seizures += 1
                latencies.append(hit_latency)

        # any accepted event matching no seizure is a false detection
        false_detections += sum(1 for m in matched_event if not m)

    sensitivity = detected_seizures / total_seizures if total_seizures else float("nan")
    fd_per_hour = false_detections / total_hours if total_hours else float("nan")

    return {
        "sensitivity": sensitivity,
        "fd_per_hour": fd_per_hour,
        "n_seizures": total_seizures,
        "n_detected": detected_seizures,
        "n_false_detections": false_detections,
        "test_hours": total_hours,
        "median_latency_sec": float(np.median(latencies)) if latencies else float("nan"),
        "latencies_sec": latencies,
    }


if __name__ == "__main__":
    # --- minimal worked example ---------------------------------------------
    # one seizure record + one seizure-free record, 10 s windows, no overlap.
    n1 = 360  # 1 hour at 10 s windows
    starts1 = np.arange(n1) * 10.0
    scores1 = np.full(n1, -3.0)
    scores1[100:106] = 2.0          # a clean ictal run (windows 100-105 -> 1000-1060 s)
    scores1[200] = 2.0              # one isolated false positive (filtered by k=2)
    rec1 = RecordPrediction("chb01_xx", starts1, 10.0, scores1, 3600.0,
                            seizures=[(1005.0, 1055.0)])

    n2 = 360
    starts2 = np.arange(n2) * 10.0
    scores2 = np.full(n2, -3.0)
    scores2[50:52] = 2.0            # a 2-window false alarm
    rec2 = RecordPrediction("chb01_yy", starts2, 10.0, scores2, 3600.0, seizures=[])

    out = score_records([rec1, rec2], threshold=0.0,
                        k_consecutive=2, refractory_sec=120.0)
    for k, v in out.items():
        if k != "latencies_sec":
            print(f"{k}: {v}")
