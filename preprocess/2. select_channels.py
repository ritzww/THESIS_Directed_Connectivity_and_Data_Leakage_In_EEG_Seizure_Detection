# ---------------------------------------
# Channel Selection
# ---------------------------------------


import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
import os
import re 
from pathlib import Path


sys.path.insert(0, os.path.abspath("."))


# Normalise Channel Name and Create New Dataset


# Normalise the channel names and drop duplicated channels, keeping the first occurrence.


def normalize(ch):
    if isinstance(ch, (bytes, bytearray)):
        ch = ch.decode()

    ch = ch.upper().strip().replace(" ", "")
    ch = re.sub(r"-\d+$", "", ch)

    return ch


def drop_duplicates(signals, ch_names):

    seen = {}
    keep_idx = []

    for i, ch in enumerate(ch_names):

        ch_norm = normalize(ch)

        # Keep the first occurrence only
        if ch_norm not in seen:
            seen[ch_norm] = i
            keep_idx.append(i)

    return signals[keep_idx, :], [normalize(ch_names[i]) for i in keep_idx]


def process_and_save(path, target_channels, clean_dir):

    with h5py.File(path, "r") as f:
        signals = f["signals"][:]
        fs = f["sampling_rate"][()]
        ch_names = f["channel_names"][:]
        intervals = f["seizure_intervals"][:] if "seizure_intervals" in f else np.array([])

    # Decode byte strings
    ch_names = [
        c.decode() if isinstance(c, (bytes, bytearray)) else c
        for c in ch_names
    ]

    # Normalise channel names
    ch_names = [normalize(c) for c in ch_names]

    # Drop duplicated channels
    signals, ch_names = drop_duplicates(signals, ch_names)

    # Keep the target channels in a fixed order
    index = {ch: i for i, ch in enumerate(ch_names)}

    missing = [c for c in target_channels if c not in index]
    if missing:
        print(f"Skipping {path} (missing: {missing})")
        return

    idx = [index[c] for c in target_channels]

    signals = signals[idx, :]
    ch_names = target_channels

    # Save to the output folder
    filename = os.path.basename(path)
    out_path = os.path.join(clean_dir, filename)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("signals", data=signals, compression="gzip")
        f.create_dataset("sampling_rate", data=fs)
        f.create_dataset("channel_names", data=np.array(ch_names, dtype="S"))
        f.create_dataset("seizure_intervals", data=intervals)

    print(f"Saved → {out_path}")


RAW_DIR = "../Dataset_hdf5_filtered"
CLEAN_DIR = "../Dataset_hdf5_18_channels"
os.makedirs(CLEAN_DIR, exist_ok=True)

# ---------------------------------------
# 18-channel bipolar montage
# ---------------------------------------

target_channels = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ']


for root, _, files in os.walk(RAW_DIR):
    for file in files:
        if file.endswith(".h5"):

            path = os.path.join(root, file)

            process_and_save(
                path,
                target_channels,
                CLEAN_DIR
            )

# ---------------------------------------
# Check the number of files
# ---------------------------------------


root = "../Dataset_hdf5_filtered"
count = 0

for _, _, files in os.walk(root):
    for file in files:
        if file.endswith(".h5"):
            count += 1

print("Total HDF5 files:", count)


root = "../Dataset_hdf5_18_channels"
count = 0

for _, _, files in os.walk(root):
    for file in files:
        if file.endswith(".h5"):
            count += 1

print("Total HDF5 files:", count)

# ---------------------------------------
# Compare with the record list
# ---------------------------------------



# File listing the expected EDF paths
expected_txt = "RECORDS.txt"

# Root folder of the HDF5 dataset
root = Path("../Dataset_hdf5")

# Read expected EDF paths
with open(expected_txt, "r") as f:
    expected = {
        line.strip().replace(".edf", ".h5")
        for line in f
        if line.strip()
    }

# Find all HDF5 files recursively
actual = {
    str(p.relative_to(root)).replace("\\", "/")
    for p in root.rglob("*.h5")
}

# Compare
missing = sorted(expected - actual)
extra = sorted(actual - expected)

print("=" * 60)
print(f"Expected : {len(expected)}")
print(f"Found    : {len(actual)}")
print(f"Missing  : {len(missing)}")
print(f"Extra    : {len(extra)}")
print("=" * 60)

if missing:
    print("\nMISSING FILES:")
    for f in missing:
        print(f)

if extra:
    print("\nEXTRA FILES:")
    for f in extra:
        print(f)



# ---------------------------------------
# Check for missing channels
# ---------------------------------------


def check_channels(path, target_channels ):

    with h5py.File(path, "r") as f:
        signals = f["signals"][:]
        ch_names = f["channel_names"][:]

    # Decode byte strings
    ch_names = [
        c.decode() if isinstance(c, (bytes, bytearray)) else c
        for c in ch_names
    ]

    # Normalise channel names
    ch_names = [normalize(c) for c in ch_names]

    # Drop duplicated channels
    signals, ch_names = drop_duplicates(signals, ch_names)

    # Keep the target channels in a fixed order
    index = {ch: i for i, ch in enumerate(ch_names)}

    missing = [c for c in target_channels if c not in index]
    if missing:
        print(f"Skipping {path} (missing: {missing})")
        return
    else:
        print(f"Done: {path}")


test_20_channels = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ' , 'T7-FT9', 'FT10-T8'

]


for root, _, files in os.walk(RAW_DIR):

    for file in files:
        if file.endswith(".h5"):

            path = os.path.join(root, file)

            check_channels(
                path,
                test_20_channels
            )
