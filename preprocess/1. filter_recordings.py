#  -------------------
# Preprocessing
#  -------------------


# 1. High-pass filter and notch filter


# Filters every recording (0.5 Hz high-pass, 60 Hz notch) and saves it as HDF5.


import os
import re
import h5py
import mne
import numpy as np


# Parse the CHB-MIT summary file
def parse_chbmit_summary(summary_path):

    seizures_dict = {}

    with open(summary_path, "r", encoding="latin1") as f:
        text = f.read()

    blocks = text.split("File Name:")

    for block in blocks:

        if ".edf" not in block:
            continue

        # EDF filename extraction
        file_match = re.search(
            r"(chb\d+[a-z]?_\d+\+?\.edf)",
            block
        )

        if not file_match:
            file_match = re.search(
                r"(chb\d+[a-z]?_\d+[^\s]*\.edf)",
                block
            )

        if not file_match:
            continue

        file_name = file_match.group(1)

        seizures = []

        # Numbered seizures
        starts_num = re.findall(
            r"Seizure\s*\d+\s*Start Time:\s*(\d+)",
            block
        )

        ends_num = re.findall(
            r"Seizure\s*\d+\s*End Time:\s*(\d+)",
            block
        )

        if len(starts_num) > 0 and len(ends_num) > 0:

            seizures = [
                (int(s), int(e))
                for s, e in zip(starts_num, ends_num)
            ]

        else:

            start = re.search(
                r"Seizure Start Time:\s*(\d+)",
                block
            )

            end = re.search(
                r"Seizure End Time:\s*(\d+)",
                block
            )

            if start and end:

                seizures = [
                    (
                        int(start.group(1)),
                        int(end.group(1))
                    )
                ]

        seizures_dict[file_name] = seizures

    return seizures_dict


#  -------------------
# Filter one EDF file and save it as HDF5
#  -------------------

def edf_to_filtered_h5(
    edf_path,
    h5_path,
    seizures_sec,
    highpass=0.5,
    notch=60
):

    # Load EDF
    raw = mne.io.read_raw_edf(
        edf_path,
        preload=True,
        verbose=False
    )

    raw.rename_channels(lambda x: x.strip())

    fs = float(raw.info["sfreq"])
    ch_names = np.array(raw.ch_names, dtype="S")

    # Filtering
    raw.filter(
        l_freq=highpass,
        h_freq=None,
        method="iir",
        iir_params=dict(order=4, ftype="butter"),
        phase="zero",
        verbose=False
    )

    raw.notch_filter(
        freqs=notch,
        method="iir",
        phase="zero",
        verbose=False
    )

    # Extract filtered signal
    data = raw.get_data().astype(np.float32)

    # Convert seizure times from seconds to samples
    seizures_samples = np.array(
        [
            (int(s * fs), int(e * fs))
            for s, e in seizures_sec
        ],
        dtype=np.int64
    )

    if seizures_samples.size == 0:
        seizures_samples = np.zeros((0, 2), dtype=np.int64)

    # Create output directory
    os.makedirs(
        os.path.dirname(h5_path),
        exist_ok=True
    )

    # Save HDF5
    with h5py.File(h5_path, "w") as f:

        f.create_dataset(
            "signals",
            data=data,
            dtype=np.float32,
            compression="gzip",
            compression_opts=4,
            shuffle=False,
            chunks=(data.shape[0], 1024)
        )

        f.create_dataset(
            "sampling_rate",
            data=fs
        )

        f.create_dataset(
            "channel_names",
            data=ch_names
        )

        f.create_dataset(
            "seizure_intervals",
            data=seizures_samples
        )

    print(f"✓ Saved → {h5_path}")

#  -------------------
# Batch processing
#  -------------------

def convert_dataset_filtered(
    seizure_map,
    DATASET_PATH,
    OUTPUT_PATH
):

    for file_name, seizures_sec in seizure_map.items():

        edf_path = os.path.join(
            DATASET_PATH,
            file_name
        )

        if not os.path.exists(edf_path):
            print(f"Missing: {edf_path}")
            continue

        h5_path = os.path.join(
            OUTPUT_PATH,
            file_name.replace(".edf", ".h5")
        )

        try:

            edf_to_filtered_h5(
                edf_path=edf_path,
                h5_path=h5_path,
                seizures_sec=seizures_sec
            )

        except Exception as e:

            print(f"✗ Error {file_name}: {e}")


nums = [f"{i:02d}" for i in range(1, 2)]

for numero in nums:

    # Summary file
    summary_path = f"../Dataset/chb{numero}/chb{numero}-summary.txt"

    seizure_map = parse_chbmit_summary(summary_path)

    print(f"\nProcessing patient chb{numero}")

    # Input / output folders
    DATASET_PATH = f"../Dataset/chb{numero}"
    OUTPUT_PATH = f"../Dataset_hdf5_filtered/chb{numero}"

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Run filtered conversion
    convert_dataset_filtered(
        seizure_map=seizure_map,
        DATASET_PATH=DATASET_PATH,
        OUTPUT_PATH=OUTPUT_PATH
    )
