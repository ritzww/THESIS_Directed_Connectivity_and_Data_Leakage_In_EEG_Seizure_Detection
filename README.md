# Directed Connectivity and Data Leakage in EEG Seizure Detection

For the MSc thesis *Directed Connectivity and Data Leakage in EEG Seizure
Detection: Band-Limited Spectral Granger Causality* (Rita Wang, NOVA IMS,
supervised by Ian Scott).



## Repository layout

| Folder | Contents |
|---|---|
| `preprocess/` | Filtering of the EDF recordings and selection of the 18-channel montage |
| `feature_extraction/` | MATLAB scripts for VAR order selection and spectral Granger causality features (MVGC toolbox) |
| `models/` | The main nested LORO model and the two fixed-hyperparameter runs used in the leakage comparison |
| `visualisation_and_results/` | Notebooks that produce the tables and figures of the thesis |
| `csv_results/` | Result files written by the models and read by the notebooks |
| `csv_txt_files/` | Record lists, seizure annotations and model-order results |
| `archive/` | The notebooks as they were when the thesis was submitted |

## Data

The CHB-MIT Scalp EEG Database is available from PhysioNet:
https://physionet.org/content/chbmit/1.0.0/

The recordings and the extracted feature files are not included in this
repository because of their size.

## Requirements

- Python 3.12 with MNE-Python, NumPy, SciPy, pandas, h5py, scikit-learn 1.6.1,
  joblib and `timescoring` (SzCORE event scoring)
- MATLAB with the MVGC toolbox (Barnett & Seth, 2014)


## Where each result in the thesis comes from

| Thesis | Produced by | Result files |
|---|---|---|
| Section 3.1, raw EEG example | `visualisation_and_results/1. data_exploration.ipynb` | recordings |
| Section 3.2, filter figures | `archive/preprocessing_before_clean/1. Preprocessing.ipynb` (last cell) | recordings |
| Section 3.3.1, VAR model order | `visualisation_and_results/0. var_model_order.ipynb` | `csv_txt_files/var_order_all.csv` |
| Section 4.1, window-level results | `visualisation_and_results/3. segment_level_results.ipynb` | `csv_results/SEGMENTS_MAIN_MODEL.csv` |
| Section 4.2, event-level results | `visualisation_and_results/2. event_level_results.ipynb` | `csv_results/RESULTS_MAIN_MODEL.csv`, `EVENTS_MAIN_MODEL.csv`, `FALSE_POSITIVES_MAIN_MODEL.csv` |
| Section 4.4, temporal leakage | `visualisation_and_results/4. leakage_analysis.ipynb` | `csv_results/SEGMENTS_FIXED.csv`, `LEAKY_FIXED.csv` |
| Sections 4.5.1 and 4.5.2, selected features | `visualisation_and_results/6. feature_selection_analysis.ipynb` | `csv_results/LONG_SELECTED_FEATURES.csv` |
| Section 4.5.3, ictal changes in connectivity | `visualisation_and_results/7. ictal_connectivity_changes.ipynb` | `csv_results/ICTAL_VS_NONICTAL_ALLREC_SUMMARY.csv` |

## Result files

| File | One row per | Contents |
|---|---|---|
| `RESULTS_MAIN_MODEL.csv` | patient | Event-level counts and metrics, false alarms per hour and per day, results at k = 2, 3, 4 |
| `SEGMENTS_MAIN_MODEL.csv` | record | Selected hyperparameters, number of windows, window-level counts and metrics |
| `EVENTS_MAIN_MODEL.csv` | annotated seizure | Whether it was detected, and the latency |
| `FALSE_POSITIVES_MAIN_MODEL.csv` | false alarm | Start time and duration |
| `SEGMENTS_FIXED.csv` | record | Window-level results of the fixed-hyperparameter LORO run |
| `LEAKY_FIXED.csv` | patient and fold | Window-level results of the random-window cross-validation |
| `LONG_SELECTED_FEATURES.csv` | selected feature | Features selected by mRMR in each fold |
| `ICTAL_VS_NONICTAL_ALLREC_SUMMARY.csv` | patient and feature |  Ictal vs. non-ictal windows |

## Notes

- Random seeds are fixed in the scripts.
- `archive/` holds the notebooks as submitted. The current notebooks and
  scripts contain the same code with tidied comments and file names.
