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

- Python 3.12 
- MATLAB with the MVGC toolbox (Barnett & Seth, 2014)


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
