import numpy as np
import matplotlib.pyplot as plt
import h5py


# ----------------------------------------------------------------------------------------------------------------------
# LOAD FILES
#----------------------------------------------------------------------------------------------------------------------

def load_h5(path):
    with h5py.File(path, "r") as f:
        signals = f["signals"][:]
        fs = f["sampling_rate"][()]
        ch_names = f["channel_names"][:]
        intervals = f["seizure_intervals"][:] if "seizure_intervals" in f else np.array([])
        
    ch_names = [
        c.decode() if isinstance(c, (bytes, bytearray)) else c
        for c in ch_names
    ]
    return signals, fs, ch_names, intervals



# ----------------------------------------------------------------------------------------------------------------------
# PLOT EEG GRAPHS 
#----------------------------------------------------------------------------------------------------------------------

def plot_eeg(
    signals,
    fs,
    ch_names=None,
    intervals=None,
    channels=None,        # list of indices OR names
    t_start=None,         # in seconds
    t_end=None,           # in seconds
    figsize=(15, 8),
    color_mode=None,
    seizure_color= '#bfd630',
    seizure_alpha = 0.4,
    title = 'EEG Signals'
):
    """
    Flexible EEG plotter.

    Parameters:
    - signals: (n_channels, n_samples)
    - fs: sampling frequency
    - ch_names: list of channel names
    - intervals: seizure intervals (samples or seconds)
    - channels: list of channel indices OR names to plot
    - t_start, t_end: time window in seconds
    - figsize: figure size
    """

    # --- Channel selection ---
    if channels is not None:
        if isinstance(channels[0], str):
            idx = [ch_names.index(ch) for ch in channels]
        else:
            idx = channels
        signals = signals[idx]
        if ch_names is not None:
            ch_names = [ch_names[i] for i in idx]

    n_channels, n_samples = signals.shape

    # --- Time window selection ---
    if t_start is None:
        t_start = 0
    if t_end is None:
        t_end = n_samples / fs

    start_sample = int(t_start * fs)
    end_sample = int(t_end * fs)

    signals = signals[:, start_sample:end_sample]
    t = np.arange(start_sample, end_sample) / fs



    # Color
    # --- COLOR SETUP ---
    if color_mode == "tab10":
        colors = plt.cm.tab10(np.linspace(0, 1, n_channels))
    elif color_mode == "viridis":
        colors = plt.cm.viridis(np.linspace(0, 1, n_channels))
    else:
        colors = ["gray"] * n_channels

        
    # --- Plot ---
    plt.figure(figsize=figsize)

    spacing = np.max(np.abs(signals)) * 2
    offset = 0

    yticks = []
    ylabels = []

    for i in range(n_channels):
        plt.plot(t, signals[i] + offset, color=colors[i])
        yticks.append(offset)
        if ch_names is not None:
            ylabels.append(ch_names[i])
        else:
            ylabels.append(f"Ch {i}")
        offset += spacing
    
    plt.yticks(yticks, ylabels)


    # --- Seizure highlighting (FIXED) ---
    if intervals is not None:
        intervals = np.array(intervals)

        if intervals.size > 0:
            if intervals.ndim == 1:
                intervals = intervals.reshape(1, -1)

            for start, end in intervals:

                # --- ALWAYS convert to seconds ---
                start = start / fs
                end = end / fs

                # --- check overlap with window ---
                if end >= t_start and start <= t_end:

                    # --- clip to visible region ---
                    start_clip = max(start, t_start)
                    end_clip = min(end, t_end)

                    plt.axvspan(start_clip, end_clip, color=seizure_color, alpha=seizure_alpha)

    plt.xlabel("Time (s)")
    plt.title(title)
    plt.tight_layout()
    plt.show()