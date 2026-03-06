import mne


def load_eeg(filepath):
    raw = mne.io.read_raw_edf(filepath, preload=True)
    return raw


def get_eeg_info(raw):
    info = {
        "sfreq": raw.info["sfreq"],
        "n_channels": raw.info["nchan"],
        "duration_sec": raw.times[-1],
        "channel_names": raw.ch_names
    }
    return info