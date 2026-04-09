import numpy as np
import pandas as pd
import os


def load_ecg(filepath, sfreq=None):
    ext = os.path.splitext(filepath)[1].lower()

    if ext not in [".csv", ".txt"]:
        raise ValueError(f"Format non supporté pour ECG : {ext}. Utilisez CSV ou TXT.")

    try:
        df = pd.read_csv(filepath)
    except Exception:
        df = pd.read_csv(filepath, header=None, skiprows=1)

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")

    first_col = df.iloc[:, 0].values

    if np.all(np.diff(first_col) > 0):
        # Colonne temps détectée
        times = first_col
        sfreq = 1.0 / np.mean(np.diff(times))
        signal = df.iloc[:, 1].values  # 1ère colonne signal
    else:
        if sfreq is None:
            raise ValueError("Aucune colonne temps détectée. Veuillez fournir sfreq.")
        n = len(first_col)
        times = np.arange(n) / sfreq
        signal = first_col

    return signal, float(sfreq), times


def load_ibi(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in [".csv", ".txt"]:
        raise ValueError(f"Format non supporté pour IBI : {ext}")

    try:
        df = pd.read_csv(filepath)
    except Exception:
        df = pd.read_csv(filepath, header=None, skiprows=1)

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")

    if df.shape[1] >= 2:
        ibi_times = df.iloc[:, 0].values.astype(float)
        ibi_values = df.iloc[:, 1].values.astype(float)
    else:
        ibi_values = df.iloc[:, 0].values.astype(float)
        # Reconstituer les temps cumulés à partir des IBI
        ibi_times = np.cumsum(ibi_values)
        ibi_times = np.insert(ibi_times[:-1], 0, 0.0)

    # Conversion ms → s si les valeurs semblent en ms (> 10 en moyenne)
    if np.mean(ibi_values) > 10:
        ibi_values = ibi_values / 1000.0

    return ibi_times, ibi_values


def load_hr(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in [".csv", ".txt"]:
        raise ValueError(f"Format non supporte pour HR : {ext}")

    # Detection format E4 : ligne 1 = timestamp Unix, ligne 2 = sfreq
    skiprows = 0
    sfreq_e4 = None
    with open(filepath, "r") as f:
        lines = f.readlines()
    try:
        first_val = float(lines[0].strip().split(",")[0])
        second_val = float(lines[1].strip().split(",")[0])
        if first_val > 1e9:  # timestamp Unix detecte
            skiprows = 2
            sfreq_e4 = second_val
    except Exception:
        pass

    try:
        df = pd.read_csv(filepath, header=None, skiprows=skiprows)
    except Exception:
        df = pd.read_csv(filepath, header=None, skiprows=skiprows + 1)

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")

    if df.shape[1] >= 2:
        hr_times  = df.iloc[:, 0].values.astype(float)
        hr_values = df.iloc[:, 1].values.astype(float)
    else:
        hr_values = df.iloc[:, 0].values.astype(float)
        sfreq_used = sfreq_e4 if sfreq_e4 else 1.0
        hr_times = np.arange(len(hr_values), dtype=float) / sfreq_used

    return hr_times, hr_values


def get_ecg_info(signal, sfreq, times):
    return {
        "sfreq": sfreq,
        "n_samples": len(signal),
        "duration_sec": times[-1] - times[0],
        "amplitude_min": float(np.min(signal)),
        "amplitude_max": float(np.max(signal)),
    }