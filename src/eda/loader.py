import numpy as np
import pandas as pd
import os


def load_eda(filepath, sfreq=None):
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in [".csv", ".txt"]:
        raise ValueError(f"Format non supporté pour EDA : {ext}. Utilisez CSV ou TXT.")

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
        signal = df.iloc[:, 1].values
    else:
        if sfreq is None:
            raise ValueError("Aucune colonne temps détectée. Veuillez fournir sfreq.")
        n = len(first_col)
        times = np.arange(n) / sfreq
        signal = first_col

    return signal, float(sfreq), times


def get_eda_info(signal, sfreq, times):
    return {
        "sfreq": sfreq,
        "n_samples": len(signal),
        "duration_sec": times[-1] - times[0],
        "amplitude_min": float(np.min(signal)),
        "amplitude_max": float(np.max(signal)),
    }