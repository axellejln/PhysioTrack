import numpy as np
import pandas as pd
import os


def _is_unix_timestamp(value):
    """Detecte si une valeur ressemble a un timestamp Unix (>1e9)."""
    return value > 1e9


def load_eda(filepath, sfreq=None):
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in [".csv", ".txt"]:
        raise ValueError(f"Format non supporte pour EDA : {ext}. Utilisez CSV ou TXT.")

    # Lire toutes les lignes brutes pour detecter le format E4
    with open(filepath, "r") as f:
        raw_lines = f.readlines()

    # Convertir chaque ligne en float si possible
    values_raw = []
    for line in raw_lines:
        try:
            values_raw.append(float(line.strip().split(",")[0]))
        except Exception:
            values_raw.append(None)

    # Detecter le format E4 (Empatica) :
    # ligne 1 = timestamp Unix (>1e9), ligne 2 = sfreq, reste = signal
    skiprows = 0
    if (len(values_raw) >= 2
            and values_raw[0] is not None
            and values_raw[1] is not None
            and _is_unix_timestamp(values_raw[0])):
        # Format E4 detecte
        if sfreq is None:
            sfreq = float(values_raw[1])
        skiprows = 2

    # Charger le signal en sautant les lignes d'en-tete
    try:
        df = pd.read_csv(filepath, header=None, skiprows=skiprows)
    except Exception:
        raise ValueError("Impossible de lire le fichier EDA.")

    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    first_col = df.iloc[:, 0].values

    # Cas 1 : colonne temps strictement croissante
    if skiprows == 0 and df.shape[1] >= 2 and np.all(np.diff(first_col) > 0):
        times = first_col
        sfreq = 1.0 / np.mean(np.diff(times))
        signal = df.iloc[:, 1].values

    # Cas 2 : format E4 ou colonne unique sans temps
    else:
        if sfreq is None:
            raise ValueError(
                "Aucune colonne temps detectee et sfreq non fournie. "
                "Veuillez indiquer la frequence d'echantillonnage."
            )
        signal = first_col
        n = len(signal)
        times = np.arange(n) / sfreq

    return signal.astype(float), float(sfreq), times.astype(float)


def get_eda_info(signal, sfreq, times):
    return {
        "sfreq":         sfreq,
        "n_samples":     len(signal),
        "duration_sec":  float(times[-1] - times[0]),
        "amplitude_min": float(np.min(signal)),
        "amplitude_max": float(np.max(signal)),
    }