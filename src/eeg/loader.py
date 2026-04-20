import mne
import numpy as np
import pandas as pd
import os

# imports de signaux 
def load_eeg_generic(file_path, sfreq=None, ch_names=None):
    ext = os.path.splitext(file_path)[1].lower()
    
    # ce qui est déjà supporté par MNE
    if ext in ['.edf', '.bdf']:
        raw = mne.io.read_raw_edf(file_path, preload=True)
    elif ext in ['.vhdr']:
        raw = mne.io.read_raw_brainvision(file_path, preload=True)
    elif ext in ['.set']:
        raw = mne.io.read_raw_eeglab(file_path, preload=True)
    elif ext in ['.fif']:
        raw = mne.io.read_raw_fif(file_path, preload=True)
    
    # formats CSV/TXT
    elif ext in ['.csv', '.txt']:
        if sfreq is None:
            raise ValueError("Pour CSV/TXT, vous devez fournir la fréquence d'échantillonnage sfreq")
        # gérer le header
        try:
            df = pd.read_csv(file_path)
        except Exception:
            df = pd.read_csv(file_path, header=None, skiprows=1)

        # Retirer toute colonne qui n'est pas numérique
        df = df.apply(pd.to_numeric, errors='coerce')

        # Supprimer les colonnes entièrement vides
        df = df.dropna(axis=1, how='all')

        first_col = df.iloc[:, 0].values
        if len(first_col) > 1 and np.all(np.diff(first_col) > 0):
            df = df.iloc[:, 1:]  # supprimer la colonne temps

        # Transposer pour avoir shape (n_channels x n_times)
        data = df.values.T

        # Noms des canaux
        if ch_names is None:
            ch_names = [f"Ch{i+1}" for i in range(data.shape[0])]

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data, info)
    else:
        raise ValueError(f"Format de fichier non supporté : {ext}")
    
    return raw
