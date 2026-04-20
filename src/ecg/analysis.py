import neurokit2 as nk
import numpy as np


def filter_hr_values(hr_values, hr_times):
    valid_mask = (hr_values >= 20) & (hr_values <= 250)
    return hr_values[valid_mask], hr_times[valid_mask], int(np.sum(~valid_mask))


def clean_hr(hr_values, sfreq=1):
    hr_cleaned = nk.ecg_clean(hr_values, sampling_rate=sfreq)
    return hr_cleaned


def detect_r_peaks(hr_cleaned, sfreq=1):
    signals, info = nk.ecg_peaks(hr_cleaned, sampling_rate=sfreq)

    # Compatibilite multi-versions NeuroKit2 :
    # info["ECG_R_Peaks"] peut etre un array numpy ou une Series pandas
    r_peaks_raw = info["ECG_R_Peaks"]
    if hasattr(r_peaks_raw, "values"):
        r_peaks = r_peaks_raw.values
    else:
        r_peaks = np.array(r_peaks_raw)

    # Fallback : chercher dans signals si info ne contient rien d'utile
    if r_peaks is None or len(r_peaks) == 0:
        r_peaks = np.where(signals["ECG_R_Peaks"].values == 1)[0]

    r_peaks = r_peaks.astype(int)
    rr_intervals = np.diff(r_peaks)
    return r_peaks, rr_intervals


def compute_fft(signal, sfreq=1.0):
    # Soustraire la moyenne (offset DC) avant FFT
    # Evite le pic geant a 0 Hz qui ecrase le reste du spectre
    signal_centered = signal - np.mean(signal)
    n = len(signal_centered)
    freqs = np.fft.rfftfreq(n, d=1 / sfreq)
    fft_values = np.abs(np.fft.rfft(signal_centered))
    return freqs, fft_values


def compute_band_energy(freqs, fft_values):
    bands = {"VLF": (0.003, 0.04), "LF": (0.04, 0.15), "HF": (0.15, 0.4)}
    band_energy = {}
    for name, (low, high) in bands.items():
        idx = np.logical_and(freqs >= low, freqs <= high)
        band_energy[name] = np.sum(fft_values[idx] ** 2)
    return band_energy


def compute_hrv_advanced(rr_intervals, sfreq=1.0):
    """
    Calcule les indicateurs HRV avances depuis les intervalles RR (en samples).
    """
    rr_ms = rr_intervals / sfreq * 1000  # samples -> ms

    mean_rr = float(np.mean(rr_ms))
    sdnn    = float(np.std(rr_ms, ddof=1))
    rmssd   = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2)))
    hr_mean = 60000.0 / mean_rr if mean_rr > 0 else 0.0
    hr_min  = 60000.0 / float(np.max(rr_ms)) if np.max(rr_ms) > 0 else 0.0
    hr_max  = 60000.0 / float(np.min(rr_ms)) if np.min(rr_ms) > 0 else 0.0

    diffs = np.abs(np.diff(rr_ms))
    pnn50 = float(np.sum(diffs > 50) / len(diffs) * 100) if len(diffs) > 0 else 0.0

    if len(rr_ms) > 4:
        freqs_rr, fft_rr = compute_fft(rr_ms, sfreq=1.0)
        be = compute_band_energy(freqs_rr, fft_rr)
        lf_hf = be["LF"] / be["HF"] if be["HF"] > 0 else 0.0
    else:
        lf_hf = 0.0

    return {
        "Moyenne RR (ms)": round(mean_rr, 1),
        "SDNN (ms)":       round(sdnn, 1),
        "RMSSD (ms)":      round(rmssd, 1),
        "pNN50 (%)":       round(pnn50, 1),
        "HR moyen (BPM)":  round(hr_mean, 1),
        "HR min (BPM)":    round(hr_min, 1),
        "HR max (BPM)":    round(hr_max, 1),
        "Ratio LF/HF":     round(lf_hf, 3),
    }