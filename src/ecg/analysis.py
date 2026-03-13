import neurokit2 as nk
import numpy as np

#filtre passe-bande
def bandpass_filter(raw, l_freq=1.0, h_freq=40.0):
    filtered = raw.copy().filter(l_freq=l_freq, h_freq=h_freq)
    return filtered


def clean_hr(hr_values, sfreq=1):
    hr_cleaned = nk.ecg_clean(hr_values, sampling_rate=sfreq)
    return hr_cleaned

def detect_r_peaks(hr_cleaned, sfreq=1):
    signals, info = nk.ecg_peaks(hr_cleaned, sampling_rate=sfreq)
    r_peaks = info['ECG_R_Peaks'].values
    rr_intervals = np.diff(r_peaks)
    return r_peaks, rr_intervals

def compute_fft(signal, sfreq=1.0):
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1/sfreq)
    fft_values = np.abs(np.fft.rfft(signal))
    return freqs, fft_values

def compute_band_energy(freqs, fft_values):
    bands = {"VLF": (0.003, 0.04), "LF": (0.04, 0.15), "HF": (0.15, 0.4)}
    band_energy = {}
    for name, (low, high) in bands.items():
        idx = np.logical_and(freqs >= low, freqs <= high)
        band_energy[name] = np.sum(fft_values[idx]**2)
    return band_energy

def hr_pipeline(hr_values, sfreq=1):
    hr_cleaned = clean_hr(hr_values, sfreq)
    r_peaks, rr_intervals = detect_r_peaks(hr_cleaned, sfreq)
    freqs, fft_values = compute_fft(hr_cleaned, sfreq)
    band_energy = compute_band_energy(freqs, fft_values)
    
    return {
        "cleaned": hr_cleaned,
        "r_peaks": r_peaks,
        "rr_intervals": rr_intervals,
        "freqs": freqs,
        "fft_values": fft_values,
        "band_energy": band_energy
    }