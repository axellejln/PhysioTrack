import numpy as np


def compute_fft(raw, channel_name):
    data = raw.copy().pick(channel_name).get_data()[0]
    sfreq = raw.info["sfreq"]
    n = len(data)
    freqs = np.fft.rfftfreq(n, d=1/sfreq)
    fft_values = np.abs(np.fft.rfft(data))
    return freqs, fft_values



def band_power(freqs, fft_values, band):
    idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    power = np.sum(fft_values[idx])  # juste la somme des amplitudes
    return power


def compute_band_energy(freqs, fft_values):
    bands = {"Delta": (1, 4), "Theta": (4, 8), "Alpha": (8, 12),
             "Beta": (12, 30), "Gamma": (30, 40)}
    band_energy = {name: band_power(freqs, fft_values, band) for name, band in bands.items()}
    return band_energy
