import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

def plot_signal(raw, channel_name, original_times=None):
    data = raw.copy().pick(channel_name).get_data()[0]
    times = original_times if original_times is not None else raw.times

    fig, ax = plt.subplots()
    ax.plot(times, data)
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Signal EEG - {channel_name}")
    return fig

def plot_filtered_signal(raw_filtered, channel_name, original_times=None):
    data = raw_filtered.copy().pick(channel_name).get_data()[0]
    times = original_times if original_times is not None else raw_filtered.times

    fig, ax = plt.subplots()
    ax.plot(times, data)
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Signal EEG filtré - {channel_name}")
    return fig

def plot_multiple_channels(raw, channel_names, original_times=None):
    fig, axes = plt.subplots(len(channel_names), 1, figsize=(10, 3 * len(channel_names)), sharex=True)
    
    for i, channel_name in enumerate(channel_names):
        data = raw.copy().pick(channel_name).get_data()[0]
        times = original_times if original_times is not None else raw.times
        axes[i].plot(times, data)
        axes[i].set_ylabel("Amplitude")
        axes[i].set_title(f"Signal EEG - {channel_name}")
    
    axes[-1].set_xlabel("Temps (s)")
    plt.tight_layout()
    return fig

def plot_fft(freqs, fft_values, channel_name):

    fig, ax = plt.subplots()
    ax.plot(freqs, fft_values)
    ax.set_xlabel("Fréquence (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"FFT du canal {channel_name}")
    return fig


def plot_band_energy(band_energy, channel_name):
    fig, ax = plt.subplots()
    ax.bar(list(band_energy.keys()), list(band_energy.values()), color='skyblue')
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Energie par bande pour {channel_name}")
    return fig

def plot_spectrogram(raw, channel_name):
    data = raw.copy().pick(channel_name).get_data()[0]
    sfreq = raw.info["sfreq"]
    freqs, times, Sxx = spectrogram(data, sfreq)
    fig, ax = plt.subplots()
    im = ax.pcolormesh(times, freqs, Sxx, shading="gouraud")
    ax.set_ylabel("Fréquence (Hz)")
    ax.set_xlabel("Temps (s)")
    ax.set_title(f"Spectrogramme - {channel_name}")
    fig.colorbar(im, ax=ax)
    return fig