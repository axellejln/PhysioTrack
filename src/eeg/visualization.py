import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

# signal temporel EEG
def plot_signal(raw, channel_name, original_times=None, color="steelblue", figsize=(5, 3), ylim=None):
    """
    Args:
        ylim : tuple (ymin, ymax) optionnel pour forcer la même échelle sur plusieurs graphes.
    """
    data = raw.copy().pick(channel_name).get_data()[0]
    times = original_times if original_times is not None else raw.times
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(times, data, linewidth=0.6, color=color)
    ax.set_xlabel("Temps (s)", fontsize=8)
    ax.set_ylabel("Amplitude", fontsize=8)
    ax.set_title(f"{channel_name}", fontsize=9)
    ax.tick_params(labelsize=7)
    if ylim is not None:
        ax.set_ylim(ylim)
    plt.tight_layout()
    return fig, ax.get_ylim()

# Trace plusieurs canaux EEG en superposition
def plot_multiple_channels(raw, channel_names, original_times=None):
    fig, axes = plt.subplots(
        len(channel_names), 1,
        figsize=(10, 3 * len(channel_names)),
        sharex=True
    )
    if len(channel_names) == 1:
        axes = [axes]

    for i, channel_name in enumerate(channel_names):
        data = raw.copy().pick(channel_name).get_data()[0]
        times = original_times if original_times is not None else raw.times
        axes[i].plot(times, data, linewidth=0.6, color="steelblue")
        axes[i].set_ylabel("Amplitude", fontsize=8)
        axes[i].set_title(f"Signal EEG - {channel_name}", fontsize=9)
        axes[i].tick_params(labelsize=7)

    axes[-1].set_xlabel("Temps (s)", fontsize=8)
    plt.tight_layout()
    return fig

# FFT avec bandes d'énergie colorées
def plot_fft(freqs, fft_values, channel_name, zoom_min=None, zoom_max=None):
    zm = zoom_min if zoom_min is not None else 0.0
    zx = zoom_max if zoom_max is not None else float(freqs[-1])

    mask = (freqs >= zm) & (freqs <= zx)
    freqs_z = freqs[mask]
    fft_z   = fft_values[mask]

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(freqs_z, fft_z, linewidth=0.8, color="darkorange")
    ax.set_xlabel("Frequence (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"FFT — {channel_name}  [{zm:.0f}–{zx:.0f} Hz]")

    bands = {
        "delta (1-4)":  (1,  4,  "#4e79a7"),
        "theta (4-8)":  (4,  8,  "#f28e2b"),
        "alpha (8-12)": (8,  12, "#e15759"),
        "beta (12-30)": (12, 30, "#76b7b2"),
        "gamma (30-40)":(30, 40, "#59a14f"),
    }
    for label, (lo, hi, color) in bands.items():
        lo_v = max(lo, zm)
        hi_v = min(hi, zx)
        if lo_v < hi_v:
            ax.axvspan(lo_v, hi_v, alpha=0.10, color=color, label=label)

    ax.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    return fig

# Energie par bande EEG
def plot_band_energy(band_energy, channel_name):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(list(band_energy.keys()), list(band_energy.values()), color="skyblue")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Energie par bande pour {channel_name}")
    plt.tight_layout()
    return fig

# Spectrogramme temps-frequence coupé à fmax si fourni
def plot_spectrogram(raw, channel_name, fmax=None):
    data = raw.copy().pick(channel_name).get_data()[0]
    sfreq = raw.info["sfreq"]
    freqs, times, Sxx = spectrogram(data, sfreq)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.pcolormesh(times, freqs, Sxx, shading="gouraud")
    if fmax is not None:
        ax.set_ylim(0, fmax)
    ax.set_ylabel("Frequence (Hz)")
    ax.set_xlabel("Temps (s)")
    title = f"Spectrogramme — {channel_name}"
    if fmax is not None:
        title += f"  [0–{fmax} Hz]"
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig