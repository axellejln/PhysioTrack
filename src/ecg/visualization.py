import matplotlib.pyplot as plt
import numpy as np


def plot_ecg_raw(times, signal):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(times, signal, color="crimson", linewidth=0.8)
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Signal ECG brut")
    plt.tight_layout()
    return fig


def plot_ecg_cleaned(times, signal, r_peaks=None):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(times, signal, color="steelblue", linewidth=0.8, label="Signal nettoyé")
    if r_peaks is not None and len(r_peaks) > 0:
        valid = r_peaks[r_peaks < len(times)]
        ax.plot(times[valid], signal[valid], "v", color="crimson",
                markersize=6, label=f"Pics R ({len(valid)})")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Signal ECG nettoyé avec pics R")
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def plot_rr_intervals(rr_ms):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(rr_ms, color="steelblue", marker="o", markersize=3, linewidth=0.8)
    ax.axhline(np.mean(rr_ms), color="crimson", linestyle="--",
               linewidth=1, label=f"Moyenne : {np.mean(rr_ms):.0f} ms")
    ax.set_xlabel("Battement n°")
    ax.set_ylabel("Intervalle RR (ms)")
    ax.set_title("Intervalles RR dans le temps")
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def plot_poincare(rr_ms):
    """Diagramme de Poincaré : RR[n] vs RR[n+1]"""
    rr_n  = rr_ms[:-1]
    rr_n1 = rr_ms[1:]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(rr_n, rr_n1, alpha=0.5, s=15, color="steelblue")
    lim_min = min(rr_ms) * 0.95
    lim_max = max(rr_ms) * 1.05
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            "r--", linewidth=1, label="Identité (RR[n]=RR[n+1])")
    ax.set_xlabel("RR[n] (ms)")
    ax.set_ylabel("RR[n+1] (ms)")
    ax.set_title("Diagramme de Poincaré")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    plt.tight_layout()
    return fig


def plot_fft(freqs, fft_values, zoom_min=None, zoom_max=None):
    mask = np.ones(len(freqs), dtype=bool)
    if zoom_min is not None:
        mask &= freqs >= zoom_min
    if zoom_max is not None:
        mask &= freqs <= zoom_max

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(freqs[mask], fft_values[mask], linewidth=0.8, color="darkorange")
    ax.set_xlabel("Fréquence (Hz)")
    ax.set_ylabel("Amplitude")
    title = "FFT du signal ECG"
    if zoom_min is not None or zoom_max is not None:
        title += f"  [{zoom_min or 0:.3f}–{zoom_max or freqs[-1]:.3f} Hz]"
    ax.set_title(title)

    bands = {
        "VLF": (0.003, 0.04,  "#4e79a7"),
        "LF":  (0.04,  0.15,  "#f28e2b"),
        "HF":  (0.15,  0.4,   "#e15759"),
    }
    zm = zoom_min or 0
    zx = zoom_max or freqs[-1]
    for label, (lo, hi, color) in bands.items():
        lo_v = max(lo, zm)
        hi_v = min(hi, zx)
        if lo_v < hi_v:
            ax.axvspan(lo_v, hi_v, alpha=0.12, color=color, label=label)
    ax.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    return fig


def plot_band_energy(band_energy):
    fig, ax = plt.subplots(figsize=(5, 3))
    colors = ["#4e79a7", "#f28e2b", "#e15759"]
    ax.bar(list(band_energy.keys()), list(band_energy.values()),
           color=colors[:len(band_energy)])
    ax.set_ylabel("Energie")
    ax.set_title("Energie par bande (VLF / LF / HF)")
    plt.tight_layout()
    return fig


def plot_hr(times, hr_values):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(times, hr_values, color="crimson", linewidth=0.8)
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("BPM")
    ax.set_title("Signal HR")
    plt.tight_layout()
    return fig