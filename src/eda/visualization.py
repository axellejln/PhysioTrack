import matplotlib.pyplot as plt
import numpy as np

def plot_eda(signal, times=None, title="EDA brut", color="steelblue", linewidth=0.8, ylabel="Amplitude (µS)"):
    times = times if times is not None else range(len(signal))
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(times, signal, color=color, linewidth=linewidth)
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_eda_decomposition(times, signal, tonic, phasic):
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(times, signal, color="gray", linewidth=0.8)
    axes[0].set_title("Signal EDA brut")
    axes[0].set_ylabel("Amplitude (µS)")
    axes[1].plot(times[:len(tonic)], tonic, color="steelblue", linewidth=0.8)
    axes[1].set_title("Composante tonique (SCL)")
    axes[1].set_ylabel("Amplitude (µS)")
    axes[2].plot(times[:len(phasic)], phasic, color="crimson", linewidth=0.8)
    axes[2].set_title("Composante phasique (SCR)")
    axes[2].set_ylabel("Amplitude (µS)")
    axes[2].set_xlabel("Temps (s)")
    plt.tight_layout()
    return fig


def plot_eda_peaks(times, phasic, peaks_idx):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(times[:len(phasic)], phasic, color="crimson", linewidth=0.8, label="Phasique")
    if len(peaks_idx) > 0:
        ax.plot(times[peaks_idx], phasic[peaks_idx],
                "v", color="navy", markersize=7, label=f"Pics ({len(peaks_idx)})")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Amplitude (µS)")
    ax.set_title("Pics phasiques détectés (SCR)")
    ax.legend()
    plt.tight_layout()
    return fig