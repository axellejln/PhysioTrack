import matplotlib.pyplot as plt

def plot_hr_signal(times, hr_values, r_peaks=None):
    fig, ax = plt.subplots()
    ax.plot(times, hr_values, label="HR")
    if r_peaks is not None:
        ax.plot(times[r_peaks], hr_values[r_peaks], "ro", label="R-peaks")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("BPM")
    ax.set_title("Signal HR")
    ax.legend()
    return fig

def plot_fft(freqs, fft_values):
    fig, ax = plt.subplots()
    ax.plot(freqs, fft_values)
    ax.set_xlabel("Fréquence (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title("FFT du signal ECG")
    return fig

def plot_band_energy(band_energy):
    fig, ax = plt.subplots()
    ax.bar(list(band_energy.keys()), list(band_energy.values()))
    ax.set_ylabel("Énergie")
    ax.set_title("Énergie par bande HR")
    return fig

def plot_hr(times, hr_values):
    fig, ax = plt.subplots()
    ax.plot(times, hr_values, label="HR")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("BPM")
    ax.set_title("Signal HR")
    ax.legend()
    return fig