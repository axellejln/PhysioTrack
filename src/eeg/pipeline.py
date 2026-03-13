from numpy import info
from matplotlib import pyplot as plt

from src.eeg.loader import load_eeg, get_eeg_info
from src.eeg.preprocessing import bandpass_filter, crop_signal
from src.eeg.visualization import plot_signal, plot_fft, plot_band_energy
from src.eeg.analysis import compute_fft, compute_band_energy

def eeg_pipeline(file_path, channel_index=0, tmin=None, tmax=50, fmin=1, fmax=40):
    raw = load_eeg(file_path)
    info = get_eeg_info(raw)
    print("Infos :", info)
    channel_name = info["channel_names"][channel_index]


    if tmin is not None and tmax is not None:
        raw = crop_signal(raw, tmin, tmax)

   
    raw_filtered = bandpass_filter(raw, fmin, fmax)

   
    fig_signal = plot_signal(raw_filtered, channel_name, original_times=raw_filtered.times + (tmin or 0))    # plot_signal(raw_filtered, channel_name)

 
    freqs, fft_values = compute_fft(raw_filtered, channel_name)
    fig_fft = plot_fft(freqs, fft_values, channel_name)
   
    
    band_energy = compute_band_energy(freqs, fft_values)
    print(f"Energie par bande pour {channel_name} :")
    for band, value in band_energy.items():
        print(f"{band}: {value:.2f}")

    fig_band = plot_band_energy(band_energy, channel_name)
    
    plt.show()

    return {"freqs": freqs, "fft_values": fft_values, "band_energy": band_energy}