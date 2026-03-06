import streamlit as st
import tempfile
import mne

import sys
import os

# Ajoute la racine du projet pour que 'src' soit trouvable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from src.eeg.loader import load_eeg
from src.eeg.preprocessing import bandpass_filter, crop_signal
from src.eeg.analysis import compute_fft, compute_band_energy
from src.eeg.visualization import plot_signal, plot_fft, plot_band_energy, plot_multiple_channels, plot_spectrogram


st.set_page_config(page_title="PhysioTrack", layout="centered")
if "page" not in st.session_state:
    st.session_state.page = "home"

# page d'accueil
if st.session_state.page == "home":
    st.title("PhysioTrack")
    st.subheader("Bienvenue dans l'interface de traitement de données physiologiques !")

    if st.button("Commencer"):
        st.session_state.page = "selection"

#sélection type de signal 
elif st.session_state.page == "selection":

    st.title("Choisissez votre type de signal")

    signal_type = st.radio(
        "Type de signal :",
        ["EEG", "ECG", "EDA"]
    )

    if st.button("Valider"):

        if signal_type == "EEG":
            st.session_state.page = "eeg"

        elif signal_type == "ECG":
            st.session_state.page = "ecg"

        elif signal_type == "EDA":
            st.session_state.page = "eda"

        st.rerun()

# page d'analyse EEG
elif st.session_state.page == "eeg":

    st.title("Analyse EEG")

    uploaded_file = st.file_uploader("Importer un fichier EEG (.edf)", type=["edf"])

    if uploaded_file is not None:

        # Crée un chemin temporaire
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Chargement EEG
        raw = load_eeg(tmp_path)

        st.success("Fichier chargé avec succès !")

        st.subheader("Informations du signal")
        st.write("Nombre de canaux :", len(raw.ch_names))
        st.write("Durée :", round(raw.times[-1],2), "sec")
        st.write("Fréquence d'échantillonnage :", raw.info["sfreq"], "Hz")

        st.divider()

        ##Pour plusieurs canaux
        st.header("Visualisation de plusieurs canaux")

        selected_channels = st.multiselect(
            "Choisir plusieurs canaux",
            raw.ch_names,
            default=raw.ch_names[:4]
        )

        if st.button("Afficher les canaux"):

            raw_multi = raw.copy().pick(selected_channels)

            fig_multi = plot_multiple_channels(raw_multi, selected_channels)

            st.pyplot(fig_multi)

        st.divider()

        ## Pour 1 seul canal 
        st.header("Analyse d'un canal")
        
        # Sélection canal
        channel = st.selectbox("Choisissez un canal :", raw.ch_names)

        # Paramètres temporels
        tmin = st.slider("Début segment (s)", 0, int(raw.times[-1]), 0)
        tmax = st.slider("Fin segment (s)", 0, int(raw.times[-1]), int(raw.times[-1]))

        # Paramètres filtre
        fmin = st.slider("Fréquence min (Hz)", 0, 20, 1)
        fmax = st.slider("Fréquence max (Hz)", 10, 80, 40)

        if st.button("Analyser canal"):

            raw_segment = raw.copy().crop(tmin, tmax)
            raw_segment.pick(channel)

            raw_filtered = bandpass_filter(raw_segment, fmin, fmax)

            # Signal brut
            st.subheader("Signal EEG brut")
            fig_raw = plot_signal(raw, channel)  # pas de filtre
            st.pyplot(fig_raw)

            #Signal filtré
            st.subheader(f"Signal filtré ({fmin}-{fmax} Hz)")
            fig_signal = plot_signal(raw_filtered, channel, original_times=raw_filtered.times + tmin)
            st.pyplot(fig_signal)


            #FFT
            st.subheader("FFT")

            freqs, fft_vals = compute_fft(raw_filtered, channel)

            fig_fft = plot_fft(freqs, fft_vals, channel)

            st.pyplot(fig_fft)

            st.subheader("Spectrogramme")

            fig_spec = plot_spectrogram(raw_filtered, channel)

            st.pyplot(fig_spec)

            st.subheader("Energie par bande")

            band_energy = compute_band_energy(freqs, fft_vals)

            fig_band = plot_band_energy(band_energy, channel)

            st.pyplot(fig_band)