import streamlit as st
import tempfile
import mne
import sys
import os
import numpy as np
import pandas as pd

# Ajoute la racine du projet pour que 'src' soit trouvable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from src.eeg.loader import load_eeg, load_eeg_generic, get_eeg_info
from src.eeg.preprocessing import bandpass_filter, crop_signal
from src.eeg.analysis import compute_fft, compute_band_energy
from src.eeg.visualization import plot_signal, plot_fft, plot_band_energy, plot_multiple_channels, plot_spectrogram

from src.ecg.loader import load_ecg, load_ibi, load_hr, get_ecg_info

from src.eda.loader import load_eda, get_eda_info

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

## ANALYSE EDA
elif st.session_state.page == "eda":

    if st.button("← Retour"):
        st.session_state.page = "selection"
        st.rerun()
    st.title("Analyse EDA")

    uploaded_eda = st.file_uploader("Importer un fichier EDA - formats supportés : CSV/TXT", type=["csv","txt"], key="eda_raw") 
    if uploaded_eda is not None:    
        ext = os.path.splitext(uploaded_eda.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(uploaded_eda.read())
            tmp_path = tmp.name

        # Détection automatique si colonne temps présente ou saisie manuelle de sfreq
        try:
            df_preview = pd.read_csv(tmp_path)
        except Exception:
            df_preview = pd.read_csv(tmp_path, header=None, skiprows=1)
        df_preview = df_preview.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
        first_col = df_preview.iloc[:, 0].values

        if np.all(np.diff(first_col) > 0):
            sfreq_eda = None  # sera détecté dans load_eda
            st.info("Colonne temps détectée, fréquence d'échantillonnage calculée automatiquement.")
        else:
            sfreq_eda = st.number_input(
                "Fréquence d'échantillonnage (Hz)",
                min_value=1.0, value=256.0, step=1.0,
            )

        try:
            signal, sfreq, times = load_eda(tmp_path, sfreq=sfreq_eda)
            st.success("Fichier EDA chargé avec succès !")
            st.write(f"Fréquence d'échantillonnage : {sfreq:.2f} Hz")
            st.write(f"Durée du signal : {times[-1]:.2f} sec")
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier EDA : {e}")

## ANALYSE ECG

elif st.session_state.page == "ecg":

    if st.button("← Retour"):
        st.session_state.page = "selection"
        st.rerun()
    st.title("Analyse ECG")

    # choisir mode import
    st.subheader("Mode d'import")
    import_mode = st.radio(
        "Que souhaitez-vous importer ?",
        ["Signal ECG brut (CSV)", "Fichier IBI", "Fichier HR"],
        horizontal=True,
    )

    # signal ECG brut
    # import
    if import_mode == "Signal ECG brut (CSV)":
        uploaded_ecg = st.file_uploader("Importer un fichier ECG - formats supportés : CSV/TXT", type=["csv","txt"], key="ecg_raw")
        if uploaded_ecg is not None:
            ext = os.path.splitext(uploaded_ecg.name)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(uploaded_ecg.read())
                tmp_path = tmp.name

            # Détection automatique si colonne temps présente ou saisie manuelle de sfreq
            try:
                df_preview = pd.read_csv(tmp_path)
            except Exception:
                df_preview = pd.read_csv(tmp_path, header=None, skiprows=1)
            df_preview = df_preview.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
            first_col = df_preview.iloc[:, 0].values

            if np.all(np.diff(first_col) > 0):
                sfreq_ecg = None  # sera détecté dans load_ecg
                st.info("Colonne temps détectée, fréquence d'échantillonnage calculée automatiquement.")
            else:
                sfreq_ecg = st.number_input(
                    "Fréquence d'échantillonnage (Hz)",
                    min_value=1.0, value=256.0, step=1.0,
                )

            try:
                signal, sfreq, times = load_ecg(tmp_path)
                st.success("Fichier ECG chargé avec succès !")
                st.write(f"Fréquence d'échantillonnage : {sfreq:.2f} Hz")
                st.write(f"Durée du signal : {times[-1]:.2f} sec")
            except Exception as e:
                st.error(f"Erreur lors du chargement du fichier ECG : {e}")
            
            st.divider()
    
    # import IBI
    elif import_mode == "Fichier IBI":
        uploaded_ibi = st.file_uploader("Importer un fichier IBI - formats supportés : CSV/TXT", type=["csv","txt"], key="ecg_ibi")
        if uploaded_ibi is not None:
            ext = os.path.splitext(uploaded_ibi.name)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(uploaded_ibi.read())
                tmp_path = tmp.name

            try:
                ibi_times, ibi_values = load_ibi(tmp_path)
                st.success("Fichier IBI chargé avec succès !")
                st.write(f"Nombre d'intervalles détectées : {len(ibi_values)}")
                st.write(f"Durée totale : {ibi_times[-1]:.2f} sec")
            except Exception as e:
                st.error(f"Erreur lors du chargement du fichier IBI : {e}")

            st.divider()
            
    # import HR
    elif import_mode == "Fichier HR":
        uploaded_hr = st.file_uploader("Importer un fichier HR - formats supportés : CSV/TXT", type=["csv","txt"], key="ecg_hr")
        if uploaded_hr is not None:
            ext = os.path.splitext(uploaded_hr.name)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(uploaded_hr.read())
                tmp_path = tmp.name

            try:
                hr_times, hr_values = load_hr(tmp_path)
                st.success("Fichier HR chargé avec succès !")
                st.write(f"Nombre de battements détectés : {len(hr_values)}")
                st.write(f"Durée totale : {hr_times[-1]:.2f} sec")
            except Exception as e:
                st.error(f"Erreur lors du chargement du fichier HR : {e}")

            st.divider()


## ANALYSE EEG
elif st.session_state.page == "eeg":
    
    if st.button("← Retour"):
        st.session_state.page = "selection"
        st.rerun()
    st.title("Analyse EEG")

    uploaded_file = st.file_uploader("Importer un fichier EEG - formats supportés : EDF, BDF, BrainVision, EEGLAB, FIF, CSV/TXT (avec sfreq)", type=["edf", "bdf", "vhdr", "set", "fif", "csv", "txt"])

    if uploaded_file is not None:

        # Crée un chemin temporaire
        ext = os.path.splitext(uploaded_file.name)[1].lower() #repère le type de fichier
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        sfreq = None

        # si CSV/TXT, détecter colonne temps ou demander sfreq
        if ext in [".csv", ".txt"]:
    
            #lire avec la première ligne, si header = la sauter
            try:
                df = pd.read_csv(tmp_path)
            except Exception:
                df = pd.read_csv(tmp_path, header=None, skiprows=1)

            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.dropna(axis=1, how='all')
            
            # hypothèse : la première colonne est temps si elle est strictement croissante
            times = df.iloc[:, 0].values
            if np.all(np.diff(times) > 0):  # colonne temps détectée
                sfreq = 1 / np.mean(np.diff(times))
                st.info(f"Colonne temps détectée. Fréquence d'échantillonnage calculée : {sfreq:.2f} Hz")
                data = df.iloc[:, 1:].values.T  # supprime la colonne temps
            else:
                # pas de colonne temps → demander à l'utilisateur
                sfreq = st.number_input(
                    "Fréquence d'échantillonnage (Hz) pour CSV/TXT",
                    min_value=1.0,
                    value=256.0
                )
                

        # Charger le signal avec le loader générique
        try:
            if ext in [".csv", ".txt"]:
                raw = load_eeg_generic(tmp_path, sfreq=sfreq)
            else:
                raw = load_eeg_generic(tmp_path)

            st.success("Fichier chargé avec succès !")

            st.subheader("Informations du signal")
            st.write("Nombre de canaux :", len(raw.ch_names))
            st.write("Durée :", round(raw.times[-1],2), "sec")
            st.write("Fréquence d'échantillonnage :", raw.info["sfreq"], "Hz")

        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier : {e}")

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
        sfreq = raw.info["sfreq"]
        nyquist = int(sfreq / 2)
        fmin = st.slider("Fréquence min (Hz)", 0, nyquist-1, 1)
        fmax = st.slider("Fréquence max (Hz)", 0, nyquist-1, 40)

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

            #Spectrogramme
            st.subheader("Spectrogramme")
            fig_spec = plot_spectrogram(raw_filtered, channel)
            st.pyplot(fig_spec)

            #Energie par bande
            st.subheader("Energie par bande")
            band_energy = compute_band_energy(freqs, fft_vals)
            fig_band = plot_band_energy(band_energy, channel)
            st.pyplot(fig_band)