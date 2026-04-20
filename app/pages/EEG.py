import streamlit as st
import tempfile
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import io

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.eeg.loader import load_eeg_generic
from src.eeg.preprocessing import bandpass_filter
from src.eeg.analysis import compute_fft, compute_band_energy
from src.eeg.visualization import (
    plot_signal, plot_fft, plot_band_energy,
    plot_multiple_channels, plot_spectrogram
)

st.set_page_config(page_title="EEG - PhysioTrack", page_icon="🧠", layout="centered")
st.title("Analyse EEG")

## IMPORTER UN SIGNAL 

with st.expander("Formats de fichiers acceptés : cliquez pour en savoir plus"):
    st.markdown("""
**Formats natifs** *(aucune configuration)* : `.edf`, `.bdf`, `.vhdr` (BrainVision), `.set` (EEGLAB), `.fif` (MNE)

**CSV / TXT** : 1 colonne par canal EEG, optionnellement précédée d'une colonne temps.
Si pas de colonne temps → sfreq à renseigner. Les noms des électrodes sont à confirmer après l'import.
```
temps,Fp1,Fp2,F3,F4          # ou sans colonne temps
0.000,0.0012,-0.0008,0.0021,-0.0015
0.004,0.0013,-0.0009,0.0019,-0.0014
```
    """)

uploaded_file = st.file_uploader(
    "Importer un fichier EEG",
    type=["edf", "bdf", "vhdr", "set", "fif", "csv", "txt"],
    help="Formats supportés : EDF, BDF, BrainVision (.vhdr), EEGLAB (.set), FIF, CSV, TXT"
)

if uploaded_file is None:
    st.stop()

# Vider les résultats si le fichier change
if st.session_state.get("_eeg_filename") != uploaded_file.name:
    st.session_state.pop("eeg_results", None)
    st.session_state["_eeg_filename"] = uploaded_file.name
    st.session_state["_fft_zoom_min"] = 0.0
    st.session_state["_fft_zoom_max"] = 0.0

ext = os.path.splitext(uploaded_file.name)[1].lower()

## CONFIGURATION POUR CSV/TXT
sfreq = None
ch_names_input = None

if ext in [".csv", ".txt"]:

    file_bytes = uploaded_file.read()

    try:
        df_preview = pd.read_csv(io.BytesIO(file_bytes))
    except Exception:
        df_preview = pd.read_csv(io.BytesIO(file_bytes), header=None, skiprows=1)

    df_preview = df_preview.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
    n_cols = df_preview.shape[1]
    first_col = df_preview.iloc[:, 0].values
    has_time_col = bool(np.all(np.diff(first_col) > 0))

    st.info(f"Fichier CSV/TXT détecté : **{n_cols} colonne(s)** trouvée(s).")

    # Noms des électrodes
    st.subheader("Noms des électrodes")
    n_signal_cols = n_cols - 1 if has_time_col else n_cols
    st.caption(
        f"{'1ère colonne = temps détectée automatiquement. ' if has_time_col else ''}"
        f"{n_signal_cols} canal(aux) de signal détecté(s)."
    )

    default_names = ", ".join([f"Ch{i+1}" for i in range(n_signal_cols)])
    ch_names_str = st.text_input(
        f"Entrez les noms des électrodes séparés par des virgules ({n_signal_cols} attendu(s))",
        value=default_names,
        help="Ex : Fp1, Fp2, F3, F4, C3, C4"
    )
    ch_names_input = [name.strip() for name in ch_names_str.split(",")]

    if len(ch_names_input) != n_signal_cols:
        st.warning(
            f"⚠️ {len(ch_names_input)} nom(s) saisi(s) mais {n_signal_cols} canal(aux) détecté(s). "
            "Corrigez les noms avant de continuer."
        )
        st.stop()

    # Fréquence d'échantillonnage 
    st.subheader("Fréquence d'échantillonnage")
    if has_time_col:
        sfreq = 1.0 / np.mean(np.diff(first_col))
        st.success(f"Fréquence calculée automatiquement depuis la colonne temps : **{sfreq:.2f} Hz**")
    else:
        sfreq = st.number_input(
            "Fréquence d'échantillonnage (Hz)",
            min_value=1.0, value=256.0, step=1.0,
            help="Fréquence à laquelle le signal a été enregistré"
        )

    st.divider()

    # Rembobiner pour le chargement
    uploaded_file = io.BytesIO(file_bytes)
    uploaded_file.name = f"signal{ext}"

## Chargement signal 
with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
    if hasattr(uploaded_file, 'seek'):
        uploaded_file.seek(0)
    tmp.write(uploaded_file.read())
    tmp_path = tmp.name

try:
    raw = load_eeg_generic(tmp_path, sfreq=sfreq, ch_names=ch_names_input)
    st.success("Fichier chargé avec succès !")
except Exception as e:
    st.error(f"Erreur lors du chargement : {e}")
    st.stop()

## Informations du signal
st.subheader("Informations du signal")
col1, col2, col3 = st.columns(3)
col1.metric("Canaux", len(raw.ch_names))
col2.metric("Durée", f"{raw.times[-1]:.1f} s")
col3.metric("Fréquence d'échantillonnage", f"{raw.info['sfreq']:.0f} Hz")

st.divider()

## VISUALISATION MULTICANAL
st.header("Visualisation de plusieurs canaux")

selected_channels = st.multiselect(
    "Choisir les canaux à afficher",
    raw.ch_names,
    default=raw.ch_names[:min(4, len(raw.ch_names))]
)

if st.button("Afficher les canaux sélectionnés"):
    if selected_channels:
        raw_multi = raw.copy().pick(selected_channels)
        fig_multi = plot_multiple_channels(raw_multi, selected_channels)
        st.pyplot(fig_multi)
    else:
        st.warning("Veuillez sélectionner au moins un canal.")

st.divider()

## ANALYSE D'UN CANAL
st.header("Analyse d'un canal")

channel = st.selectbox("Canal à analyser", raw.ch_names)

col_t1, col_t2 = st.columns(2)
with col_t1:
    tmin = st.slider("Début du segment (s)", 0, int(raw.times[-1]) - 1, 0)
with col_t2:
    tmax = st.slider("Fin du segment (s)", 1, int(raw.times[-1]), int(raw.times[-1]))

nyquist = int(raw.info["sfreq"] / 2)
col_f1, col_f2 = st.columns(2)
with col_f1:
    fmin = st.slider("Fréquence min du filtre (Hz)", 0, nyquist - 1, 1)
with col_f2:
    fmax = st.slider("Fréquence max du filtre (Hz)", 1, nyquist - 1, min(40, nyquist - 1))

if st.button("Analyser le canal"):
    if fmin >= fmax:
        st.error("La fréquence min doit être inférieure à la fréquence max.")
        st.stop()
    raw_segment = raw.copy().crop(tmin, tmax).pick(channel)
    raw_filtered = bandpass_filter(raw_segment, fmin, fmax)
    freqs, fft_vals = compute_fft(raw_filtered, channel)
    band_energy = compute_band_energy(freqs, fft_vals)

    st.session_state["eeg_results"] = {
        "raw": raw,
        "raw_filtered": raw_filtered,
        "channel": channel,
        "tmin": tmin,
        "fmin": fmin,
        "fmax": fmax,
        "freqs": freqs,
        "fft_vals": fft_vals,
        "band_energy": band_energy,
    }
    st.session_state["_fft_zoom_min"] = 0.0
    st.session_state["_fft_zoom_max"] = float(freqs[-1])

## Affichage des résultats
if "eeg_results" not in st.session_state:
    st.stop()

r            = st.session_state["eeg_results"]
raw          = r["raw"]
raw_filtered = r["raw_filtered"]
channel      = r["channel"]
tmin         = r["tmin"]
fmin         = r["fmin"]
fmax         = r["fmax"]
freqs        = r["freqs"]
fft_vals     = r["fft_vals"]
band_energy  = r["band_energy"]

## Signal brut + filtré côte à côte
st.subheader("Signal brut  vs  Signal filtré")
st.caption("Comparez l'effet du filtre passe-bande sur le signal. Les deux graphes sont à la même échelle.")

col_sig1, col_sig2 = st.columns(2)

with col_sig1:
    st.caption("**Signal brut**")
    fig_raw, ylim_raw = plot_signal(raw.copy().pick(channel), channel, color="steelblue")
    st.pyplot(fig_raw)

with col_sig2:
    st.caption(f"**Signal filtré ({fmin}–{fmax} Hz)**")
    fig_filt, _ = plot_signal(
        raw_filtered, channel,
        original_times=raw_filtered.times + tmin,
        color="crimson",
        ylim=ylim_raw
    )
    st.pyplot(fig_filt)

st.divider()

## FFT avec zoom interactif et bandes EEG colorées
st.subheader("FFT : Spectre de fréquences")

col_z1, col_z2, col_z3 = st.columns([2, 2, 1])
with col_z1:
    fft_zoom_min = st.number_input(
        "Zoom : fréq. min (Hz)",
        min_value=0.0,
        max_value=float(freqs[-1]),
        value=float(st.session_state.get("_fft_zoom_min", 0.0)),
        step=1.0,
    )
with col_z2:
    fft_zoom_max = st.number_input(
        "Zoom : fréq. max (Hz)",
        min_value=0.0,
        max_value=float(freqs[-1]),
        value=float(st.session_state.get("_fft_zoom_max", float(freqs[-1]))),
        step=1.0,
    )
with col_z3:
    st.write("")
    st.write("")
    if st.button("Reset"):
        st.session_state["_fft_zoom_min"] = 0.0
        st.session_state["_fft_zoom_max"] = float(freqs[-1])
        st.rerun()

st.session_state["_fft_zoom_min"] = fft_zoom_min
st.session_state["_fft_zoom_max"] = fft_zoom_max

fig_fft = plot_fft(freqs, fft_vals, channel,
                   zoom_min=fft_zoom_min, zoom_max=fft_zoom_max)
st.pyplot(fig_fft)
st.caption("Décompose le signal en fréquences. Les zones colorées indiquent les bandes cérébrales : delta (1-4 Hz) : sommeil, theta (4-8 Hz) : mémoire, alpha (8-12 Hz) : repos, beta (12-30 Hz) : concentration, gamma (30-40 Hz) : traitement sensoriel.")

st.divider()

## Spectrogramme
st.subheader("Spectrogramme")
fig_spec = plot_spectrogram(raw_filtered, channel, fmax=fmax)
st.pyplot(fig_spec)
st.caption("Montre comment le contenu fréquentiel évolue dans le temps. Utile pour détecter des transitions (ex : apparition d'alpha quand le sujet ferme les yeux).")

st.divider()

## Énergie par bande + tableau côte à côte
st.subheader("Énergie par bande de fréquence")

col_b1, col_b2 = st.columns([2, 1])
with col_b1:
    fig_band = plot_band_energy(band_energy, channel)
    st.pyplot(fig_band)
with col_b2:
    df_band = pd.DataFrame(
        list(band_energy.items()),
        columns=["Bande", "Énergie"]
    )
    df_band["Énergie"] = df_band["Énergie"].round(2)
    st.dataframe(df_band, use_container_width=True, hide_index=True)
    st.caption("Les bandes à 0 sont hors de la plage du filtre appliqué.")

st.divider()

## Export
st.header("Exporter les résultats")

data_filt_export = raw_filtered.copy().get_data()[0]
times_filt_export = raw_filtered.times + tmin
df_signal = pd.DataFrame({
    "temps_s": times_filt_export,
    "amplitude": data_filt_export
})
st.download_button(
    label="Signal filtré",
    data=df_signal.to_csv(index=False).encode("utf-8"),
    file_name=f"eeg_{channel}_filtre_{fmin}_{fmax}Hz.csv",
    mime="text/csv",
    use_container_width=True
)

df_band_export = pd.DataFrame(
    list(band_energy.items()),
    columns=["Bande", "Energie"]
)
df_band_export["Energie"] = df_band_export["Energie"].round(4)
st.download_button(
    label="Énergie par bande",
    data=df_band_export.to_csv(index=False).encode("utf-8"),
    file_name=f"eeg_{channel}_energie_bandes.csv",
    mime="text/csv",
    use_container_width=True
)

df_fft_export = pd.DataFrame({
    "frequence_hz": freqs,
    "amplitude": fft_vals
})
st.download_button(
    label="Spectre FFT",
    data=df_fft_export.to_csv(index=False).encode("utf-8"),
    file_name=f"eeg_{channel}_fft.csv",
    mime="text/csv",
    use_container_width=True
)