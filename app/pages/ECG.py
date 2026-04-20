import streamlit as st
import tempfile
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.ecg.loader import load_ecg, load_ibi, load_hr
from src.ecg.analysis import (
    filter_hr_values, clean_hr, detect_r_peaks, compute_fft,
    compute_band_energy, compute_hrv_advanced
)
from src.ecg.visualization import (
    plot_ecg_raw, plot_rr_intervals,
    plot_poincare, plot_fft, plot_band_energy, plot_hr
)

st.set_page_config(page_title="ECG - PhysioTrack", layout="centered")
st.title("Analyse ECG")

## IMPORTER UN SIGNAL
import_mode = st.radio(
    "Type de fichier à importer",
    ["Signal ECG brut (CSV/TXT)", "Fichier IBI", "Fichier HR"],
    horizontal=True
)

# Vider les résultats si le mode change
if st.session_state.get("_ecg_mode") != import_mode:
    st.session_state.pop("ecg_results", None)
    st.session_state["_ecg_mode"] = import_mode
    st.session_state["_ecg_fft_zoom_min"] = 0.0
    st.session_state["_ecg_fft_zoom_max"] = 0.0

with st.expander("Format de fichier attendu : cliquez pour en savoir plus"):
    if import_mode == "Signal ECG brut (CSV/TXT)":
        st.markdown("""
CSV avec **1 colonne signal** (forme d'onde en mV/µV), optionnellement précédée d'une colonne temps.
Si pas de colonne temps → sfreq à renseigner manuellement. ⚠️ Pas des valeurs BPM : utilisez **Fichier HR** pour ça.
```
temps,ECG          # ou juste une colonne ECG sans temps
0.000,0.0023
0.004,0.0031
```
        """)
    elif import_mode == "Fichier IBI":
        st.markdown("""
CSV avec les intervalles inter-battements en **secondes** (ou ms, converti auto si valeurs > 10).
Formats acceptés : `temps,IBI` / `IBI` seul / format **Empatica E4** (timestamp Unix en ligne 1, détecté auto).
```
1544027337.000000, IBI    # format E4
84.847634,0.468771        # ou temps(s), IBI(s)
85.347657,0.500023
```
        """)
    elif import_mode == "Fichier HR":
        st.markdown("""
CSV avec la fréquence cardiaque en **BPM**. Avec ou sans colonne temps (sans → 1 valeur/seconde assumée).
Exporté par : Empatica E4 (HR.csv), Polar, Fitbit, Apple Watch.
```
temps,HR    # ou juste une colonne HR
0.0,72
1.0,74
```
        """)

## VARIABLES COMMUNES
signal = None
sfreq  = None
times  = None
mode   = None
rr_ms_direct = None  # intervalles RR en ms (pour IBI et HR, avant analyse)

## ECG BRUT
if import_mode == "Signal ECG brut (CSV/TXT)":
    mode = "ecg"
    uploaded = st.file_uploader("Importer un fichier ECG", type=["csv", "txt"], key="ecg_raw")
    if uploaded:
        ext = os.path.splitext(uploaded.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        try:
            signal, sfreq, times = load_ecg(tmp_path)
            st.success("Fichier ECG chargé avec succès !")
            c1, c2, c3 = st.columns(3)
            c1.metric("Fréquence d'échantillonnage", f"{sfreq:.1f} Hz")
            c2.metric("Durée", f"{times[-1]:.1f} s")
            c3.metric("Nb d'échantillons", len(signal))
        except Exception as e:
            st.error(f"Erreur : {e}")

## IBI
elif import_mode == "Fichier IBI":
    mode = "ibi"
    uploaded = st.file_uploader("Importer un fichier IBI", type=["csv", "txt"], key="ecg_ibi")
    if uploaded:
        ext = os.path.splitext(uploaded.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        try:
            ibi_times, ibi_values = load_ibi(tmp_path)
            # Convertir IBI (s) → RR (ms) et reconstruire une timeline
            rr_ms_direct = ibi_values * 1000
            times = np.cumsum(ibi_values)  # positions temporelles des battements
            signal = rr_ms_direct          # on passe les RR comme "signal"
            sfreq  = 1.0                   # 1 valeur par battement
            st.success("Fichier IBI chargé avec succès !")
            c1, c2, c3 = st.columns(3)
            c1.metric("Intervalles détectés", len(ibi_values))
            c2.metric("Durée totale", f"{times[-1]:.1f} s")
            c3.metric("RR moyen", f"{np.mean(rr_ms_direct):.0f} ms")
        except Exception as e:
            st.error(f"Erreur : {e}")

## HR
elif import_mode == "Fichier HR":
    mode = "hr"
    uploaded = st.file_uploader("Importer un fichier HR", type=["csv", "txt"], key="ecg_hr")
    if uploaded:
        ext = os.path.splitext(uploaded.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        try:
            hr_times, hr_values = load_hr(tmp_path)
            hr_values, hr_times, n_invalides = filter_hr_values(hr_values, hr_times)
            # Convertir BPM → RR ms
            rr_ms_direct = 60000.0 / hr_values
            signal = hr_values
            times  = hr_times
            sfreq  = 1.0 / np.mean(np.diff(hr_times)) if len(hr_times) > 1 else 1.0
            st.success("Fichier HR chargé avec succès !")
            if n_invalides > 0:
                st.warning(f"⚠️ {n_invalides} valeur(s) aberrante(s) retirée(s) (BPM hors de [20–250]).")
            c1, c2, c3 = st.columns(3)
            c1.metric("Durée", f"{hr_times[-1]:.1f} s")
            c2.metric("HR moyen", f"{np.mean(hr_values):.0f} BPM")
            c3.metric("RR moyen", f"{np.mean(rr_ms_direct):.0f} ms")
        except Exception as e:
            st.error(f"Erreur : {e}")

## AFFICHAGE DU SIGNAL BRUT
if signal is not None:
    st.divider()
    st.subheader("Signal brut")
    if mode == "ecg":
        fig_raw = plot_ecg_raw(times, signal)
    elif mode == "hr":
        fig_raw = plot_hr(times, signal)
    else:  # ibi
        fig_raw, ax = plt.subplots(figsize=(10, 3))
        marker = "o" if len(times) < 200 else None
        ax.plot(times, rr_ms_direct, color="crimson", linewidth=0.8,
                marker=marker, markersize=3)
        ax.axhline(np.mean(rr_ms_direct), color="steelblue", linestyle="--",
                   linewidth=1, label=f"Moy. {np.mean(rr_ms_direct):.0f} ms")
        ax.set_xlabel("Temps (s)")
        ax.set_ylabel("RR (ms)")
        ax.set_title("Intervalles IBI")
        ax.legend(fontsize=8)
        plt.tight_layout()
    st.pyplot(fig_raw)

    st.divider()

    label_btn = "Analyser le signal"
    if st.button(label_btn):
        try:
            if mode == "ecg":
                hr_cleaned = clean_hr(signal, sfreq=int(sfreq))
                r_peaks, rr_intervals = detect_r_peaks(hr_cleaned, sfreq=int(sfreq))
                rr_ms = rr_intervals / sfreq * 1000
                freqs, fft_vals = compute_fft(hr_cleaned, sfreq=sfreq)
                band_energy = compute_band_energy(freqs, fft_vals)
                hrv = compute_hrv_advanced(rr_intervals, sfreq=sfreq)
                hr_cleaned_out = hr_cleaned
                r_peaks_out = r_peaks
            else:
                # IBI et HR : RR déjà calculés
                rr_ms = rr_ms_direct
                hr_cleaned_out = signal.copy()
                r_peaks_out = np.array([])
                # FFT sur les RR (1 valeur par battement)
                freqs, fft_vals = compute_fft(rr_ms, sfreq=1.0)
                band_energy = compute_band_energy(freqs, fft_vals)
                hrv = compute_hrv_advanced(rr_ms, sfreq=1000.0)

            st.session_state["ecg_results"] = {
                "signal":       signal,
                "times":        times,
                "sfreq":        sfreq,
                "hr_cleaned":   hr_cleaned_out,
                "r_peaks":      r_peaks_out,
                "rr_ms":        rr_ms,
                "freqs":        freqs,
                "fft_vals":     fft_vals,
                "band_energy":  band_energy,
                "hrv":          hrv,
                "mode":         mode,
            }
            st.session_state["_ecg_fft_zoom_min"] = 0.003
            st.session_state["_ecg_fft_zoom_max"] = float(freqs[-1])

        except Exception as e:
            st.error(f"Erreur lors de l'analyse : {e}")

## AFFICHAGE DES RÉSULTATS
if "ecg_results" not in st.session_state:
    st.stop()

r           = st.session_state["ecg_results"]
signal      = r["signal"]
times       = r["times"]
sfreq       = r["sfreq"]
hr_cleaned  = r["hr_cleaned"]
r_peaks     = r["r_peaks"]
rr_ms       = r["rr_ms"]
freqs       = r["freqs"]
fft_vals    = r["fft_vals"]
band_energy = r["band_energy"]
hrv         = r["hrv"]
mode        = r["mode"]

# ECG brut : signal brut vs signal nettoyé côte à côte
# HR / IBI : signal brut pleine largeur seulement (RR déjà affichés dans signal brut)
if mode == "ecg":
    st.subheader("Signal brut  vs  Signal nettoyé")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.caption("**Signal brut**")
        fig_b, ax = plt.subplots(figsize=(5, 3))
        ax.plot(times, signal, linewidth=0.6, color="crimson")
        ax.set_xlabel("Temps (s)", fontsize=8)
        ax.set_ylabel("Amplitude", fontsize=8)
        ax.set_title("Signal brut", fontsize=9)
        ax.tick_params(labelsize=7)
        plt.tight_layout()
        st.pyplot(fig_b)
    with col_s2:
        st.caption("**Signal nettoyé avec pics R**")
        fig_c, ax = plt.subplots(figsize=(5, 3))
        ax.plot(times[:len(hr_cleaned)], hr_cleaned, linewidth=0.6, color="steelblue")
        valid = r_peaks[r_peaks < len(times)]
        ax.plot(times[valid], hr_cleaned[valid], "v", color="crimson",
                markersize=5, label=f"Pics R ({len(valid)})")
        ax.set_xlabel("Temps (s)", fontsize=8)
        ax.set_ylabel("Amplitude", fontsize=8)
        ax.set_title("Signal nettoyé", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
        plt.tight_layout()
        st.pyplot(fig_c)

st.divider()

# Intervalles RR pleine largeur + Poincaré en dessous
st.subheader("Intervalles RR")
fig_rr = plot_rr_intervals(rr_ms)
st.pyplot(fig_rr)
st.caption("Temps entre deux battements consécutifs (ms). Une variabilité élevée est généralement signe de bonne santé cardiovasculaire.")

st.subheader("Diagramme de Poincaré")
fig_pc = plot_poincare(rr_ms)
st.pyplot(fig_pc)
st.caption("Chaque point représente un battement RR[n] et le suivant RR[n+1]. Un nuage allongé = bonne variabilité, un nuage compact = peu de variabilité (stress, fatigue). *Task Force, Circulation, 1996*")

st.divider()

# HRV
st.subheader("Variabilité de la fréquence cardiaque (HRV)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("HR moyen",  f"{hrv['HR moyen (BPM)']} BPM")
c2.metric("SDNN",      f"{hrv['SDNN (ms)']} ms")
c3.metric("RMSSD",     f"{hrv['RMSSD (ms)']} ms")
c4.metric("pNN50",     f"{hrv['pNN50 (%)']} %")

hrv_df = pd.DataFrame(list(hrv.items()), columns=["Indicateur", "Valeur"])
st.dataframe(hrv_df, use_container_width=True, hide_index=True)

with st.expander("ℹ️ Comprendre les indicateurs HRV"):
    st.markdown("""
| Indicateur | Signification | Interprétation |
|------------|---------------|----------------|
| **Moyenne RR (ms)** | Durée moyenne entre deux battements | 600–1000 ms au repos = normal |
| **SDNN (ms)** | Écart-type de tous les intervalles RR | Variabilité globale : plus élevée = meilleure santé cardiovasculaire |
| **RMSSD (ms)** | Racine des différences successives RR | Activité parasympathique : indicateur de stress et récupération |
| **pNN50 (%)** | % de RR consécutifs différant de > 50 ms | Complète RMSSD : très bas = signal peu variable (stress, fatigue) |
| **HR moyen/min/max** | Fréquence cardiaque en BPM | - |
| **Ratio LF/HF** | Énergie basse fréq. / haute fréq. | Équilibre sympathique/parasympathique : élevé = activation/stress |

*Référence : Task Force of the European Society of Cardiology, Circulation, 93(5):1043–1065, 1996.*
    """)

if mode == "hr":
    st.caption("⚠️ Signal HR à 1 Hz : RMSSD et pNN50 sous-estimés. Utilisez un fichier IBI pour plus de précision.")

st.divider()

# FFT avec zoom
st.subheader("FFT : Spectre de fréquences")
col_z1, col_z2, col_z3 = st.columns([2, 2, 1])
with col_z1:
    fft_zoom_min = st.number_input(
        "Zoom : fréq. min (Hz)",
        min_value=0.0, max_value=float(freqs[-1]),
        value=float(st.session_state.get("_ecg_fft_zoom_min", 0.003)),
        step=0.001, format="%.3f"
    )
with col_z2:
    fft_zoom_max = st.number_input(
        "Zoom : fréq. max (Hz)",
        min_value=0.0, max_value=float(freqs[-1]),
        value=float(st.session_state.get("_ecg_fft_zoom_max", float(freqs[-1]))),
        step=0.01, format="%.3f"
    )
with col_z3:
    st.write("")
    st.write("")
    if st.button("🔄 Reset"):
        st.session_state["_ecg_fft_zoom_min"] = 0.003
        st.session_state["_ecg_fft_zoom_max"] = float(freqs[-1])
        st.rerun()

st.session_state["_ecg_fft_zoom_min"] = fft_zoom_min
st.session_state["_ecg_fft_zoom_max"] = fft_zoom_max

fig_fft = plot_fft(freqs, fft_vals, zoom_min=fft_zoom_min, zoom_max=fft_zoom_max)
st.pyplot(fig_fft)
st.caption("Décomposition fréquentielle du rythme cardiaque. VLF (0.003–0.04 Hz) : thermorégulation : LF (0.04–0.15 Hz) : système sympathique : HF (0.15–0.4 Hz) : système parasympathique, lié à la respiration.")

st.divider()

# Énergie par bande
st.subheader("Énergie par bande (VLF / LF / HF)")
col_b1, col_b2 = st.columns([2, 1])
with col_b1:
    fig_band = plot_band_energy(band_energy)
    st.pyplot(fig_band)
with col_b2:
    df_band = pd.DataFrame(list(band_energy.items()), columns=["Bande", "Énergie"])
    df_band["Énergie"] = df_band["Énergie"].round(4)
    st.dataframe(df_band, use_container_width=True, hide_index=True)
    st.caption("Un ratio LF/HF élevé indique une activation sympathique (stress ou effort).")

st.divider()

# Export
st.header("Exporter les résultats")

col_e1, col_e2, col_e3 = st.columns(3)

with col_e1:
    if mode == "ecg":
        df_sig = pd.DataFrame({
            "temps_s": times[:len(hr_cleaned)],
            "signal_nettoye": hr_cleaned
        })
        st.download_button(
            "Signal nettoyé (CSV)",
            df_sig.to_csv(index=False).encode("utf-8"),
            "ecg_signal_nettoye.csv", "text/csv",
            use_container_width=True
        )
    else:
        st.download_button(
            "Signal brut (CSV)",
            pd.DataFrame({"temps_s": times, "valeur": signal}).to_csv(index=False).encode("utf-8"),
            "ecg_signal_brut.csv", "text/csv",
            use_container_width=True
        )

with col_e2:
    df_rr = pd.DataFrame({
        "battement": range(1, len(rr_ms) + 1),
        "rr_ms": np.round(rr_ms, 2)
    })
    st.download_button(
        "Intervalles RR (CSV)",
        df_rr.to_csv(index=False).encode("utf-8"),
        "ecg_intervalles_rr.csv", "text/csv",
        use_container_width=True
    )

with col_e3:
    st.download_button(
        "Tableau HRV (CSV)",
        hrv_df.to_csv(index=False).encode("utf-8"),
        "ecg_hrv.csv", "text/csv",
        use_container_width=True
    )