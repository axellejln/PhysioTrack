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
    clean_hr, detect_r_peaks, compute_fft,
    compute_band_energy, compute_hrv_advanced
)
from src.ecg.visualization import (
    plot_ecg_raw, plot_ecg_cleaned, plot_rr_intervals,
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

# import selon le mode choisi
signal = None
sfreq  = None
times  = None
mode   = None

# ECG brut
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

# IBI 
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
            st.success("Fichier IBI chargé avec succès !")
            c1, c2, c3 = st.columns(3)
            c1.metric("Intervalles détectés", len(ibi_values))
            c2.metric("Durée totale", f"{ibi_times[-1]:.1f} s")
            c3.metric("IBI moyen", f"{np.mean(ibi_values)*1000:.0f} ms")

            st.divider()

            # Reconstruction des positions des pics R depuis les IBI
            # Pic R[n] = somme cumulative des IBI jusqu'a n
            r_peaks_times = np.cumsum(ibi_values)  # en secondes

            rr_ms_ibi = ibi_values * 1000
            bpm_reconstructed = 60.0 / ibi_values

            # Vue globale 
            st.subheader("Vue globale")
            col_ibi1, col_ibi2 = st.columns(2)

            with col_ibi1:
                st.caption("**Intervalles IBI (ms)**")
                fig_ibi, ax = plt.subplots(figsize=(5, 3))
                marker = "o" if len(ibi_times) < 200 else None
                ax.plot(ibi_times, rr_ms_ibi, color="crimson",
                        linewidth=0.8, marker=marker, markersize=3)
                ax.axhline(np.mean(rr_ms_ibi), color="steelblue", linestyle="--",
                           linewidth=1, label=f"Moy. {np.mean(rr_ms_ibi):.0f} ms")
                ax.set_xlabel("Temps (s)", fontsize=8)
                ax.set_ylabel("IBI (ms)", fontsize=8)
                ax.set_title("Intervalles IBI", fontsize=9)
                ax.tick_params(labelsize=7)
                ax.legend(fontsize=7)
                plt.tight_layout()
                st.pyplot(fig_ibi)

            with col_ibi2:
                st.caption("**Rythme cardiaque (BPM)**")
                fig_bpm, ax = plt.subplots(figsize=(5, 3))
                ax.plot(r_peaks_times, bpm_reconstructed,
                        color="steelblue", linewidth=0.8)
                ax.axhline(np.mean(bpm_reconstructed), color="crimson",
                           linestyle="--", linewidth=1,
                           label=f"Moy. {np.mean(bpm_reconstructed):.0f} BPM")
                ax.set_xlabel("Temps (s)", fontsize=8)
                ax.set_ylabel("BPM", fontsize=8)
                ax.set_title("HR reconstruit", fontsize=9)
                ax.tick_params(labelsize=7)
                ax.legend(fontsize=7)
                plt.tight_layout()
                st.pyplot(fig_bpm)

            # Zoom pics R sur segment 
            st.subheader("Pics R : zoom sur un segment")
            st.caption("Sélectionnez une fenêtre courte pour visualiser les pics R individuels.")

            duree_totale = float(r_peaks_times[-1])
            col_z1, col_z2 = st.columns(2)
            with col_z1:
                seg_start = st.slider("Début (s)", 0.0, duree_totale - 1.0,
                                      0.0, step=1.0, key="ibi_seg_start")
            with col_z2:
                seg_end = st.slider("Fin (s)", 1.0, duree_totale,
                                    min(30.0, duree_totale), step=1.0, key="ibi_seg_end")

            # Filtrer les pics R dans la fenêtre
            mask_seg = (r_peaks_times >= seg_start) & (r_peaks_times <= seg_end)
            peaks_seg = r_peaks_times[mask_seg]
            bpm_seg   = bpm_reconstructed[mask_seg]

            fig_seg, ax = plt.subplots(figsize=(10, 3))
            # Ligne BPM reconstruit sur le segment
            ax.plot(peaks_seg, bpm_seg, color="steelblue",
                    linewidth=1, alpha=0.5)
            # Pics R comme barres verticales
            for t in peaks_seg:
                ax.axvline(t, color="crimson", linewidth=0.8, alpha=0.7)
            ax.set_xlabel("Temps (s)")
            ax.set_ylabel("BPM")
            ax.set_title(f"Pics R sur [{seg_start:.0f}s – {seg_end:.0f}s]  "
                         f"({len(peaks_seg)} battements)")
            plt.tight_layout()
            st.pyplot(fig_seg)

            # HRV depuis IBI directement
            st.subheader("HRV")
            hrv = {
                "Moyenne RR (ms)": round(float(np.mean(rr_ms_ibi)), 1),
                "SDNN (ms)":       round(float(np.std(rr_ms_ibi, ddof=1)), 1),
                "RMSSD (ms)":      round(float(np.sqrt(np.mean(np.diff(rr_ms_ibi)**2))), 1),
                "pNN50 (%)":       round(float(np.sum(np.abs(np.diff(rr_ms_ibi)) > 50) / max(len(rr_ms_ibi)-1,1) * 100), 1),
                "HR moyen (BPM)":  round(60000.0 / float(np.mean(rr_ms_ibi)), 1),
                "HR min (BPM)":    round(60000.0 / float(np.max(rr_ms_ibi)), 1),
                "HR max (BPM)":    round(60000.0 / float(np.min(rr_ms_ibi)), 1),
            }
            hrv_df = pd.DataFrame(list(hrv.items()), columns=["Indicateur", "Valeur"])
            st.dataframe(hrv_df, use_container_width=True, hide_index=True)

            # Poincaré IBI
            st.subheader("Diagramme de Poincaré")
            fig_pc = plot_poincare(rr_ms_ibi)
            st.pyplot(fig_pc)

            # Export HRV
            st.divider()
            st.subheader("Exporter")
            st.download_button(
                "Tableau HRV (CSV)",
                hrv_df.to_csv(index=False).encode("utf-8"),
                "ecg_hrv_ibi.csv", "text/csv",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"Erreur : {e}")

# HR 
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

            # Filtrer les valeurs aberrantes (BPM valide : 20-250)
            valid_mask = (hr_values >= 20) & (hr_values <= 250)
            n_invalides = np.sum(~valid_mask)
            hr_times  = hr_times[valid_mask]
            hr_values = hr_values[valid_mask]

            signal = hr_values
            times  = hr_times
            sfreq  = 1.0 / np.mean(np.diff(hr_times)) if len(hr_times) > 1 else 1.0
            st.success("Fichier HR chargé avec succès !")
            if n_invalides > 0:
                st.warning(f"⚠️ {n_invalides} valeur(s) aberrante(s) retirée(s) "
                           f"(BPM hors de [20–250]).")
            c1, c2, c3 = st.columns(3)
            c1.metric("Durée", f"{hr_times[-1]:.1f} s")
            c2.metric("HR moyen", f"{np.mean(hr_values):.0f} BPM")
            c3.metric("Nb de points", len(hr_values))
        except Exception as e:
            st.error(f"Erreur : {e}")

# Affichage du signal brut (ECG brut ou HR) + bouton d'analyse
if signal is not None and mode in ["ecg", "hr"]:
    st.divider()

    # Signal brut
    st.subheader("Signal brut")
    if mode == "hr":
        fig_raw = plot_hr(times, signal)
    else:
        fig_raw = plot_ecg_raw(times, signal)
    st.pyplot(fig_raw)

    st.divider()
    st.subheader("Nettoyage et détection des pics R")

    if st.button("Analyser le signal"):
        try:
            if mode == "hr":
                # Signal HR en BPM : pas de nettoyage ECG possible
                # On travaille directement sur les valeurs BPM
                hr_cleaned = signal.copy()
                # Intervalles RR depuis BPM : RR(ms) = 60000 / BPM
                rr_ms = 60000.0 / signal
                # On fabrique des rr_intervals en samples fictifs pour HRV
                rr_intervals = rr_ms  # deja en ms, on adapte compute_hrv_advanced
                r_peaks = np.array([])  # pas de pics R detectable sur signal HR
                freqs, fft_vals = compute_fft(hr_cleaned, sfreq=sfreq)
                band_energy = compute_band_energy(freqs, fft_vals)
                # HRV depuis RR en ms directement (sfreq=1000 pour convertir ms->ms)
                hrv = compute_hrv_advanced(rr_ms * 1.0, sfreq=1000.0)
            else:
                # Signal ECG brut : nettoyage + detection pics R
                hr_cleaned = clean_hr(signal, sfreq=int(sfreq))
                r_peaks, rr_intervals = detect_r_peaks(hr_cleaned, sfreq=int(sfreq))
                rr_ms = rr_intervals / sfreq * 1000
                freqs, fft_vals = compute_fft(hr_cleaned, sfreq=sfreq)
                band_energy = compute_band_energy(freqs, fft_vals)
                hrv = compute_hrv_advanced(rr_intervals, sfreq=sfreq)

            st.session_state["ecg_results"] = {
                "signal":      signal,
                "times":       times,
                "sfreq":       sfreq,
                "hr_cleaned":  hr_cleaned,
                "r_peaks":     r_peaks,
                "rr_ms":       rr_ms,
                "freqs":       freqs,
                "fft_vals":    fft_vals,
                "band_energy": band_energy,
                "hrv":         hrv,
                "mode":        mode,
            }
            # Pour HR/IBI : démarrer le zoom à 0.003 Hz (début VLF)
            # pour éviter l'affichage du pic basse fréquence non informatif
            fft_zoom_default_min = 0.003 if mode == "hr" else 0.0
            st.session_state["_ecg_fft_zoom_min"] = fft_zoom_default_min
            st.session_state["_ecg_fft_zoom_max"] = float(freqs[-1])

        except Exception as e:
            st.error(f"Erreur lors de l'analyse : {e}")

## AFICHAGE DES RÉSULTATS
if "ecg_results" not in st.session_state:
    st.stop()

r          = st.session_state["ecg_results"]
signal     = r["signal"]
times      = r["times"]
sfreq      = r["sfreq"]
hr_cleaned = r["hr_cleaned"]
r_peaks    = r["r_peaks"]
rr_ms      = r["rr_ms"]
freqs      = r["freqs"]
fft_vals   = r["fft_vals"]
band_energy= r["band_energy"]
hrv        = r["hrv"]
mode       = r["mode"]

# Signal brut vs nettoyé côte à côte
st.subheader("Signal brut  vs  Signal nettoyé")
col_s1, col_s2 = st.columns(2)

with col_s1:
    st.caption("**Signal brut**")
    fig_b, ax = plt.subplots(figsize=(5, 3))
    ax.plot(times, signal, linewidth=0.6,
            color="crimson" if mode == "ecg" else "steelblue")
    ax.set_xlabel("Temps (s)", fontsize=8)
    ax.set_ylabel("Amplitude", fontsize=8)
    ax.set_title("Signal brut", fontsize=9)
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    st.pyplot(fig_b)

with col_s2:
    if mode == "hr":
        # Signal HR : afficher les RR en BPM
        st.caption("**Intervalles RR (depuis BPM)**")
        fig_c, ax = plt.subplots(figsize=(5, 3))
        ax.plot(times, rr_ms, linewidth=0.6, color="steelblue")
        ax.set_xlabel("Temps (s)", fontsize=8)
        ax.set_ylabel("RR (ms)", fontsize=8)
        ax.set_title("RR depuis BPM", fontsize=9)
        ax.tick_params(labelsize=7)
        plt.tight_layout()
        st.pyplot(fig_c)
    else:
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

# Intervalles RR + Poincaré côte à côte 
st.subheader("Intervalles RR")
col_rr1, col_rr2 = st.columns([3, 2])

with col_rr1:
    fig_rr = plot_rr_intervals(rr_ms)
    st.pyplot(fig_rr)

with col_rr2:
    fig_pc = plot_poincare(rr_ms)
    st.pyplot(fig_pc)

st.divider()

# HRV + tableau côte à côte
st.subheader("Variabilité de la fréquence cardiaque (HRV)")

# Métriques principales en haut
c1, c2, c3, c4 = st.columns(4)
c1.metric("HR moyen",    f"{hrv['HR moyen (BPM)']} BPM")
c2.metric("SDNN",        f"{hrv['SDNN (ms)']} ms")
c3.metric("RMSSD",       f"{hrv['RMSSD (ms)']} ms")
c4.metric("pNN50",       f"{hrv['pNN50 (%)']} %")

# Tableau complet
hrv_df = pd.DataFrame(list(hrv.items()), columns=["Indicateur", "Valeur"])
st.dataframe(hrv_df, use_container_width=True, hide_index=True)

st.divider()

# FFT avec zoom 
st.subheader("FFT : Spectre de fréquences")

col_z1, col_z2, col_z3 = st.columns([2, 2, 1])
with col_z1:
    fft_zoom_min = st.number_input(
        "Zoom : fréq. min (Hz)",
        min_value=0.0, max_value=float(freqs[-1]),
        value=float(st.session_state.get("_ecg_fft_zoom_min", 0.0)),
        step=0.01, format="%.3f"
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
        st.session_state["_ecg_fft_zoom_min"] = 0.0
        st.session_state["_ecg_fft_zoom_max"] = float(freqs[-1])
        st.rerun()

st.session_state["_ecg_fft_zoom_min"] = fft_zoom_min
st.session_state["_ecg_fft_zoom_max"] = fft_zoom_max

fig_fft = plot_fft(freqs, fft_vals, zoom_min=fft_zoom_min, zoom_max=fft_zoom_max)
st.pyplot(fig_fft)

st.divider()

# Énergie par bande + tableau côte à côte
st.subheader("Énergie par bande (VLF / LF / HF)")

col_b1, col_b2 = st.columns([2, 1])
with col_b1:
    fig_band = plot_band_energy(band_energy)
    st.pyplot(fig_band)
with col_b2:
    df_band = pd.DataFrame(list(band_energy.items()), columns=["Bande", "Énergie"])
    df_band["Énergie"] = df_band["Énergie"].round(4)
    st.dataframe(df_band, use_container_width=True, hide_index=True)

st.divider()

# Export CSV
st.header("Exporter les résultats")

col_e1, col_e2, col_e3 = st.columns(3)

with col_e1:
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

with col_e2:
    df_rr = pd.DataFrame({
        "battement": range(1, len(rr_ms) + 1),
        "rr_ms": rr_ms.round(2)
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