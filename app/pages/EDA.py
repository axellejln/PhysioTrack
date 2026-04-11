import streamlit as st
import tempfile
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.eda.loader import load_eda
from src.eda.analysis import analyze_eda
from src.eda.visualization import plot_eda

st.set_page_config(page_title="EDA - PhysioTrack", layout="centered")
st.title("Analyse EDA")

## IMPORTER UN SIGNAL
with st.expander("Format de fichier attendu : cliquez pour en savoir plus"):
    st.markdown("""
CSV avec **1 colonne signal EDA** (en µS), optionnellement précédée d'une colonne temps.
Si pas de colonne temps → sfreq à renseigner (typiquement **4 Hz** pour Empatica E4).
Format **Empatica E4** détecté automatiquement (timestamp Unix ligne 1, sfreq ligne 2).
```
1544027337.000000    # format E4  ou avec colonne temps :
4.000000             # sfreq (Hz)
0.005125             # valeurs EDA (µS)
0.020501
```
```
temps,EDA            # format CSV standard
0.00,0.0051
0.25,0.0205
0.50,0.0218
```
    """)

uploaded = st.file_uploader(
    "Importer un fichier EDA",
    type=["csv", "txt"],
    help="Le fichier doit contenir une colonne signal EDA (et optionnellement une colonne temps)"
)

if uploaded is None:
    st.stop()

ext = os.path.splitext(uploaded.name)[1].lower()
with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
    tmp.write(uploaded.read())
    tmp_path = tmp.name

try:
    df_preview = pd.read_csv(tmp_path)
except Exception:
    df_preview = pd.read_csv(tmp_path, header=None, skiprows=1)
df_preview = df_preview.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
first_col = df_preview.iloc[:, 0].values

sfreq_input = None
if np.all(np.diff(first_col) > 0):
    st.info("Colonne temps détectée : fréquence d'échantillonnage calculée automatiquement.")
else:
    sfreq_input = st.number_input("Fréquence d'échantillonnage (Hz)", min_value=1.0, value=4.0, step=1.0)

try:
    signal, sfreq, times = load_eda(tmp_path, sfreq=sfreq_input)
    st.success("Fichier EDA chargé avec succès !")
except Exception as e:
    st.error(f"Erreur lors du chargement : {e}")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Fréquence d'échantillonnage", f"{sfreq:.1f} Hz")
col2.metric("Durée", f"{times[-1]:.1f} s")
col3.metric("Nb d'échantillons", len(signal))

st.divider()

## AFFICHAGE DU SIGNAL BRUT AVEC ZOOM
st.subheader("Signal EDA brut")

col_z1, col_z2, col_z3 = st.columns([2, 2, 1])
with col_z1:
    zoom_start = st.number_input(
        "Zoom : début (s)",
        min_value=0.0, max_value=float(times[-1]),
        value=float(st.session_state.get("_eda_zoom_start", 0.0)),
        step=1.0
    )
with col_z2:
    zoom_end = st.number_input(
        "Zoom : fin (s)",
        min_value=0.0, max_value=float(times[-1]),
        value=float(st.session_state.get("_eda_zoom_end", float(times[-1]))),
        step=1.0
    )
with col_z3:
    st.write("")
    st.write("")
    if st.button("🔄 Reset"):
        st.session_state["_eda_zoom_start"] = 0.0
        st.session_state["_eda_zoom_end"] = float(times[-1])
        st.rerun()

st.session_state["_eda_zoom_start"] = zoom_start
st.session_state["_eda_zoom_end"]   = zoom_end

zoom_mask = (times >= zoom_start) & (times <= zoom_end)
fig_raw, ax = plt.subplots(figsize=(10, 3))
ax.plot(times[zoom_mask], signal[zoom_mask], color="steelblue", linewidth=0.8)
ax.set_xlabel("Temps (s)")
ax.set_ylabel("Amplitude (µS)")
ax.set_title(f"Signal EDA brut  [{zoom_start:.0f}s – {zoom_end:.0f}s]")
plt.tight_layout()
st.pyplot(fig_raw)

st.divider()

# LISSAGE
# st.subheader("Filtrage / lissage")

# window_sec = st.slider(
#     "Fenêtre de lissage (secondes)",
#     min_value=0.1, max_value=10.0, value=1.0, step=0.1,
#     help="Lissage par moyenne glissante"
# )
# window_samples = max(1, int(window_sec * sfreq))
# signal_smooth = np.convolve(signal, np.ones(window_samples) / window_samples, mode="same")

# col_l1, col_l2 = st.columns(2)
# with col_l1:
#     st.caption("**Signal brut**")
#     fig_brut, ax = plt.subplots(figsize=(5, 3))
#     ax.plot(times[zoom_mask], signal[zoom_mask], color="gray", linewidth=0.8)
#     ax.set_xlabel("Temps (s)", fontsize=8)
#     ax.set_ylabel("Amplitude (µS)", fontsize=8)
#     ax.set_title("Brut", fontsize=9)
#     ax.tick_params(labelsize=7)
#     plt.tight_layout()
#     st.pyplot(fig_brut)

# with col_l2:
#     st.caption(f"**Signal lissé ({window_sec:.1f}s)**")
#     fig_lis, ax = plt.subplots(figsize=(5, 3))
#     ax.plot(times[zoom_mask], signal_smooth[zoom_mask], color="steelblue", linewidth=0.8)
#     ax.set_xlabel("Temps (s)", fontsize=8)
#     ax.set_ylabel("Amplitude (µS)", fontsize=8)
#     ax.set_title(f"Lissé ({window_sec:.1f}s)", fontsize=9)
#     ax.tick_params(labelsize=7)
#     plt.tight_layout()
#     st.pyplot(fig_lis)

# st.divider()


## DECOMPOSITION PHASIQUE / TONIQUE
st.subheader("Décomposition phasique / tonique")
st.caption("Utilise NeuroKit2 pour séparer la composante lente (tonique) de la composante rapide (phasique).")

if st.button("Décomposer le signal EDA"):
    try:
        signals_nk = analyze_eda(signal, sampling_rate=int(sfreq))
        tonic  = signals_nk["EDA_Tonic"].values
        phasic = signals_nk["EDA_Phasic"].values
        st.session_state["eda_results"] = {
            "signal": signal,
            "times":  times,
            "sfreq":  sfreq,
            "tonic":  tonic,
            "phasic": phasic,
        }
        st.session_state["_eda_peak_height"] = float(np.max(phasic) * 0.1) if np.max(phasic) > 0 else 0.05
        st.session_state["_eda_peak_dist"]   = 1.0
    except Exception as e:
        st.error(f"Erreur lors de la décomposition : {e}")
        st.info("Vérifiez que NeuroKit2 est bien installé et que le signal est valide.")

if "eda_results" not in st.session_state:
    st.stop()

## AFFICHAGE
r      = st.session_state["eda_results"]
signal = r["signal"]
times  = r["times"]
sfreq  = r["sfreq"]
tonic  = r["tonic"]
phasic = r["phasic"]

# Graphe décomposition
fig_decomp, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
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
st.pyplot(fig_decomp)

# # Avertissement si valeurs phasiques négatives
# if np.min(phasic) < 0:
#     st.info(
#         "ℹ️ La composante phasique contient des valeurs négatives : "
#         "cela est normal et dû à la méthode de décomposition de NeuroKit2 sur un signal bruité. "
#         "Seuls les **pics positifs** sont physiologiquement interprétables (réponses SCR)."
#     )

st.divider()

## DETECTION DES PICS PHASIQUES (SCR)
st.subheader("Détection des pics phasiques (SCR)")
st.caption("Ajustez les paramètres : les pics se mettent à jour sans relancer la décomposition.")

phasic_max = float(np.max(phasic)) if np.max(phasic) > 0 else 1.0

col_p1, col_p2 = st.columns(2)
with col_p1:
    min_height = st.slider(
        "Amplitude minimale des pics (µS)",
        min_value=0.0, max_value=phasic_max,
        value=float(st.session_state.get("_eda_peak_height", phasic_max * 0.1)),
        step=phasic_max / 100,
        format="%.4f"
    )
with col_p2:
    min_dist_sec = st.slider(
        "Distance minimale entre pics (s)",
        0.5, 30.0,
        value=float(st.session_state.get("_eda_peak_dist", 1.0)),
        step=0.5
    )

st.session_state["_eda_peak_height"] = min_height
st.session_state["_eda_peak_dist"]   = min_dist_sec

min_dist_samples = max(1, int(min_dist_sec * sfreq))
peaks_idx, _ = find_peaks(phasic, height=min_height, distance=min_dist_samples)

# Graphe phasic + pics
fig_peaks, ax = plt.subplots(figsize=(10, 3))
ax.plot(times[:len(phasic)], phasic, color="crimson", linewidth=0.8, label="Phasique")
if len(peaks_idx) > 0:
    ax.plot(times[peaks_idx], phasic[peaks_idx],
            "v", color="navy", markersize=7, label=f"Pics ({len(peaks_idx)})")
ax.set_xlabel("Temps (s)")
ax.set_ylabel("Amplitude (µS)")
ax.set_title("Pics phasiques détectés (SCR)")
ax.legend()
plt.tight_layout()
st.pyplot(fig_peaks)

# Stats des pics
if len(peaks_idx) > 0:
    peak_times = times[peaks_idx]
    peak_amps  = phasic[peaks_idx]

    st.subheader("Statistiques des pics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nombre de pics", len(peaks_idx))
    c2.metric("Amplitude moyenne (µS)", f"{np.mean(peak_amps):.4f}")
    c3.metric("Amplitude max (µS)", f"{np.max(peak_amps):.4f}")
    freq_pics = len(peaks_idx) / (times[-1] - times[0]) * 60
    c4.metric("Fréquence (pics/min)", f"{freq_pics:.1f}")

    peaks_df = pd.DataFrame({
        "Pic n°":        range(1, len(peaks_idx) + 1),
        "Temps (s)":     np.round(peak_times, 2),
        "Amplitude (µS)": np.round(peak_amps, 4)
    })
    st.dataframe(peaks_df, use_container_width=True, hide_index=True)
else:
    st.info("Aucun pic détecté : essayez de diminuer l'amplitude minimale.")
    peaks_df = pd.DataFrame()

st.divider()

## EXPORT
st.header("Exporter les résultats")

col_e1, col_e2 = st.columns(2)

with col_e1:
    df_export = pd.DataFrame({
        "temps_s":      times[:len(tonic)],
        "eda_brut":     signal[:len(tonic)],
        "eda_tonique":  tonic,
        "eda_phasique": phasic,
    })
    st.download_button(
        "Signal décomposé (CSV)",
        df_export.to_csv(index=False).encode("utf-8"),
        "eda_decomposed.csv", "text/csv",
        use_container_width=True
    )

with col_e2:
    if not peaks_df.empty:
        st.download_button(
            "Tableau des pics (CSV)",
            peaks_df.to_csv(index=False).encode("utf-8"),
            "eda_peaks.csv", "text/csv",
            use_container_width=True
        )
    else:
        st.button("Tableau des pics (CSV)", disabled=True, use_container_width=True)