import streamlit as st
import tempfile
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.eeg.loader import load_eeg_generic
from src.ecg.loader import load_ecg, load_ibi, load_hr
from src.eda.loader import load_eda

st.set_page_config(page_title="Comparaison - PhysioTrack", layout="centered")
st.title("Comparaison multi-signaux")
st.caption("Visualisez plusieurs signaux physiologiques sur une timeline commune.")

# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────
def _read_unix_timestamp(filepath):
    """Lit le timestamp Unix sur la premiere ligne d'un fichier E4 si present."""
    try:
        with open(filepath, "r") as f:
            first_line = f.readline().strip().split(",")[0]
            val = float(first_line)
            if val > 1e9:  # timestamp Unix
                return val
    except Exception:
        pass
    return None


def load_signal_generic(uploaded_file, signal_type, sfreq_manual=None):
    """Charge n'importe quel signal.
    Retourne (signal, sfreq, times_recales, unix_timestamp_or_None).
    times est toujours recale a 0. Le timestamp Unix sert a la synchronisation."""
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Lire le timestamp Unix si present (format E4)
    unix_ts = _read_unix_timestamp(tmp_path)

    if signal_type == "EEG":
        raw = load_eeg_generic(tmp_path, sfreq=sfreq_manual)
        data = raw.get_data()[0]
        sfreq = raw.info["sfreq"]
        times = raw.times

    elif signal_type == "ECG":
        data, sfreq, times = load_ecg(tmp_path, sfreq=sfreq_manual)

    elif signal_type == "EDA":
        data, sfreq, times = load_eda(tmp_path, sfreq=sfreq_manual)

    elif signal_type == "HR":
        hr_times, data = load_hr(tmp_path)
        sfreq = 1.0 / np.mean(np.diff(hr_times)) if len(hr_times) > 1 else 1.0
        times = hr_times

    elif signal_type == "IBI":
        times, data = load_ibi(tmp_path)

    else:
        raise ValueError(f"Type inconnu : {signal_type}")

    # Recaler les temps a 0
    times = times - times[0]

    return data, float(sfreq) if signal_type != "IBI" else 1.0, times, unix_ts


def normalize(x):
    xmin, xmax = np.min(x), np.max(x)
    if xmax == xmin:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)


COLORS = ["steelblue", "crimson", "seagreen"]
SIGNAL_TYPES = ["EEG", "ECG", "EDA", "HR", "IBI"]

# IMPORT DES SIGNAUX
st.header("Importer les signaux")
st.caption(
    "Importez jusqu'à 3 signaux. Pour chaque signal, indiquez son type "
    "et son heure de début pour synchroniser la timeline."
)

n_signals = st.radio("Nombre de signaux à comparer", [2, 3], horizontal=True)

signals_raw = []  # données brutes avant synchronisation

for i in range(n_signals):
    st.subheader(f"Signal {i+1}")
    col_a, col_b = st.columns([2, 1])

    with col_a:
        uploaded = st.file_uploader(
            f"Fichier signal {i+1}",
            type=["edf", "bdf", "csv", "txt"],
            key=f"comp_file_{i}"
        )
    with col_b:
        sig_type = st.selectbox(
            "Type",
            SIGNAL_TYPES,
            key=f"comp_type_{i}"
        )

    label = st.text_input(f"Nom du signal {i+1}", value=sig_type, key=f"comp_label_{i}")

    if uploaded is not None:
        try:
            sig, sfreq, times, unix_ts = load_signal_generic(uploaded, sig_type)
            signals_raw.append({
                "label":    label,
                "type":     sig_type,
                "signal":   sig,
                "sfreq":    sfreq,
                "times":    times,
                "unix_ts":  unix_ts,
                "color":    COLORS[i],
            })
            ts_info = f" | timestamp : {unix_ts:.0f}" if unix_ts else " | pas de timestamp"
            st.success(f"✅ Chargé : {len(sig)} points, {times[-1]:.1f}s, {sfreq:.1f} Hz{ts_info}")
        except Exception as e:
            st.error(f"Erreur : {e}")

    if i < n_signals - 1:
        st.divider()

## Synchronisation des signaux et visualisation sur timeline commune
if len(signals_raw) == 0:
    st.stop()

# Synchronisation automatique par timestamp Unix
signals_data = []
unix_timestamps = [d["unix_ts"] for d in signals_raw if d["unix_ts"] is not None]

if len(unix_timestamps) == len(signals_raw) and len(unix_timestamps) > 0:
    # Tous les signaux ont un timestamp → décalage automatique
    ts_min = min(unix_timestamps)
    for d in signals_raw:
        t_offset = d["unix_ts"] - ts_min
        signals_data.append({**d, "times": d["times"] + t_offset, "t_start": t_offset})
    offsets = ", ".join([f"{d['label']} : +{d['t_start']:.1f}s" for d in signals_data])
    st.info(f"🔄 Synchronisation automatique : décalages détectés : {offsets}")
else:
    # Pas de timestamp → affichage sans synchronisation
    for d in signals_raw:
        signals_data.append({**d, "t_start": 0.0})
    if len(signals_raw) > 1:
        st.warning("⚠️ Certains fichiers n'ont pas de timestamp : signaux affichés sans synchronisation.")

st.divider()
st.header("Visualisation sur timeline commune")

t_global_max = max(d["times"][-1] for d in signals_data)

# Réinitialiser le zoom si les signaux changent (nouvelle durée max)
if st.session_state.get("_comp_t_max") != t_global_max:
    st.session_state["_comp_zoom_start"] = 0.0
    st.session_state["_comp_zoom_end"]   = float(t_global_max)
    st.session_state["_comp_t_max"]      = float(t_global_max)

# Zoom timeline
col_z1, col_z2, col_z3 = st.columns([2, 2, 1])
with col_z1:
    view_start = st.number_input(
        "Zoom : début (s)",
        min_value=0.0,
        max_value=float(t_global_max),
        value=float(st.session_state.get("_comp_zoom_start", 0.0)),
        step=1.0
    )
with col_z2:
    view_end = st.number_input(
        "Zoom : fin (s)",
        min_value=0.0,
        max_value=float(t_global_max),
        value=float(st.session_state.get("_comp_zoom_end", t_global_max)),
        step=1.0
    )
with col_z3:
    st.write("")
    st.write("")
    if st.button("🔄 Reset"):
        st.session_state["_comp_zoom_start"] = 0.0
        st.session_state["_comp_zoom_end"]   = float(t_global_max)
        st.rerun()

st.session_state["_comp_zoom_start"] = view_start
st.session_state["_comp_zoom_end"]   = view_end

# Option normalisation
normalize_signals = st.checkbox(
    "Normaliser les signaux [0–1]",
    value=True,
    help="Permet de comparer des signaux d'amplitudes très différentes sur le même axe"
)

# Graphe multi-signaux 
n = len(signals_data)
fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)
if n == 1:
    axes = [axes]

for i, d in enumerate(signals_data):
    mask = (d["times"] >= view_start) & (d["times"] <= view_end)
    t_plot = d["times"][mask]
    s_plot = d["signal"][mask]

    if len(t_plot) == 0:
        axes[i].text(0.5, 0.5, "Aucune donnée dans cette fenêtre",
                     ha="center", va="center", transform=axes[i].transAxes)
        continue

    y_plot = normalize(s_plot) if normalize_signals else s_plot
    axes[i].plot(t_plot, y_plot, color=d["color"], linewidth=0.8)
    axes[i].set_ylabel("Norm. [0-1]" if normalize_signals else "Amplitude", fontsize=8)
    axes[i].set_title(f"{d['label']}  ({d['type']}) : {d['sfreq']:.1f} Hz", fontsize=9)
    axes[i].tick_params(labelsize=7)

    # Barre verticale pour marquer le début de l'enregistrement
    if d["t_start"] >= view_start and d["t_start"] <= view_end:
        axes[i].axvline(d["t_start"], color="orange", linewidth=1,
                        linestyle="--", alpha=0.7, label="Début enregistrement")
        axes[i].legend(fontsize=7, loc="upper right")

axes[-1].set_xlabel("Temps (s)", fontsize=9)
plt.tight_layout()
st.pyplot(fig)

st.divider()

## STATS
st.header("Statistiques comparatives")

stats_rows = []
for d in signals_data:
    # Stats sur la fenêtre de zoom
    mask = (d["times"] >= view_start) & (d["times"] <= view_end)
    s = d["signal"][mask]
    if len(s) == 0:
        continue
    stats_rows.append({
        "Signal":        d["label"],
        "Type":          d["type"],
        "Début (s)":     f"{d['t_start']:.1f}",
        "Durée (s)":     f"{d['times'][-1] - d['times'][0]:.1f}",
        "Moyenne":       f"{np.mean(s):.4f}",
        "Écart-type":    f"{np.std(s):.4f}",
        "Min":           f"{np.min(s):.4f}",
        "Max":           f"{np.max(s):.4f}",
    })

if stats_rows:
    df_stats = pd.DataFrame(stats_rows)
    st.dataframe(df_stats, use_container_width=True, hide_index=True)

    st.divider()

    #  Export
    st.header("Exporter")
    st.download_button(
        "Statistiques comparatives (CSV)",
        df_stats.to_csv(index=False).encode("utf-8"),
        "comparaison_stats.csv", "text/csv",
        use_container_width=True
    )