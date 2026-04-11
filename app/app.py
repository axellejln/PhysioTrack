import streamlit as st

st.set_page_config(
    page_title="PhysioTrack",
    layout="centered"
)

st.title("PhysioTrack")
st.subheader("Interface de traitement de données physiologiques")

st.markdown("""
Bienvenue sur **PhysioTrack** !

Cette application vous permet de charger, visualiser, pré-traiter et analyser
des signaux physiologiques, sans nécessiter de compétences en programmation.

---

### Choisissez un type de signal dans la barre latérale

| Page | Signal | Fonctionnalités |
|------|--------|-----------------|
| EEG | Électroencéphalogramme | Import, filtrage, FFT, spectrogramme, énergie par bande |
| ECG | Électrocardiogramme | Import HR/IBI/ECG brut, pics R, HRV, FFT |
| EDA | Activité électrodermale | Import, filtrage, décomposition phasic/tonic, pics |
| Comparaison | Multi-signaux | Synchronisation, comparaison, statistiques |


""")