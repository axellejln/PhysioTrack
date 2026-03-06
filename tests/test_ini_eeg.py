import streamlit as st
import pandas as pd
import numpy as np
import mne 
import time

st.title("PhysioTrack")
st.write("Bienvenue dans l'interface de traitement de données physiologiques !")

# Simulation du chargement de données
st.header("Chargement des données")
edf_file = "../data/raw/eeg/S001R01.edf"
@st.cache_data
def load_eeg(edf_file):
    raw = mne.io.read_raw_edf(edf_file, preload=True)
    return raw

raw_load_state = st.text('Loading data...')
raw = load_eeg(edf_file)
raw_load_state.text("Done! (using st.cache_data)")

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(raw)
