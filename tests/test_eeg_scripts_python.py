from src.eeg.pipeline import eeg_pipeline

if __name__ == "__main__":
    results = eeg_pipeline("data/raw/eeg/S001R01.edf", channel_index=0, tmin=0, tmax=20)