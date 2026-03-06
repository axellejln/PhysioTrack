import pandas as pd
import numpy as np

def load_hr(file_path):
    df = pd.read_csv(file_path, header=None)

    # Ignorer la première ligne (timestamp ou valeur non HR)
    hr_values = df.iloc[1:, 0].values.astype(float)

    # Génère un vecteur temps en secondes, 1 valeur par seconde
    times = np.arange(len(hr_values))

    return hr_values, times