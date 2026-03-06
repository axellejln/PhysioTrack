import numpy as np
import matplotlib.pyplot as plt
from src.ecg.loader import load_hr
from src.ecg.visualization import plot_hr


hr_values, times = load_hr("data/raw/ecg/HR.csv")


fig = plot_hr(times, hr_values)
plt.show()


print("10 premières valeurs HR :", hr_values[:10])
print("Signal HR affiché avec succès !")