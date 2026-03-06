import matplotlib.pyplot as plt

def plot_eda(signal, times=None, title="EDA brut"):
    times = times if times is not None else range(len(signal))
    fig, ax = plt.subplots()
    ax.plot(times, signal)
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    return fig