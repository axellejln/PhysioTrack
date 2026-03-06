import neurokit2 as nk

def analyze_eda(eda_signal, sampling_rate=1000):
    """
    Décomposition tonic/phasic avec NeuroKit2.
    """
    signals, info = nk.eda_process(eda_signal, sampling_rate=sampling_rate)
    return signals