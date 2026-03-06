
#filtre passe-bande
def bandpass_filter(raw, l_freq=1.0, h_freq=40.0):
    filtered = raw.copy().filter(l_freq=l_freq, h_freq=h_freq)
    return filtered

# filtre notch pour supprimer le bruit secteur
def notch_filter(raw, freq=50):
    filtered = raw.copy().notch_filter(freqs=freq)
    return filtered


def crop_signal(raw, tmin, tmax):
    cropped = raw.copy().crop(tmin=tmin, tmax=tmax)
    return cropped