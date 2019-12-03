import numpy as np
from scipy import signal
from scipy.signal import butter, lfilter


def createSpec(signals, sr, n_channels=22):
    # Reference: https://github.com/MesSem/CNNs-on-CHB-MIT, DataSetToSpectrogram

    n_channels = min(n_channels, 22)

    for channel in range(n_channels):
        y = signals[channel]

        Pxx = signal.spectrogram(y, nfft=sr, fs=sr, return_onesided=True, noverlap=128)[2]
        Pxx = np.delete(Pxx, np.s_[117:123 + 1], axis=0)
        Pxx = np.delete(Pxx, np.s_[57:63 + 1], axis=0)
        Pxx = np.delete(Pxx, 0, axis=0)
        # Pxx = Pxx + 0.000001

        # result = ((10 * np.log10(Pxx).T - (10 * np.log10(Pxx)).T.mean(axis=0)) / (10 * np.log10(Pxx)).T.std(axis=0))
        result = Pxx.T

        if channel == 0:
            spect = np.zeros((n_channels, *result.shape))

        result = np.nan_to_num(result)

        spect[channel] = result

    return spect


# Filtro taglia banda
def butter_bandstop_filter(data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    i, u = butter(order, [low, high], btype='bandstop')
    y = lfilter(i, u, data)
    return y

# Filtro taglia banda, passa alta
def butter_highpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = lfilter(b, a, data)
    return y
