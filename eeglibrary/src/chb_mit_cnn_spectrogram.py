import pyedflib
import numpy as np
from scipy import signal
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os


def createSpec(signals, sr):
    # Reference: https://github.com/MesSem/CNNs-on-CHB-MIT, DataSetToSpectrogram

    n_channels = 22

    for channel in range(n_channels):
        data = signals[channel]
        fs = sr
        lowcut = 117
        highcut = 123

        y = butter_bandstop_filter(data, lowcut, highcut, fs, order=6)
        lowcut = 57
        highcut = 63
        y = butter_bandstop_filter(y, lowcut, highcut, fs, order=6)

        cutoff = 1
        y = butter_highpass_filter(y, cutoff, fs, order=6)

        Pxx = signal.spectrogram(y, nfft=sr, fs=sr, return_onesided=True, noverlap=128)[2]
        Pxx = np.delete(Pxx, np.s_[117:123 + 1], axis=0)
        Pxx = np.delete(Pxx, np.s_[57:63 + 1], axis=0)
        Pxx = np.delete(Pxx, 0, axis=0)

        result = (10 * np.log10(np.transpose(Pxx)) - (10 * np.log10(np.transpose(Pxx))).min()) / (
                    10 * np.log10(np.transpose(Pxx))).ptp()
        if channel == 0:
            spect = np.zeros((n_channels, *result.shape))

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
