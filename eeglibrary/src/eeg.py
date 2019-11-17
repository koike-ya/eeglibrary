import copy
import pickle
import numpy as np
import scipy
from tqdm import tqdm
from collections import OrderedDict
from joblib import Parallel, delayed


FILE_FORMAT = ['.mat', '.pkl']


class EEG:
    """
    要件: - もともとのファイル形式に関係なく、eegに関するデータにアクセスできること
            - 波形データ np.array: values
            - チャンネル名 list(str): channel_list
            - チャンネル数 int: num_channel
            - 波形長 float: len_sec
            - サンプリング周波数 sr: float
            -
            -
            -
    """
    def __init__(self, values, channel_list, len_sec, sr, header=None):
        self.values = values
        self.channel_list = channel_list
        self.len_sec = float(len_sec)
        self.sr = int(sr)
        self.header = header

    def info(self):
        print('Data: \n {}'.format(self.values))
        print('Data Shape: \t {}'.format(self.values.shape))
        print('Data length sec: \t {}'.format(self.len_sec))
        print('Data sampling frequency: \t {}'.format(self.sr))
        print('All channels: \n {}'.format(self.channel_list))
        print('Number of channels: \t {}'.format(len(self.channel_list)))
        # print('Data sequense: \t {}'.format(data['sequence'][0][0]))

    @classmethod
    def load_pkl(cls, file_path):
        with open(file_path, mode='rb') as f:
            eeg_ = pickle.load(f)
        return eeg_

    @classmethod
    def from_edf(cls, edf, verbose=True):
        n = edf.signals_in_file
        signals = np.zeros((n, edf.getNSamples()[0]))
        for i in tqdm(np.arange(n), disable=not verbose):
            try:
                signals[i, :] = edf.readSignal(i)
            except ValueError as e:
                np.delete(signals, i, 0)

        return EEG(signals, edf.getSignalLabels(), edf.getFileDuration(), edf.getSampleFrequencies()[0])

    def __repr__(self):
        self.info()
        return ""

    def to_pkl(self, file_path):
        with open(file_path, mode='wb') as f:
            pickle.dump(self, f)

    def _validate_values(self, window_size, window_stride, padding):
        if window_stride == 'same' or float(window_stride) == 0.0:
            window_stride = window_size

        n_eeg = (self.len_sec - window_size) // window_stride + 1

        if padding == 'same':
            padding = (self.len_sec - n_eeg * window_stride) // 2
        else:
            n_eeg = (self.len_sec + padding * 2 - window_size) // window_stride

        assert self.values.shape[1] >= int(self.len_sec * self.sr)

        return int(n_eeg), window_stride, padding

    def split_and_save(self, window_size=0.5, window_stride='same', padding='same', n_jobs=-1, save_dir='',
                       suffix='') -> list:
        assert float(window_size) != 0.0, 'window_size must be over 0.'

        def split_(j):
            start_index = int(j * self.sr * window_stride)
            eeg = EEG(None, self.channel_list, self.len_sec, self.sr, self.header)
            eeg.values = self.values[:, start_index:start_index + duration]
            assert eeg.values.shape[1] == duration
            assert not np.isnan(np.sum(eeg.values))
            eeg.len_sec = window_size
            filename = f'{start_index}_{start_index + duration}{suffix}.pkl'
            eeg.to_pkl(f'{save_dir}/{filename}')
            return f'{save_dir}/{filename}'

        n_eeg, window_stride, padding = self._validate_values(window_size, window_stride, padding)

        # add padding
        n_channel = len(self.channel_list)
        pad_matrix = np.zeros((n_channel, int(padding * self.sr)))
        if padding:
            padded_waves = np.hstack((pad_matrix, self.values, pad_matrix))

        duration = int(window_size * self.sr)

        # For debugging
        if n_jobs == 1:
            path_list = [split_(i) for i in range(n_eeg)]
        else:
            path_list = Parallel(n_jobs=n_jobs, verbose=1)([delayed(split_)(i) for i in range(n_eeg)])

        return path_list

    def split(self, window_size=0.5, window_stride='same', padding='same', n_jobs=-1) -> list:
        assert float(window_size) != 0.0, 'window_size must be over 0.'

        def split_(j):
            start_index = int(j * self.sr * window_stride)
            eeg = EEG(None, self.channel_list, self.len_sec, self.sr, self.header)
            eeg.values = self.values[:, start_index:start_index + duration]
            assert eeg.values.shape[1] == duration
            assert not np.isnan(np.sum(eeg.values))
            eeg.len_sec = window_size
            return eeg

        n_eeg, window_stride, padding = self._validate_values(window_size, window_stride, padding)

        # add padding
        n_channel = len(self.channel_list)
        pad_matrix = np.zeros((n_channel, int(padding * self.sr)))
        if padding:
            padded_waves = np.hstack((pad_matrix, self.values, pad_matrix))

        duration = int(window_size * self.sr)

        # For debugging
        if n_jobs == 1:
            splitted_eegs = [split_(i) for i in range(n_eeg)]
        else:
            splitted_eegs = Parallel(n_jobs=n_jobs, verbose=1)([delayed(split_)(i) for i in range(n_eeg)])

        return splitted_eegs

    def resample(self, n_resample) -> np.array([]):
        resampled = np.zeros((len(self.channel_list), int(n_resample * self.len_sec)))
        for i in range(len(self.channel_list)):
            try:
                resampled[i, :] = scipy.signal.resample(self.values[i], int(n_resample * self.len_sec))
            except ValueError as e:
                exit(1)
        return resampled


if __name__ == '__main__':
    import pyedflib
    edfreader = pyedflib.EdfReader('/media/tomoya/SSD-PGU3/research/brain/children/YJ0112PQ_1-1.edf')
    eeg = EEG.from_edf(edfreader)
    import matplotlib.pyplot as plt
    import pandas as pd
    # for i in range(44):
    print(pd.DataFrame(eeg.values).std(axis=1))
    # plt.show()
    a = ''
