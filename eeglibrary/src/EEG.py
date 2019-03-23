import copy
import pickle
import numpy as np
import scipy


FILE_FORMAT = ['.mat']


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
            eeg = pickle.load(f)
        return eeg

    def __repr__(self):
        self.info()
        return ""

    def to_pkl(self, file_path):
        with open(file_path, mode='wb') as f:
            pickle.dump(self, f)

    def split(self, window_size=0.5, window_stride='same', padding='same') -> list:
        assert float(window_size) != 0.0, 'window_size must be over 0.'

        def validate_values(window_stride, padding):
            if window_stride == 'same' or float(window_stride) == 0.0:
                window_stride = window_size

            n_eeg = (self.len_sec - window_size) // window_stride + 1

            if padding == 'same':
                padding = (self.len_sec - n_eeg * window_stride) // 2
            else:
                n_eeg = (self.len_sec + padding * 2 - window_size) // window_stride

            return int(n_eeg), window_stride, padding

        n_eeg, window_stride, padding = validate_values(window_stride, padding)

        # add padding
        n_channel = len(self.channel_list)
        pad_matrix = np.zeros((n_channel, int(padding * self.sr)))
        if padding:
            padded_waves = np.hstack((pad_matrix, self.values, pad_matrix))

        splitted_eegs = []
        duration = int(window_size * self.sr)
        for i in range(n_eeg):
            eeg = copy.deepcopy(self)
            start_index = int(i * self.sr * window_stride)
            eeg.values = self.values[:, start_index:start_index + duration]
            eeg.len_sec = window_size
            splitted_eegs.append(eeg)

        return splitted_eegs

    def resample(self, num_rsmpl):
        resampled = np.zeros((len(self.channel_list), int(num_rsmpl * self.len_sec)))
        for i in range(len(self.channel_list)):
            resampled[i, :] = scipy.signal.resample(self.values[i], int(num_rsmpl * self.len_sec))
        return resampled
