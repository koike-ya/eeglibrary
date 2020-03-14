import numpy as np
import torch
from ml.src.signal_processor import *
from ml.src.preprocessor import preprocess_args, Preprocessor
from eeglibrary.src.chb_mit_cnn_spectrogram import createSpec
from eeglibrary.src.signal_processor import *
from sklearn import preprocessing


def eeg_preprocess_args(parser):
    parser = preprocess_args(parser)

    eeg_prep_parser = parser.add_argument_group("EEG Preprocess options")

    eeg_prep_parser.add_argument('--duration', default=10.0, type=float, help='Duration of one EEG dataset')
    eeg_prep_parser.add_argument('--n-use-eeg', default=1, type=int, help='Number of eeg to use')
    eeg_prep_parser.add_argument('--n-features', type=int, help='Number of features to reshape from 1 channel feature')
    eeg_prep_parser.add_argument('--sample-rate', default='same', help='Sample rate')
    eeg_prep_parser.add_argument('--num-eigenvalue', default=0, type=int,
                                 help='Number of eigen values to use from spectrogram')
    eeg_prep_parser.add_argument('--to-1d', dest='to_1d', action='store_true', help='Preprocess inputs to 1 dimension')

    return parser


class EEGPreprocessor(Preprocessor):
    def __init__(self, eeg_conf, phase, to_1d=False, scaling_axis=None):
        super(EEGPreprocessor, self).__init__(eeg_conf, phase, eeg_conf['sample_rate'])
        self.n_features = eeg_conf['n_features']
        self.to_1d = to_1d
        self.time_corr = True
        self.freq_corr = True
        self.use_eig_values = True
        self.scaling_axis = scaling_axis
        self.reproduce = eeg_conf['reproduce']

    def _calc_correlation(self, matrix):
        if self.scaling_axis:
            matrix = preprocessing.scale(matrix, axis=self.scaling_axis)

        return to_correlation_matrix(matrix)

    def calc_corr_frts(self, eeg, space='time'):
        if space == 'time':
            corr_matrix = self._calc_correlation(eeg.values)
            y = flatten_corr_upper_right(corr_matrix)
        if space == 'freq':
            corr_matrix = self._calc_correlation(np.absolute(np.fft.rfft(eeg.values, axis=1)))
            y = flatten_corr_upper_right(corr_matrix)
        if self.use_eig_values:
            y = np.hstack((y, calc_eigen_values_sorted(corr_matrix)))
        return y

    def preprocess(self, eeg, label=None):

        if self.sr == 'same':
            self.sr = eeg.sr
        elif int(self.sr) != eeg.sr:
            eeg.values = eeg.resample(self.sr)
            eeg.sr = int(self.sr)
        else:
            self.sr = int(self.sr)

        if self.reproduce == 'chbmit-cnn':
            return torch.from_numpy(createSpec(eeg.values, eeg.sr, len(eeg.channel_list)))

        if self.reproduce == 'bonn-rnn':
            n_channel = min(len(eeg.channel_list), 22)
            timesteps = int(eeg.len_sec / 2 * eeg.sr)     # 15 * 256
            feature_dim = n_channel * eeg.values.shape[1] // timesteps     # 22 x (30 x 256) / (15 x 256) = 22 x 2
            return torch.from_numpy(eeg.values[:n_channel, :].reshape((feature_dim, timesteps)).T) # 22 x (30 x 256) -> (22 x 2) x (15 x 256)

        y, label = super().preprocess(eeg.values, label)

        # y = np.clip(y,  -1000, 1000)

        if self.to_1d:
            y = np.array([])
            if self.time_corr:
                y = np.hstack((y, self.calc_corr_frts(eeg, 'time')))
            if self.freq_corr:
                y = np.hstack((y, self.calc_corr_frts(eeg, 'freq')))
                y = torch.from_numpy(y)

        if self.n_features:
            y = y.reshape(self.n_features, -1)

        return y

    def mfcc(self):
        raise NotImplementedError
