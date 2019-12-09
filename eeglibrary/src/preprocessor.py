import numpy as np
import torch
from eeglibrary.src.signal_processor import *
from eeglibrary.src.chb_mit_cnn_spectrogram import createSpec
from sklearn import preprocessing


def preprocess_args(parser):

    prep_parser = parser.add_argument_group("Preprocess options")

    prep_parser.add_argument('--scaling', dest='scaling', action='store_true', help='Feature scaling or not')
    prep_parser.add_argument('--augment', dest='augment', action='store_true',
                        help='Use random tempo and gain perturbations.')
    prep_parser.add_argument('--duration', default=10.0, type=float, help='Duration of one EEG dataset')
    prep_parser.add_argument('--n-use-eeg', default=1, type=int, help='Number of eeg to use')
    prep_parser.add_argument('--n-features', type=int, help='Number of features to reshape from 1 channel feature')
    prep_parser.add_argument('--window-size', default=4.0, type=float, help='Window size for spectrogram in seconds')
    prep_parser.add_argument('--window-stride', default=2.0, type=float, help='Window stride for spectrogram in seconds')
    prep_parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
    prep_parser.add_argument('--spect', dest='spect', action='store_true', help='Use spectrogram as input')
    prep_parser.add_argument('--sample-rate', default='same', help='Sample rate')
    prep_parser.add_argument('--num-eigenvalue', default=0, type=int,
                             help='Number of eigen values to use from spectrogram')
    prep_parser.add_argument('--low-cutoff', default=0.01, type=float, help='High pass filter')
    prep_parser.add_argument('--high-cutoff', default=200.0, type=float, help='Low pass filter')
    prep_parser.add_argument('--muscle-noise', default=0.0, type=float)
    prep_parser.add_argument('--eye-noise', default=0.0, type=float)
    prep_parser.add_argument('--white-noise', default=0.0, type=float)
    prep_parser.add_argument('--shift-gain', default=0.0, type=float)
    prep_parser.add_argument('--spec-augment', default=0.0, type=float)
    prep_parser.add_argument('--channel-wise-mean', action='store_true')
    prep_parser.add_argument('--inter-channel-mean', action='store_true')
    prep_parser.add_argument('--no-power-noise', action='store_true')
    prep_parser.add_argument('--mfcc', dest='mfcc', action='store_true', help='MFCC')
    prep_parser.add_argument('--to-1d', dest='to_1d', action='store_true', help='Preprocess inputs to 1 dimension')

    return parser


class Preprocessor:
    def __init__(self, eeg_conf, phase, to_1d=False, scaling_axis=None):
        self.phase = phase
        self.sr = eeg_conf['sample_rate']
        self.l_cutoff = eeg_conf['low_cutoff']
        self.h_cutoff = eeg_conf['high_cutoff']
        self.spect = eeg_conf['spect']
        if self.spect:
            self.window_stride = eeg_conf['window_stride']
            self.window_size = eeg_conf['window_size']
            self.window = eeg_conf['window']
        self.n_features = eeg_conf['n_features']
        self.normalize = eeg_conf['scaling']
        self.cfg = eeg_conf
        self.spec_augment = eeg_conf['spec_augment']
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

    def preprocess(self, eeg):

        if self.sr != 'same' and int(self.sr) != eeg.sr:
            eeg.values = eeg.resample(self.sr)
            eeg.sr = self.sr

        if self.reproduce == 'chbmit-cnn':
            return torch.from_numpy(createSpec(eeg.values, eeg.sr, len(eeg.channel_list)))

        if self.reproduce == 'bonn-rnn':
            n_channel = min(len(eeg.channel_list), 22)
            timesteps = int(eeg.len_sec / 2 * eeg.sr)     # 15 * 256
            feature_dim = n_channel * eeg.values.shape[1] // timesteps     # 22 x (30 x 256) / (15 x 256) = 22 x 2
            return torch.from_numpy(eeg.values[:n_channel, :].reshape((feature_dim, timesteps)).T) # 22 x (30 x 256) -> (22 x 2) x (15 x 256)

        eeg.values = bandpass_filter(eeg.values, self.l_cutoff, self.h_cutoff, eeg.sr)

        eeg.values = np.clip(eeg.values,  -1000, 1000)

        if self.cfg['no_power_noise']:
            eeg.values = remove_power_noise(eeg.values, eeg.sr)

        n_channel = eeg.values.shape[0]

        if self.cfg['channel_wise_mean']:
            diff = eeg.values[:n_channel] - eeg.values[:n_channel].mean(axis=0)
            eeg.values = np.vstack((eeg.values, diff))
            eeg.channel_list += eeg.channel_list[:n_channel]

        if self.cfg['inter_channel_mean']:
            diff = (eeg.values[:n_channel].T - eeg.values[:n_channel].mean(axis=1).T).T
            eeg.values = np.vstack((eeg.values, diff))
            eeg.channel_list += eeg.channel_list[:n_channel]

        if self.phase in ['train']:
            if self.cfg['muscle_noise']:
                eeg.values = add_muscle_noise(eeg.values, eeg.sr, self.cfg['muscle_noise'])
            if self.cfg['eye_noise']:
                eeg.values = add_eye_noise(eeg.values, eeg.sr, self.cfg['eye_noise'])
            if self.cfg['white_noise']:
                eeg.values = add_white_noise(eeg.values, self.cfg['white_noise'])
            if self.cfg['shift_gain']:
                rate = np.random.normal(1.0 - self.cfg['shift_gain'], 1.0 + self.cfg['shift_gain'])
                eeg.values = shift_gain(eeg.values, rate=rate)

            # for i in range(len(eeg.channel_list)):
            #     eeg.values[i] = shift(eeg.values[i], eeg.sr * 5)
            #     eeg.values[i] = stretch(eeg.values[i], rate=0.3)
            #     eeg.values[i] = shift_pitch(eeg.values[i], rate=0.3)

        if self.to_1d:
            y = np.array([])
            if self.time_corr:
                y = np.hstack((y, self.calc_corr_frts(eeg, 'time')))
            if self.freq_corr:
                y = np.hstack((y, self.calc_corr_frts(eeg, 'freq')))
                y = torch.from_numpy(y)
        elif self.spect:
            y = to_spect(eeg, self.window_size, self.window_stride, self.window)    # channel x freq x time

            if self.spec_augment and self.phase in ['train']:
                y = time_and_freq_mask(y, rate=self.spec_augment)

        else:
            y = torch.from_numpy(eeg.values)  # channel x time

        if self.normalize:
            y = (y - y.mean()).div(y.std() + 0.001)

        if self.n_features:
            y = y.reshape(self.n_features, -1)

        return y

    def mfcc(self):
        raise NotImplementedError


def to_correlation_matrix(waves):
    return np.corrcoef(waves)


def calc_eigen_values_sorted(matrix):
    if np.any(np.isnan(matrix)):
        a = ''
    w, v = np.linalg.eig(matrix)
    w = np.absolute(w)
    w.sort()
    return w


# Take the upper right triangle of a matrix
def flatten_corr_upper_right(matrix):
    accum = []
    for i in range(matrix.shape[0]):
        for j in range(i+1, matrix.shape[1]):
            accum.append(matrix[i, j])

    return np.array(accum)
