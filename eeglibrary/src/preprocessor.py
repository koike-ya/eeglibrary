import numpy as np
import torch
from eeglibrary.src.signal_processor import to_spect
from sklearn import preprocessing


def preprocess_args(parser):

    prep_parser = parser.add_argument_group("Preprocess options")

    prep_parser.add_argument('--scaling', dest='scaling', action='store_true', help='Feature scaling or not')
    prep_parser.add_argument('--augment', dest='augment', action='store_true',
                        help='Use random tempo and gain perturbations.')
    prep_parser.add_argument('--duration', default=10.0, type=float, help='Duration of one EEG dataset')
    prep_parser.add_argument('--window-size', default=4.0, type=float, help='Window size for spectrogram in seconds')
    prep_parser.add_argument('--window-stride', default=2.0, type=float, help='Window stride for spectrogram in seconds')
    prep_parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
    prep_parser.add_argument('--spect', dest='spect', action='store_true', help='Use spectrogram as input')
    prep_parser.add_argument('--sample-rate', default=400, type=int, help='Sample rate')
    prep_parser.add_argument('--num-eigenvalue', default=0, type=int,
                             help='Number of eigen values to use from spectrogram')
    prep_parser.add_argument('--low-cutoff', default=0.01, type=float, help='Low pass filter')
    prep_parser.add_argument('--high-cutoff', default=10000.0, type=float, help='High pass filter')
    prep_parser.add_argument('--mfcc', dest='mfcc', action='store_true', help='MFCC')
    prep_parser.add_argument('--to_1d', dest='to_1d', action='store_true', help='Preprocess inputs to 1 dimension')

    return parser


class Preprocessor:
    def __init__(self, eeg_conf, normalize=False, augment=False, to_1d=False, scaling_axis=None):
        self.sr = eeg_conf['sample_rate']
        self.l_cutoff = eeg_conf['low_cutoff']
        self.h_cutoff = eeg_conf['high_cutoff']
        self.spect = eeg_conf['spect']
        if self.spect:
            self.window_stride = eeg_conf['window_stride']
            self.window_size = eeg_conf['window_size']
            self.window = eeg_conf['window']
        self.normalize = normalize
        self.augment = augment
        self.to_1d = to_1d
        self.time_corr = True
        self.freq_corr = True
        self.use_eig_values = True
        self.scaling_axis = scaling_axis

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

        if self.sr != eeg.sr:
            eeg.values = eeg.resample(self.sr)
            eeg.sr = self.sr

        if self.augment:
            raise NotImplementedError

        # Filtering
        # eeg.values = self.bandpass_filter(eeg.values)

        if self.to_1d:
            y = np.array([])
            if self.time_corr:
                y = np.hstack((y, self.calc_corr_frts(eeg, 'time')))
            if self.freq_corr:
                y = np.hstack((y, self.calc_corr_frts(eeg, 'freq')))
                y = torch.from_numpy(y)
        elif self.spect:
            y = to_spect(eeg, self.window_size, self.window_stride, self.window)    # channel x freq x time
        else:
            y = torch.from_numpy(eeg.values)  # channel x time

        if self.normalize:
            # TODO Feature(time) axis normalization, Index(channel) axis normalization
            raise NotImplementedError
            # y = (y - y.mean()).div(y.std())

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
