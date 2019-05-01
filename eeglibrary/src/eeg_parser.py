from eeglibrary.src import eeg_loader
import eeglibrary
import numpy as np
import scipy.signal
import librosa
import torch


windows = {'hamming': scipy.signal.hamming, 'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}


class EEGParser:
    def __init__(self, eeg_conf, normalize=False, augment=False):
        self.sr = eeg_conf['sample_rate']
        self.spect = eeg_conf['spect']
        if self.spect:
            # self.sr = eeg_conf['window_stride']
            self.window_stride = eeg_conf['window_stride']
            self.window_size = eeg_conf['window_size']
            self.window = windows.get(eeg_conf['window'], windows['hamming'])
        self.normalize = normalize
        self.augment = augment

    def _load_eeg(self, eeg_path):
        if eeg_path[-4:] == '.pkl':
            eeg = eeglibrary.EEG.load_pkl(eeg_path)
        else:
            eeg = eeg_loader.from_mat(eeg_path, mat_col='')
        return eeg

    def _merge_eeg(self, paths):
        eeg = self._load_eeg(paths[0])
        if len(paths) != 1:
            for path in paths[1:]:
                eeg.values = np.hstack((eeg.values, self._load_eeg(path).values))
        eeg.len_sec = eeg.len_sec * len(paths)
        return eeg

    def parse_eeg(self, eeg_path) -> np.array:
        if isinstance(eeg_path, list):
            eeg = self._merge_eeg(eeg_path)
        else:
            eeg = self._load_eeg(eeg_path)

        if self.augment:
            raise NotImplementedError

        if self.sr != eeg.sr:
            eeg.values = eeg.resample(self.sr)
            eeg.sr = self.sr

        if self.spect:
            y = self.to_spect(eeg)
        else:
            y = torch.FloatTensor(eeg.values).view(1, eeg.values.shape[0], eeg.values.shape[1])

        return y

    def to_spect(self, eeg):
        n_fft = int(eeg.sr * self.window_size)
        win_length = n_fft
        hop_length = int(eeg.sr * self.window_stride)
        spect_tensor = torch.Tensor()
        # STFT
        for i in range(len(eeg.channel_list)):
            y = eeg.values[i].astype(float)
            D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                             win_length=win_length, window=self.window)
            spect, phase = librosa.magphase(D)
            spect = torch.FloatTensor(spect)
            if self.normalize:
                mean = spect.mean()
                std = spect.std()
                spect.add_(-mean)
                spect.div_(std)
            spect_tensor = torch.cat((spect_tensor, spect.view(1, spect.size(0), -1)), 0)

        return spect_tensor
