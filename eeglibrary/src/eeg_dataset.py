from torch.utils.data import Dataset
from eeglibrary.src.eeg_parser import EEGParser
from eeglibrary.src import EEG
import numpy as np
import torch

class EEGDataSet(Dataset, EEGParser):
    def __init__(self, manifest_filepath, eeg_conf, classes=None, duration=1.0, normalize=False, augment=False,
                 return_path=False):
        super(EEGDataSet, self).__init__(eeg_conf, normalize, augment)
        self.classes = classes # self.classes is None in test dataset
        with open(manifest_filepath, 'r') as f:
            path_list = f.readlines()
        path_list = [p.strip() for p in path_list]
        self.suffix = path_list[0][-4:]
        self.duration = eeg_conf['duration']
        self.path_list = self.pack_paths(path_list)
        self.size = len(self.path_list)
        self.return_path = return_path

    def __getitem__(self, idx):
        eeg_paths, label = self.path_list[idx]
        y = self.parse_eeg(eeg_paths)
        if self.classes:
            return y, label
        elif self.return_path:
            return y, eeg_paths
        else:
            return y

    def __len__(self):
        return self.size

    def _parse_label(self, path):
        if self.suffix == '.pkl':
            return path.split('/')[-2].split('_')[2]
        else:
            return path.split('_')[2]

    def labels_index(self, paths=None) -> [int]:
        if not self.classes:
            return [None] * len(paths)
        if paths:
            return [self.classes.index(self._parse_label(path)) for path in paths]
        return [label for path, label in self.path_list]

    def pack_paths(self, path_list):
        if self.duration == 1:
            return [([p], label) for p, label in zip(path_list, self.labels_index(path_list))]

        one_eeg = EEG.load_pkl(path_list[0])
        len_sec = one_eeg.len_sec
        n_use_eeg = int(self.duration / len_sec)
        assert n_use_eeg == self.duration / len_sec, 'Duration must be common multiple of {}'.format(len_sec)

        labels = self.labels_index(path_list)
        packed_path_label_list = [(path_list[i:i + n_use_eeg], labels[i:i + n_use_eeg]) for i in
                                  range(0, len(path_list), n_use_eeg)]
        for i, (paths, labels) in enumerate(packed_path_label_list):
            if len(set(labels)) != 1:
                packed_path_label_list.pop(i)
            else:
                packed_path_label_list[i] = (paths, labels[0])

        return packed_path_label_list