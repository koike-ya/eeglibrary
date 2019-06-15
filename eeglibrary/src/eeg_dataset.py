from eeglibrary.eeglibrary.src.eeg_parser import parse_eeg
from eeglibrary.eeglibrary.src import EEG
from eeglibrary.eeglibrary.src.preprocessor import Preprocessor
from wrapper.src import ManifestDataSet


class EEGDataSet(ManifestDataSet):
    def __init__(self, manifest_path, data_conf, eeg_conf, to_1d=False, normalize=False, augment=False,
                 device='cpu', return_path=False):
        """
        data_conf: {
            'load_func': Function to load data from manifest correctly,
            'labels': List of labels or None if it's made from path
            'label_func': Function to extract labels from path or None if labels are given as 'labels'
        }

        """
        super(EEGDataSet, self).__init__(manifest_path, data_conf)
        self.preprocessor = Preprocessor(eeg_conf, normalize, augment, to_1d, scaling_axis=None)
        path_list = self._load_path_list(manifest_path)
        self.suffix = path_list[0][-4:]
        self.duration = eeg_conf['duration']
        self.path_list = self.pack_paths(path_list, data_conf['label_func'])
        self.size = len(self.path_list)
        self.return_path = return_path
        self.device = device

    def __getitem__(self, idx):
        eeg_paths, label = self.path_list[idx]
        eeg = parse_eeg(eeg_paths)
        y = self.preprocessor.preprocess(eeg)

        if self.labels:
            return y, label
        elif self.return_path:
            return (y, eeg_paths)
        else:
            return y

    def __len__(self):
        return self.size

    def _load_path_list(self, paths):
        path_list = []
        for path in str(paths).split(','):
            with open(path, 'r') as f:
                path_list.extend(f.readlines())

        # removing \n character from string
        return [p.strip() for p in path_list]

    def labels_index(self, paths=None, label_func=None) -> [int]:
        if not self.labels:
            return [None] * len(paths)
        if paths:
            return [self.labels.index(label_func(path)) for path in paths]
        return [label for path, label in self.path_list]

    def pack_paths(self, path_list, label_func=None):
        if self.duration == 1:
            return [([p], label) for p, label in zip(path_list, self.labels_index(path_list, label_func))]

        one_eeg = EEG.load_pkl(path_list[0])
        len_sec = one_eeg.len_sec
        n_use_eeg = int(self.duration / len_sec)
        assert n_use_eeg == self.duration / len_sec, 'Duration must be common multiple of {}'.format(len_sec)

        labels = self.labels_index(path_list, label_func)
        packed_path_label_list = [(path_list[i:i + n_use_eeg], labels[i:i + n_use_eeg]) for i in
                                  range(0, len(path_list), n_use_eeg)]
        for i, (paths, labels) in enumerate(packed_path_label_list):
            if len(set(labels)) != 1:
                packed_path_label_list.pop(i)
            else:
                packed_path_label_list[i] = (paths, labels[0])

        return packed_path_label_list
