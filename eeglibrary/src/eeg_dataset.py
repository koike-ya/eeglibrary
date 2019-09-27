from eeglibrary.src import EEG
from eeglibrary.src.eeg_parser import parse_eeg
from eeglibrary.src.preprocessor import Preprocessor
from ml.src.dataset import ManifestDataSet


class EEGDataSet(ManifestDataSet):
    def __init__(self, manifest_path, data_conf, to_1d=False, normalize=False, augment=False,
                 device='cpu', return_path=False):
        """
        data_conf: {
            'load_func': Function to load data from manifest correctly,
            'labels': List of labels or None if it's made from path
            'label_func': Function to extract labels from path or None if labels are given as 'labels'
        }

        """
        super(EEGDataSet, self).__init__(manifest_path, data_conf, load_func=data_conf['load_func'],
                                         label_func=data_conf['label_func'])
        self.preprocessor = Preprocessor(data_conf, normalize, augment, to_1d, scaling_axis=None)
        # self.suffix = self.path_list[0][-4:]
        self.path_list = self.pack_paths(self.path_list, data_conf['duration'])
        self.return_path = return_path

    def __getitem__(self, idx):
        eeg_paths, label = self.path_list[idx]
        eeg = parse_eeg(eeg_paths)
        # TODO multi channel でRNNをどう使うか
        x = self.preprocessor.preprocess(eeg)
        x = x.narrow(0, 0, 1).squeeze(dim=0)   # channel 次元を一つだけ取り出す

        if self.labels:
            return x, label
        elif self.return_path:
            return (x, eeg_paths)
        else:
            return x

    def pack_paths(self, path_list, duration):
        one_eeg = EEG.load_pkl(path_list[0])
        len_sec = one_eeg.len_sec
        n_use_eeg = int(duration / len_sec)
        assert n_use_eeg == duration / len_sec, f'Duration must be common multiple of {len_sec}'

        label_list = [self.label_func(path) for path in self.path_list]

        if n_use_eeg == 1:
            return [([p], label) for p, label in zip(path_list, label_list)]

        packed_path_label_list = []
        for i in range(0, len(path_list), n_use_eeg):
            paths, labels = path_list[i:i + n_use_eeg], label_list[i:i + n_use_eeg]

            # TODO ソフトラベルを作るのもあり。
            if len(set(labels)) != 1:  # 結合したときにラベルが異なるものがある場合は、データから除外する
                continue

            packed_path_label_list.append((paths, labels[0]))

        return packed_path_label_list

    def get_feature_size(self):
        eeg = parse_eeg(self.path_list[0][0])
        x = self.preprocessor.preprocess(eeg)
        return x.size(1)  # freq

    def get_seq_len(self):
        eeg = parse_eeg(self.path_list[0][0])
        x = self.preprocessor.preprocess(eeg)
        return x.size(2)  # time

    def get_labels(self):
        return [label for paths, label in self.path_list]
