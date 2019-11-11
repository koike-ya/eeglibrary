import numpy as np
from eeglibrary.src import EEG
from eeglibrary.src.eeg_parser import parse_eeg
from eeglibrary.src.preprocessor import Preprocessor
from ml.src.dataset import ManifestDataSet


class EEGDataSet(ManifestDataSet):
    def __init__(self, manifest_path, data_conf, phase, load_func, label_func, normalize=False, augment=False,
                 device='cpu', return_path=False):
        """
        data_conf: {
            'load_func': Function to load data from manifest correctly,
            'labels': List of labels or None if it's made from path
            'label_func': Function to extract labels from path or None if labels are given as 'labels'
        }

        """
        super(EEGDataSet, self).__init__(manifest_path, data_conf, load_func=load_func, label_func=label_func)
        self.preprocessor = Preprocessor(data_conf, normalize, augment, data_conf['to_1d'], scaling_axis=None)
        # self.suffix = self.path_list[0][-4:]
        self.path_list = self.pack_paths(self.path_list, data_conf['duration'], data_conf['n_use_eeg'])
        self.return_path = return_path
        self.model_type = data_conf['model_type']
        self.processed_input_size = self.get_processed_size()
        self.batch_size = data_conf['batch_size']

    def __getitem__(self, idx):
        eeg_paths, label = self.path_list[idx]
        eeg_ = parse_eeg(eeg_paths)
        # import numpy as np
        # eeg.values = np.nan_to_num(eeg.values)
        try:
            x = self.preprocessor.preprocess(eeg_)
        except np.linalg.LinAlgError as e:
            return self.__getitem__(idx + 1)

        x = self._reshape_input(x)

        if self.labels:
            return x, label
        elif self.return_path:
            return (x, eeg_paths)
        else:
            return x

    def _reshape_input(self, x):
        if len(self.processed_input_size) == 3 and self.model_type == 'rnn':
            x = x.reshape(self.processed_input_size[0], -1, self.processed_input_size[2])
        return x

    def pack_paths(self, path_list, duration, n_use_eeg):
        one_eeg = EEG.load_pkl(path_list[0])
        len_sec = one_eeg.len_sec
        if not n_use_eeg:
            n_use_eeg = int(duration / len_sec)
            assert n_use_eeg == duration / len_sec, f'Duration must be common multiple of {len_sec}'

        label_list = [self.label_func(path) for path in self.path_list]

        if n_use_eeg == 1:
            return [([p], label) for p, label in zip(path_list, label_list)]

        packed_path_label_list = []
        for i in range(0, len(path_list), n_use_eeg):
            if i + n_use_eeg >= len(path_list):
                continue
            paths, labels = path_list[i:i + n_use_eeg], label_list[i:i + n_use_eeg]
            assert len(paths) == n_use_eeg

            # TODO ソフトラベルを作るのもあり。
            if len(set(labels)) != 1:  # 結合したときにラベルが異なるものがある場合は、データから除外する
                continue

            packed_path_label_list.append((paths, labels[0]))

        return packed_path_label_list

    def get_labels(self):
        return [label for paths, label in self.path_list]

    def get_processed_size(self):
        eeg = parse_eeg(self.path_list[0][0])
        x = self.preprocessor.preprocess(eeg)
        return x.size()

    def get_feature_size(self):
        if self.model_type == 'rnn':
            return self.get_processed_size()[0]
        elif self.model_type in ['2d_cnn', 'cnn_rnn']:
            return self.get_image_size()

    def get_seq_len(self):
        size = self.get_processed_size()
        if len(size) == 2:
            return size[1]
        elif len(size) == 3:
            return size[2]
        else:
            NotImplementedError

    def get_batch_norm_size(self, sequense_wise=False):
        size = self.get_processed_size()
        if not sequense_wise:
            return self.get_n_channels()

        if self.model_type in ['rnn', 'cnn_rnn']:
            return size[1]  # b x f x t -> t x b x f になったあと、(t x b) x fになるので、fを返す
        else:
            NotImplementedError

    def get_image_size(self):
        size = self.get_processed_size()
        return size[1], size[2]

    def get_n_channels(self):
        return self.get_processed_size()[0]
