from torch.utils.data import Dataset
from eeglibrary.src.eeg_parser import EEGParser


class EEGDataSet(Dataset, EEGParser):
    def __init__(self, manifest_filepath, eeg_conf, classes=None, normalize=False, augment=False, return_path=False):
        with open(manifest_filepath, 'r') as f:
            path_list = f.readlines()
        path_list = [p.strip() for p in path_list]

        self.path_list = path_list
        self.size = len(path_list)
        self.classes = classes # self.classes is None in test dataset
        self.suffix = path_list[0][-4:]
        self.return_path = return_path
        super(EEGDataSet, self).__init__(eeg_conf, normalize, augment)

    def __getitem__(self, index):
        eeg_path = self.path_list[index]
        y = self.parse_eeg(eeg_path)
        if self.classes:
            label = self._parse_label(eeg_path)
            return y, self.classes.index(label)
        elif self.return_path:
            return y, eeg_path
        else:
            return y

    def __len__(self):
        return self.size

    def _parse_label(self, path):
        if self.suffix == '.pkl':
            return path.split('/')[-2].split('_')[2]
        else:
            return path.split('_')[2]

    def labels_index(self) -> [int]:
        return [self.classes.index(self._parse_label(path)) for path in self.path_list]
