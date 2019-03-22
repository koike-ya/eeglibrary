from torch.utils.data import Dataset
from eeglibrary.src.eeg_parser import EEGParser


class EEGDataSet(Dataset, EEGParser):
    def __init__(self, manifest_filepath, eeg_conf, normalize=False, augment=False):
        with open(manifest_filepath, 'r') as f:
            path_list = f.readlines()
        path_list = [p.strip() for p in path_list]

        self.path_list = path_list
        self.size = len(path_list)
        super(EEGDataSet, self).__init__(eeg_conf, normalize, augment)

    def __getitem__(self, index):
        eeg_path = self.path_list[index]
        y = self.parse_eeg(eeg_path)
        label = eeg_path.split('_')[2]
        return y, label

    def __len__(self):
        return self.size
