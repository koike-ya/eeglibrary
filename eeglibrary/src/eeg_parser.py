import eeglibrary
import numpy as np
from eeglibrary.src import eeg_loader


def _load_eeg(eeg_path):
    if eeg_path[-4:] == '.pkl':
        eeg = eeglibrary.EEG.load_pkl(eeg_path)
    else:
        eeg = eeg_loader.from_mat(eeg_path, mat_col='')
    return eeg


def _merge_eeg(paths):
    eeg = _load_eeg(paths[0])
    if len(paths) != 1:
        for path in paths[1:]:
            eeg.values = np.hstack((eeg.values, _load_eeg(path).values))
    eeg.len_sec = eeg.len_sec * len(paths)
    return eeg


def parse_eeg(eeg_path) -> np.array:
    if isinstance(eeg_path, list):
        return _merge_eeg(eeg_path)
    else:
        return _load_eeg(eeg_path)
