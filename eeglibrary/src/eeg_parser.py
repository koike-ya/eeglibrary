from eeglibrary.src import eeg
import numpy as np
from eeglibrary.src import eeg_loader


def _load_eeg(eeg_path):
    if eeg_path[-4:] == '.pkl':
        eeg_ = eeg.EEG.load_pkl(eeg_path)
    else:
        eeg_ = eeg_loader.from_mat(eeg_path, mat_col='')
    return eeg_


def _merge_eeg(paths):
    eeg_ = _load_eeg(paths[0])
    if len(paths) != 1:
        for path in paths[1:]:
            eeg_.values = np.hstack((eeg_.values, _load_eeg(path).values))

    eeg_.len_sec = eeg_.len_sec * len(paths)
    return eeg_


def parse_eeg(eeg_path) -> np.array:
    if isinstance(eeg_path, list):
        return _merge_eeg(eeg_path)
    else:
        return _load_eeg(eeg_path)
