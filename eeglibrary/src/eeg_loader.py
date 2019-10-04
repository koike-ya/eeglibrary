from eeglibrary.src.eeg import EEG
from scipy.io import loadmat


def from_mat(file_path, mat_col) -> EEG:
    data = {}
    mat = loadmat(file_path)
    header = str(mat['__header__'])

    value_col = detect_mat_value_col(mat)

    try:
        for i, key in enumerate(mat[value_col].dtype.names):
            data[key] = mat[value_col][0][0][i]
    except KeyError as e:
        raise KeyError("eeg_file {} doesn't have info about 'interictal_segment_1', " +
                       "not implemented except this key.".format(value_col))

    eeg = EEG(values=data['data'],
              channel_list=data['channels'][0],
              len_sec=data['data_length_sec'][0][0],
              sr=data['sampling_frequency'][0][0],
              header=header)

    return eeg


def detect_mat_value_col(mat):
    # TODO try exceptで必ず検出する
    mat_col = [col for col in list(mat.keys()) if not col.startswith('_')][0]
    return mat_col


def from_eeg(file_path):
    raise NotImplementedError
