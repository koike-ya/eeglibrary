from eeglibrary.src.eeg_loader import from_mat


def common_eeg_setup(eeg_path='', mat_col=''):
    eeg_conf = dict(window_size=0.1,
                    window_stride=0.1,
                    window='hamming',
                    noise_dir=None,
                    noise_prob=0.4,
                    noise_levels=(0.0, 0.5))
    eeg_path = eeg_path or '/home/tomoya/workspace/kaggle/seizure-prediction/input/Dog_1/train/Dog_1_interictal_segment_0001.mat'
    mat_col = mat_col or 'interictal_segment_1'
    return from_mat(eeg_path, mat_col), eeg_conf
