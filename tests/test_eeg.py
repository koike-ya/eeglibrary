from unittest import TestCase
from eeglibrary.src.eeg import EEG
from pathlib import Path
from eeglibrary.src.eeg_loader import from_mat


class TestEEG(TestCase):

    def setUp(self):
        eeg_conf = dict(sample_rate=16000,
                        window_size=0.02,
                        window_stride=0.01,
                        window='hamming',
                        wave_split_sec=2.0,
                        noise_dir=None,
                        noise_prob=0.4,
                        noise_levels=(0.0, 0.5))
        self.eeg = from_mat(
            '/home/tomoya/workspace/kaggle/seizure-prediction/input/Dog_1/train/Dog_1_interictal_segment_0001.mat',
            'interictal_segment_1'
        )

    def tearDown(self):
        pass

    def test_to_pkl(self):
        out_path = 'tmp.pkl'
        self.eeg.to_pkl(out_path)
        self.assertTrue(Path(out_path).is_file())
        Path(out_path).unlink()

    def test_load_pkl(self):
        out_path = 'tmp.pkl'
        self.eeg.to_pkl(out_path)
        eeg = EEG.load_pkl(out_path)
        self.assertTrue(isinstance(eeg, EEG))
        Path(out_path).unlink()

    def test_split(self):
        self.assertEqual(len(self.eeg.split()), 1200)
        window_size_pattern = (0, 0.5, 4)
        window_stride_pattern = (0, 0.5, 'same', 4)
        padding_pattern = (0.0, 'same', 0.5)
        for window_size in window_size_pattern:
            for window_stride in window_stride_pattern:
                for padding in padding_pattern:
                    try:
                        # TODO ちゃんと計算してlenが合っているかテストする
                        self.eeg.split(window_size, window_stride, padding)
                    except AssertionError as e:
                        continue

    def test_resample(self):
        upsampled = self.eeg.resample(int(self.eeg.sr) * 2)
        self.assertEqual(upsampled .shape[1] // self.eeg.len_sec, int(self.eeg.sr) * 2)

        donwsampled = self.eeg.resample(int(self.eeg.sr) // 2)
        self.assertEqual(donwsampled.shape[1] // self.eeg.len_sec, int(self.eeg.sr) // 2)

