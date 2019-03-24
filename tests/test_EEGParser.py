from unittest import TestCase

from eeglibrary.src.utils import common_eeg_setup
from eeglibrary.src.eeg_parser import EEGParser


class TestEEGParser(TestCase):

    def setUp(self):
        self.eeg, self.eeg_conf = common_eeg_setup()
        spect = True
        self.parser = EEGParser(self.eeg_conf, spect)

    def test_to_spect(self):
        self.eeg.values = self.eeg.resample(self.eeg_conf['sample_rate'])
        self.eeg.sr = self.eeg_conf['sample_rate']
        spect_tensor = self.parser.to_spect(self.eeg)
        self.assertEqual(spect_tensor.size(0), len(self.eeg.channel_list))
