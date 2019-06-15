from torch.utils.data import DataLoader
import torch
import numpy as np
from wrapper.src import WrapperDataLoader


class EEGDataLoader(WrapperDataLoader):
    def __init__(self, *args, **kwargs):
        super(EEGDataLoader, self).__init__(*args, **kwargs)
