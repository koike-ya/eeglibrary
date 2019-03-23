from torch.utils.data import DataLoader


class EEGDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(EEGDataLoader, self).__init__(*args, **kwargs)

    def collate_fn(self, batch):
        batch
        a = ''
