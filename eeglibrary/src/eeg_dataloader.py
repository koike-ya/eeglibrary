from ml.src.dataloader import WrapperDataLoader


class EEGDataLoader(WrapperDataLoader):
    def __init__(self, *args, **kwargs):
        super(EEGDataLoader, self).__init__(*args, **kwargs)
