from ml.src.dataloader import WrapperDataLoader, make_weights_for_balanced_classes, WeightedRandomSampler



class EEGDataLoader(WrapperDataLoader):
    def __init__(self, *args, **kwargs):
        super(EEGDataLoader, self).__init__(*args, **kwargs)
        self.seq_len = self.dataset.get_seq_len()


def set_dataloader(dataset, phase, cfg, shuffle=False):
    if phase in ['test', 'infer']:
        dataloader = EEGDataLoader(dataset, batch_size=cfg['batch_size'], num_workers=cfg['n_jobs'],
                                       pin_memory=True, sampler=None, shuffle=shuffle)
    else:
        if sum(cfg['sample_balance']) != 0.0:
            if cfg['task_type'] == 'classify':
                _ = dataset.get_labels()
                weights = make_weights_for_balanced_classes(dataset.get_labels(), len(cfg['class_names']),
                                                            cfg['sample_balance'])
            else:
                weights = [torch.Tensor([1.0])] * len(dataset.get_labels())
            sampler = WeightedRandomSampler(weights, int(len(dataset) * cfg['epoch_rate']))
        else:
            sampler = None
        dataloader = EEGDataLoader(dataset, batch_size=cfg['batch_size'], num_workers=cfg['n_jobs'],
                                       pin_memory=True, sampler=sampler, drop_last=True, shuffle=shuffle)
    return dataloader
