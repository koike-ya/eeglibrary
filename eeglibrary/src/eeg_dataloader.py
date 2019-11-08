from ml.src.dataloader import WrapperDataLoader, make_weights_for_balanced_classes, WeightedRandomSampler
import torch


class EEGDataLoader(WrapperDataLoader):
    def __init__(self, model_type, *args, **kwargs):
        super(EEGDataLoader, self).__init__(*args, **kwargs)
        self.seq_len = self.dataset.get_seq_len()
        self.model_type = model_type

    def get_input_size(self):
        return self.dataset.get_feature_size()

    def get_image_size(self):
        return self.dataset.get_image_size()

    def get_n_channels(self):
        return self.dataset.get_n_channels()

    def get_batch_norm_size(self):
        return self.dataset.get_batch_norm_size()

    def get_seq_len(self):
        return self.dataset.get_seq_len()


def set_dataloader(dataset, phase, cfg, shuffle=True):
    if isinstance(cfg['sample_balance'], str):
        cfg['sample_balance'] = [1.0] * len(cfg['class_names'])
    if phase in ['test', 'infer']:
        # TODO batch normalization をeval()してdrop_lastしなくてよいようにする。
        dataloader = EEGDataLoader(model_type=cfg['model_type'], dataset=dataset, batch_size=cfg['batch_size'],
                                   num_workers=cfg['n_jobs'], pin_memory=True, sampler=None, shuffle=False)
    else:
        if sum(cfg['sample_balance']) != 0.0:
            if cfg['task_type'] == 'classify':
                weights = make_weights_for_balanced_classes(dataset.get_labels(), len(cfg['class_names']),
                                                            cfg['sample_balance'])
            else:
                weights = [torch.Tensor([1.0])] * len(dataset.get_labels())
            sampler = WeightedRandomSampler(weights, int(len(dataset) * cfg['epoch_rate']))
            shuffle = False
        else:
            sampler = None
        dataloader = EEGDataLoader(cfg['model_type'], dataset=dataset, batch_size=cfg['batch_size'],
                                   num_workers=cfg['n_jobs'], pin_memory=True, sampler=sampler,
                                   shuffle=shuffle)

    return dataloader
