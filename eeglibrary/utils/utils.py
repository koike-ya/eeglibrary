from __future__ import print_function, division

from pathlib import Path

import pandas as pd
from ml.models.ml_models.toolbox import *
from ml.src.dataloader import make_weights_for_balanced_classes
from torch.utils.data.sampler import WeightedRandomSampler

from eeglibrary.src import EEGDataSet, EEGDataLoader
from eeglibrary.src.eeg_loader import from_mat


def common_eeg_setup(eeg_path='', mat_col=''):
    eeg_conf = dict(spect=True,
                    window_size=1.0,
                    window_stride=1.0,
                    window='hamming',
                    sample_rate=1500,
                    noise_dir=None,
                    noise_prob=0.4,
                    noise_levels=(0.0, 0.5))
    eeg_path = eeg_path or '/home/tomoya/workspace/kaggle/seizure-prediction/input/Dog_1/train/Dog_1_interictal_segment_0001.mat'
    mat_col = mat_col or 'interictal_segment_1'
    return from_mat(eeg_path, mat_col), eeg_conf


def set_dataloader(args, eeg_conf, class_names, phase, label_func, device='cpu'):
    if phase in ['test', 'inference']:
        return_path = True if phase == 'inference' else False
        dataset = EEGDataSet(args.test_manifest, eeg_conf, label_func, classes=class_names, to_1d=args.to_1d,
                             return_path=return_path)
        dataloader = EEGDataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                          pin_memory=True, shuffle=False)
    else:
        manifest_path = [value for key, value in vars(args).items() if phase in key][0]
        dataset = EEGDataSet(manifest_path, eeg_conf, label_func, classes=class_names, to_1d=args.to_1d, device=device)
        weights = make_weights_for_balanced_classes(dataset.labels_index(), len(class_names))
        sampler = WeightedRandomSampler(weights, int(len(dataset) * args.epoch_rate))
        dataloader = EEGDataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                          pin_memory=True, sampler=sampler, drop_last=True)
    return dataloader


def set_model(args, class_names, eeg_conf, device):
    if args.model_name == '2dcnn_1':
        model = cnn_1_16_399(eeg_conf, n_labels=len(class_names))
    elif args.model_name == '2dcnn_2':
        model = cnn_16_751_751(eeg_conf, n_labels=len(class_names))
    elif args.model_name == 'rnn':
        cnn, out_ftrs = cnn_ftrs_16_751_751(eeg_conf)
        model = RNN(cnn, out_ftrs, args.batch_size, args.rnn_type, class_names, eeg_conf=eeg_conf,
                    rnn_hidden_size=args.rnn_hidden_size, nb_layers=args.rnn_n_layers)
    elif args.model_name == '3dcnn':
        model = cnn_1_16_751_751(eeg_conf, n_labels=len(class_names))
    elif args.model_name == 'xgboost':
        model = XGBoost(list(range(len(class_names))))
    elif args.model_name == 'sgdc':
        model = SGDC(list(range(len(class_names))))
    elif args.model_name in ['kneighbor', 'knn']:
        args.model_name = 'kneighbor'
        model = KNN(list(range(len(class_names))))
    else:
        raise NotImplementedError

    if 'nn' in args.model_name:
        model = model.to(device)
        if hasattr(args, 'silent') and (not args.silent):
            print(model)

    return model


def arrange_paths(args, sub_name):
    args.train_manifest = 'splitted/{}/train_manifest.csv'.format(sub_name)
    args.val_manifest = 'splitted/{}/val_manifest.csv'.format(sub_name)
    args.test_manifest = 'splitted/{}/test_manifest.csv'.format(sub_name)
    args.model_path = 'model/' + args.model_name + '/{}.pkl'.format(sub_name)
    args.log_id = sub_name

    return args


def concat_manifests(manifests, save_name):
    # manifestファイルのpathが入ったリストを受け取って、dfを読み込んで結合して返す
    assert isinstance(manifests, list)

    master_df = pd.DataFrame()
    for manifest in manifests:
        master_df = pd.concat([master_df, pd.read_csv(manifest, header=None)], axis=0, sort=False)

    master_df.to_csv(Path(manifests[0]).parent / save_name, index=False, header=None)
    return Path(manifests[0]).parent / save_name