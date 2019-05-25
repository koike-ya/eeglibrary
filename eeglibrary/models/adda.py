"""
in: 学習済みモデルとちゃんとtarget用のアノテーションされたデータ
out: よりdomain不変な特徴量を作成できるようになった学習済みモデル
"""

"""
Implements ADDA:
Adversarial Discriminative Domain Adaptation, Tzeng et al. (2017)
"""

from pathlib import Path

import torch
from eeglibrary.models import adda
from eeglibrary.src import EEGDataSet, EEGDataLoader
from eeglibrary.utils import train_args, add_adda_args, TensorBoardLogger
from eeglibrary.utils import set_eeg_conf, init_device, init_seed, concat_manifests, set_model
from torch import nn
from tqdm import tqdm, trange
from copy import deepcopy


def loop_iterable(iterable):
    while True:
        yield from iterable


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


def adda(args, source_model, eeg_conf, label_func, class_names, device, src_manifests, target_manifests):
    target_model = deepcopy(source_model)
    source_model.load_state_dict(torch.load(args.model_path))
    source_model.eval()
    set_requires_grad(source_model, requires_grad=False)

    clf = source_model
    source_model = source_model.features

    target_model.load_state_dict(torch.load(args.model_path))
    in_features = target_model.classifier[0].in_features
    target_model = target_model.features

    discriminator = nn.Sequential(
        nn.Linear(in_features, 400),
        nn.ReLU(),
        nn.Linear(400, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    ).to(device)

    half_batch = args.batch_size // 2

    source_manifest = concat_manifests(src_manifests, 'source')
    target_manifest = concat_manifests(target_manifests, 'target')

    source_dataset = EEGDataSet(source_manifest, eeg_conf, label_func, class_names, args.to_1d, device=device)
    source_loader = EEGDataLoader(source_dataset, batch_size=half_batch, num_workers=args.num_workers,
                                      pin_memory=True, drop_last=True)

    target_dataset = EEGDataSet(target_manifest, eeg_conf, label_func, class_names, args.to_1d, device=device)
    target_loader = EEGDataLoader(target_dataset, batch_size=half_batch, num_workers=args.num_workers,
                                      pin_memory=True, drop_last=True)

    discriminator_optim = torch.optim.Adam(discriminator.parameters())
    target_optim = torch.optim.Adam(target_model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, args.adda_epochs + 1):
        batch_iterator = zip(loop_iterable(source_loader), loop_iterable(target_loader))

        total_loss = 0
        total_accuracy = 0
        for _ in trange(args.iterations, leave=False):
            # Train discriminator
            set_requires_grad(target_model, requires_grad=False)
            set_requires_grad(discriminator, requires_grad=True)
            for _ in range(args.k_disc):
                (source_x, _), (target_x, _) = next(batch_iterator)
                source_x, target_x = source_x.to(device), target_x.to(device)

                source_features = source_model(source_x).view(source_x.shape[0], -1)
                target_features = target_model(target_x).view(target_x.shape[0], -1)

                discriminator_x = torch.cat([source_features, target_features])
                discriminator_y = torch.cat([torch.ones(source_x.shape[0], device=device),
                                             torch.zeros(target_x.shape[0], device=device)])

                preds = discriminator(discriminator_x).squeeze()
                loss = criterion(preds, discriminator_y)

                discriminator_optim.zero_grad()
                loss.backward()
                discriminator_optim.step()

                total_loss += loss.item()
                total_accuracy += ((preds > 0).long() == discriminator_y.long()).float().mean().item()

            # Train classifier
            set_requires_grad(target_model, requires_grad=True)
            set_requires_grad(discriminator, requires_grad=False)
            for _ in range(args.k_clf):
                _, (target_x, _) = next(batch_iterator)
                target_x = target_x.to(device)
                target_features = target_model(target_x).view(target_x.shape[0], -1)

                # flipped labels
                discriminator_y = torch.ones(target_x.shape[0], device=device)

                preds = discriminator(target_features).squeeze()
                loss = criterion(preds, discriminator_y)

                target_optim.zero_grad()
                loss.backward()
                target_optim.step()

        mean_loss = total_loss / (args.iterations * args.k_disc)
        mean_accuracy = total_accuracy / (args.iterations * args.k_disc)
        tqdm.write(f'EPOCH {epoch:03d}: discriminator_loss={mean_loss:.4f}, '
                   f'discriminator_accuracy={mean_accuracy:.4f}')

        # Create the full target model and save it
        clf.feature_extractor = target_model
        torch.save(clf.state_dict(), Path(args.model_path).parent / 'adda.pt')

    Path(source_manifest).unlink()
    Path(target_manifest).unlink()


def main(args, class_names, label_func, metrics):
    init_seed(args)
    Path(args.model_path).parent.mkdir(exist_ok=True, parents=True)

    # init setting
    classes = [i for i in range(len(class_names))]
    device = init_device(args)
    eeg_conf = set_eeg_conf(args)
    model = set_model(args, classes, eeg_conf, device)

    src_manifests = args.source_manifests.split(',')
    target_manifests = args.target_manifests.split(',')

    adda(args, model, eeg_conf, label_func, class_names, device, src_manifests, target_manifests)


if __name__ == '__main__':
    args = add_adda_args(train_args()).parse_args()
    class_names = ['interictal', 'preictal']
    from eeglibrary.src import Metric

    metrics = [Metric('loss', initial_value=1000, inequality='less', save_model=True),
               # Metric('recall'),
               # Metric('far')
               ]

    def label_func(path):
        return path.split('/')[-2].split('_')[2]

    main(args, class_names, label_func, metrics)
