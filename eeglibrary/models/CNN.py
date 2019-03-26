import torch
from torch import nn
import torchvision
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
from random import shuffle
import random
random.seed(seed)
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models, transforms, utils


class vgg_16:
    # (batch, channel, 224, 224)が必須
    def __init__(self, kernel, padding, n_channel, class_names):
        self.model = models.vgg11_bn(pretrained=True)
        n_cnn2_ftrs = self.model.features[0].out_channels
        self.model.features[0] = nn.Conv2d(n_channel, n_cnn2_ftrs, kernel_size=kernel, padding=padding)
        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_ftrs, len(class_names))
        return self.model


class CNN(nn.Module):
    def __init__(self, features, in_features, n_labels=2, dim=2):
        super(CNN, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, n_labels),
        )
        self.softmax = nn.Softmax(dim=-1)
        self.dim = dim

    def forward(self, x):
        if self.dim == 3:
            x = torch.unsqueeze(x, dim=1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.softmax(x)
        return x


def make_layers(cfg, in_channel=1, dim='2'):
    if dim == 2:
        conv_cls, max_pool_cls, batch_norm_cls = nn.Conv2d, nn.MaxPool2d, nn.BatchNorm2d
    elif dim == 3:
        conv_cls, max_pool_cls, batch_norm_cls = nn.Conv3d, nn.MaxPool3d, nn.BatchNorm3d

    layers = []
    for i, (channel, kernel_size, stride, padding) in enumerate(cfg):
        if channel == 'M':
            layers += [max_pool_cls(kernel_size=kernel_size, stride=stride, padding=padding)]
        else:
            conv = conv_cls(in_channel, channel, kernel_size=kernel_size, stride=stride, padding=padding)
            layers += [conv, batch_norm_cls(channel), nn.ReLU(inplace=True)]
        in_channel = channel
    return nn.Sequential(*layers)


def cnn_1_16_399(n_labels=2): # 16 × 399
    cfg = [(32, (2, 4), (1, 3), (1, 2)),
           (64, (2, 4), (1, 3), (1, 1)),
           (128, (2, 3), (1, 2), (1, 1)),
           (256, (4, 4), (2, 3), (1, 2))]
    model = CNN(make_layers(cfg, in_channel=1), in_features=256 * 9 * 8, n_labels=n_labels)

    return model


def cnn_16_751_751(n_labels=2):
    cfg = [(32, (4, 4), (3, 3), (0, 0)),
           (64, (4, 4), (3, 3), (2, 2)),
           (128, (4, 4), (3, 3), (1, 1)),
           (256, (4, 4), (3, 3), (2, 2))]
    model = CNN(make_layers(cfg, in_channel=16), in_features=256 * 10 * 10, n_labels=n_labels)

    return model


def cnn_ftrs_16_751_751(n_labels=2):
    cfg = [
        (32, (4, 4), (3, 3), (0, 0)),
        (64, (4, 4), (3, 3), (2, 2)),
    ]
    out_ftrs = 64 * 12 * 7
    return make_layers(cfg, in_channel=16), out_ftrs


def cnn_1_16_751_751(n_labels=1):
    cfg = [(16, (4, 4, 4), (2, 3, 3), (2, 0, 0)),
           (32, (4, 4, 4), (2, 3, 3), (2, 2, 2)),
           (64, (4, 4, 4), (2, 3, 3), (2, 1, 1)),
           (128, (2, 4, 4), (1, 3, 3), (0, 2, 2))]
    model = CNN(make_layers(cfg, in_channel=1, dim=3), in_features=256 * 10 * 10, n_labels=n_labels, dim=3)

    return model
