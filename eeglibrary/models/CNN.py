import torch

seed = 0
torch.manual_seed(seed)
import math
torch.cuda.manual_seed_all(seed)
import random
random.seed(seed)
import torch.nn as nn
from torchvision import models
from eeglibrary.src import eeg_loader


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


def make_layers(cfg, eeg_conf, dim=2):
    if eeg_conf['spect'] and dim == 2:
        in_channel = eeg_conf['n_elect']
    else:
        in_channel = 1
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


def calc_out_size(cfg, eeg_conf, dim=2, last_shape=False):
    if eeg_conf['spect']:
        n_fft = int(eeg_conf['sample_rate'] * eeg_conf['window_size'])
        t = eeg_conf['sample_rate'] // int(eeg_conf['sample_rate'] * eeg_conf['window_stride']) + 1
        out_sizes = [(1 + n_fft) / 2, t]
        if dim == 3:
            out_sizes = [eeg_conf['n_elect']] + out_sizes
    else:
        out_sizes = [eeg_conf['n_elect'], eeg_conf['sample_rate']*eeg_conf['window_size']]
    # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
    for dim in range(dim):  # height and width if dim=2
        for layer in cfg:
            channel, kernel, stride, padding = layer
            out_sizes[dim] = int(math.floor(out_sizes[dim] + 2 * padding[dim] - kernel[dim]) / stride[dim] + 1)
    out_size = out_sizes[0] * out_sizes[1] * cfg[-1][0] # multiply last channel number
    if last_shape:
        return cfg[-1][0], out_sizes
    return out_size


def cnn_1_16_399(eeg_conf, n_labels=2): # 16 × 399
    cfg = [(32, (2, 4), (1, 3), (1, 2)),
           (64, (2, 4), (1, 3), (1, 1)),
           (128, (2, 3), (1, 2), (1, 1)),
           (256, (4, 4), (2, 3), (1, 2))]
    layers = make_layers(cfg, eeg_conf)
    out_size = calc_out_size(cfg, eeg_conf)
    model = CNN(layers, out_size, n_labels=n_labels)

    return model


def cnn_16_751_751(eeg_conf, n_labels=2):
    cfg = [(32, (4, 2), (3, 2), (0, 1)),
           (64, (4, 2), (3, 2), (2, 1)),
           (128, (4, 2), (3, 2), (1, 0))]
    layers = make_layers(cfg, eeg_conf)
    out_size = calc_out_size(cfg, eeg_conf)
    model = CNN(layers, out_size, n_labels=n_labels)

    return model


def cnn_ftrs_16_751_751(eeg_conf):
    cfg = [
        (32, (4, 2), (3, 2), (0, 1)),
        (64, (4, 2), (3, 2), (0, 1)),
    ]
    layers = make_layers(cfg, eeg_conf)
    n_channel, (out_height, out_width) = calc_out_size(cfg, eeg_conf, last_shape=True)
    return layers, n_channel * out_height # for RNN input size


def cnn_1_16_751_751(eeg_conf, n_labels=1):
    cfg = [(16, (4, 4, 4), (2, 3, 3), (2, 0, 0)),
           (32, (4, 4, 4), (2, 3, 3), (2, 2, 2)),
           (64, (4, 4, 4), (2, 3, 3), (2, 1, 1)),
           (128, (2, 4, 4), (1, 3, 3), (0, 2, 2))]
    layers = make_layers(cfg, eeg_conf, dim=3)
    out_size = calc_out_size(cfg, eeg_conf, dim=3)
    model = CNN(layers, out_size, n_labels, dim=3)

    return model
