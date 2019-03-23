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
    def __init__(self, features, in_features, n_labels=2):
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

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.softmax(x)
        return x


def make_layers(cfg, in_channel=1):
    layers = []
    for i, (channel, kernel_size, stride, padding) in enumerate(cfg):
        if channel == 'M':
            layers += [nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)]
        else:
            conv2d = nn.Conv2d(in_channel, channel, kernel_size=kernel_size, stride=stride, padding=padding)
            layers += [conv2d, nn.BatchNorm2d(channel), nn.ReLU(inplace=True)]
        in_channel = channel
    return nn.Sequential(*layers)


def cnn_v1(in_channel=1, n_labels=2):
    cfg = [(32, (2, 4), (1, 3), (1, 2)),
           (64, (2, 4), (1, 3), (1, 1)),
           (128, (2, 3), (1, 2), (1, 1)),
           (256, (4, 4), (2, 3), (1, 2))]
    model = CNN(make_layers(cfg, in_channel), in_features=256 * 9 * 8, n_labels=n_labels)

    return model


