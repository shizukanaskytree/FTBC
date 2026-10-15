import torch.nn as nn
import torch
import sys
sys.path.append("..")
from modules import QCFS


cfg = {
    'VGG16': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 256, 'M'],
        [512, 512, 512, 'M'],
        [512, 512, 512, 'M']
    ]
}


class VGG(nn.Module):
    def __init__(self, T, vgg_name, is_cab, num_classes, dropout):
        super(VGG, self).__init__()
        self.init_channels = 3
        self.T = T
        self.is_cab = is_cab
        self.layer1 = self._make_layers(cfg[vgg_name][0], dropout)
        self.layer2 = self._make_layers(cfg[vgg_name][1], dropout)
        self.layer3 = self._make_layers(cfg[vgg_name][2], dropout)
        self.layer4 = self._make_layers(cfg[vgg_name][3], dropout)
        self.layer5 = self._make_layers(cfg[vgg_name][4], dropout)
        if num_classes == 1000:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512*7*7, 4096),
                QCFS(t=T, dim=3),
                nn.Dropout(dropout),
                nn.Linear(4096, 4096),
                QCFS(t=T, dim=3),
                nn.Dropout(dropout),
                nn.Linear(4096, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512, 4096),
                QCFS(t=T, dim=3),
                nn.Dropout(dropout),
                nn.Linear(4096, 4096),
                QCFS(t=T, dim=3),
                nn.Dropout(dropout),
                nn.Linear(4096, num_classes)
            )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def _make_layers(self, cfg, dropout):
        layers = []
        for x in cfg:
            if x == 'M':
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(self.init_channels, x, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(x))
                layers.append(QCFS(in_ch=x, t=self.T, is_cab=self.is_cab))
                layers.append(nn.Dropout(dropout))
                self.init_channels = x
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.classifier(out)
        return out


def vgg16_qcfs(T, is_cab, num_classes, dropout=0.):
    return VGG(T, 'VGG16', is_cab, num_classes, dropout)
