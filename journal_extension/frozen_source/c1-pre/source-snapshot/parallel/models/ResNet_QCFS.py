### original authors' implementation

import torch.nn as nn
import torch
import numpy as np
import sys
sys.path.append("..")
from modules import QCFS

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, T, in_channels, out_channels, is_cab, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            QCFS(in_ch=out_channels, t=T, is_cab=is_cab),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
        self.relu = QCFS(in_ch=out_channels * BasicBlock.expansion, t=T, is_cab=is_cab)

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        return self.relu(x)


class ResNet(nn.Module):
    def __init__(self, T, block, num_block, is_cab, num_classes=1000):
        super().__init__()
        self.in_channels = 64
        self.T = T
        self.is_cab = is_cab
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            QCFS(in_ch=64, t=T, is_cab=is_cab))
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.T, self.in_channels, out_channels, self.is_cab, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = torch.flatten(output, 1)
        output = self.fc(output)
        return output


class ResNet4Cifar(nn.Module):
    def __init__(self, T, block, num_block, is_cab, num_classes=10):
        super().__init__()
        self.in_channels = 16
        self.T = T
        self.is_cab = is_cab
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            QCFS(in_ch=16, t=T, is_cab=is_cab))
        self.conv2_x = self._make_layer(block, 16, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 32, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 64, num_block[2], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.T, self.in_channels, out_channels, self.is_cab, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.avg_pool(output)
        output = torch.flatten(output, 1)
        output = self.fc(output)
        return output


def resnet18_qcfs(T, is_cab, num_classes):
    return ResNet(T, BasicBlock, [2, 2, 2, 2], is_cab, num_classes=num_classes)

def resnet20_qcfs(T, is_cab, num_classes):
    return ResNet4Cifar(T, BasicBlock, [3, 3, 3], is_cab, num_classes=num_classes)

def resnet34_qcfs(T, is_cab, num_classes):
    return ResNet(T, BasicBlock, [3, 4, 6, 3], is_cab, num_classes=num_classes)
