"""
Author: Soyed Tuhin Ahmed
"""

from binarized_modules import BinarizeLinear, BinarizeConv2d
import torch
import torch.nn as nn


class BinLeNet(nn.Module):
    def __init__(self, c_in=1, num_classes=10):
        super(BinLeNet, self).__init__()

        self.infl_ratio = 1  # Hidden unit multiplier
        self.in_channels = c_in
        self.num_classes = num_classes
        self.drop_prob = 0.3

        self.avgpool = nn.AvgPool2d(2, 2)
        self.features = nn.Sequential(
            BinarizeConv2d(self.in_channels, 8, kernel_size=5, padding=1, bias=True),  # rows= 5*5*1 = 25 cols 8, raise = 3
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(8),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(8, 16, kernel_size=2, padding=1, bias=True),  # rows= 2*2*8 = 32 cols 16, raise = 2
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(16, 150, kernel_size=5, padding=1, bias=True),  # rows= 5*5*16 = 400 cols 150, raise = 2
            nn.BatchNorm2d(150),
            nn.Hardtanh(inplace=True),
        )

        self.soft = nn.Softmax()

        self.classifier = nn.Sequential(
            BinarizeLinear(150 * 5 * 5, 128, bias=True),
            nn.BatchNorm1d(128),
            nn.Hardtanh(inplace=True),

            BinarizeLinear(128, 128, bias=True),
            nn.BatchNorm1d(128),
            nn.Hardtanh(inplace=True),

            BinarizeLinear(128, self.num_classes, bias=True),
            nn.BatchNorm1d(self.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)
        x = self.soft(x)

        return x


class BinMLP(nn.Module):
    def __init__(self, c_in=1, num_classes=10):
        super(BinMLP, self).__init__()

        self.infl_ratio = 1  # Number of neurons multiplier
        self.num_classes = num_classes

        self.soft = nn.Softmax()
        self.neurons = 256 * self.infl_ratio

        self.classifier = nn.Sequential(
            BinarizeLinear(28*28*c_in, self.neurons, bias=True),  # 0
            nn.BatchNorm1d(self.neurons),
            nn.Hardtanh(inplace=True),

            BinarizeLinear(self.neurons, self.neurons, bias=True), # 3
            nn.BatchNorm1d(self.neurons),
            nn.Hardtanh(inplace=True),

            BinarizeLinear(self.neurons, self.neurons, bias=True), # 6
            nn.BatchNorm1d(self.neurons),
            nn.Hardtanh(inplace=True),

            BinarizeLinear(self.neurons, self.neurons, bias=True),  # 9
            nn.BatchNorm1d(self.neurons),
            nn.Hardtanh(inplace=True),

            BinarizeLinear(self.neurons, self.neurons, bias=True),  # 12
            nn.BatchNorm1d(self.neurons),
            nn.Hardtanh(inplace=True),

            BinarizeLinear(self.neurons, self.num_classes, bias=True),
            nn.BatchNorm1d(self.num_classes)
        )

    def forward(self, x):
        x = torch.flatten(x, 1)

        x = self.classifier(x)
        x = self.soft(x)

        return x


