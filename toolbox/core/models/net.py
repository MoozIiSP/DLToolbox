from __future__ import absolute_import

import torch
import torch.nn as nn

import toolbox.core.nn as tnn


__all__ = ['NetA']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# Network in Network Modules
class BiBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BiBlock, self).__init__()
        self.conv1 = conv1x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2_1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2_2 = conv3x3(inplanes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(2*inplanes, planes, stride)
        self.birelu = tnn.BiReLU(inplace=True, concat=False)
        self.prelu = nn.PReLU()
        self.downsample = downsample

        # init
        torch.nn.init.normal_(self.conv1.weight, 0.99, 1.01)
        torch.nn.init.normal_(self.conv3.weight, 0.99, 1.01)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        pout, nout = self.birelu(out)
        out1 = self.conv2_1(pout)
        out1 = self.bn2(out1)
        out2 = self.conv2_2(nout)
        out2 = self.bn2(out2)
        out = self.conv3(torch.cat([out1, out2], dim=1))
        out = self.bn3(out)

        # if self.downsample is not None:
        #     identity = self.downsample(x)

        out += identity
        out = self.prelu(out)

        return out


class NetA(nn.Module):
    def __init__(
            self,
            num_classes,
            inplanes=3,
            kernel_width=64,
            sppool_divs=(4, 2, 1),
    ):
        super(NetA, self).__init__()
        self.kernel_width = kernel_width

        self.feature = nn.Sequential(
            nn.Conv2d(inplanes, kernel_width,
                      kernel_size=7, stride=4, padding=2),
            self._make_layer(BiBlock, kernel_width, kernel_width, 1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self._make_layer(BiBlock, kernel_width, kernel_width, 1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self._make_layer(BiBlock, kernel_width, kernel_width, 3),
            nn.MaxPool2d(kernel_size=2, stride=1)
        )
        self.pool = tnn.SPPool(sppool_divs, pooling_type='adaptiva_avgpool')
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(kernel_width * sum(map(lambda x: x ** 2, sppool_divs)), 4096),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, dilate=False):
        layers = []
        layers.append(block(inplanes, planes, stride))
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x
