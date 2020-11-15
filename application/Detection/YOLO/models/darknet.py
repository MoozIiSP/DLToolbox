from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


__all__ = ['darknet19', 'darknet53', 'darknet53backbone']


def conv3x3(in_planes, out_planes, stride=1, padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, bias=False)


def conv1x1(in_planes, out_planes, stride=1, padding=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=padding, bias=bias)


# FIXME rename to ConvBlock?
class ConvBlock(nn.Module):
    def __init__(self, conv, inplanes, planes, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = conv(inplanes, planes, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(planes)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leakyrelu(x)
        return x


class DarkResidualBlock(nn.Module):
    def __init__(self, inplanes, planes, downsample=None):
        super(DarkResidualBlock, self).__init__()
        self.downsample = downsample
        self.conv1 = ConvBlock(conv1x1, inplanes, planes, stride=1, padding=0)
        self.conv2 = ConvBlock(conv3x3, planes, planes * 2, stride=1, padding=1)

    def forward(self, x):
        if self.downsample:
            x = self.downsample(x)

        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        out += identity
        return out


class DarkNet(nn.Module):
    def __init__(self, inplances, block, layers, downsample='convolution', backbone=False):
        super(DarkNet, self).__init__()
        # misc
        self.downsample = downsample
        self.backbone = backbone
        self.cache = {}
        # build
        self.conv1 = ConvBlock(conv3x3, inplances, 32, 1, 1)  # conv3x3(inplances, 32, stride=1, padding=1)
        self.layer1 = self._make_layer(block, layers[0], 32)
        self.layer2 = self._make_layer(block, layers[1], 64)
        self.layer3 = self._make_layer(block, layers[2], 128)
        self.layer4 = self._make_layer(block, layers[3], 256)
        self.layer5 = self._make_layer(block, layers[4], 512)
        # darknet19 conv1x1 1024 -> 1000
        if not self.backbone:
            self.avgpool = nn.AdaptiveAvgPool2d((8, 8))
            self.fc = nn.Linear(8*8*1024, 1000)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)

        x = self.layer3(x)
        self.cache['scale1'] = x
        x = self.layer4(x)
        self.cache['scale2'] = x
        x = self.layer5(x)

        if not self.backbone:
            x = self.avgpool(x)
            x = self.fc(x.flatten(1, -1))
        return x

    def _make_layer(self, block, blocks, planes):
        # downsample
        if self.downsample == 'maxpool':
            downsample = nn.MaxPool2d(2, 2)
        else:
            downsample = nn.Sequential(OrderedDict({
                'conv': conv3x3(planes, planes * 2, stride=2, padding=1),
                'bn': nn.BatchNorm2d(planes * 2),
                'leakyrelu': nn.LeakyReLU(0.1, inplace=True),
            }))

        layers = []
        layers.append(block(planes*2, planes, downsample))
        for i in range(1, blocks):
            layers.append(block(planes * 2, planes))
        return nn.Sequential(*layers)


def darknet53(inplances=3, pretrained=False):
    net = DarkNet(inplances=inplances, block=DarkResidualBlock,
                  layers=[1, 2, 8, 8, 4], backbone=backbone)
    if pretrained:
        # FIXME need a path to pretrained weights
        net.load_state_dict(torch.load('weights/darknet53.pt'), strict=False)
        print('loaded darknet53 weights')
    return net


# TODO
def darknet19(inplances=3, pretrained=False, backbone=False):
    net = DarkNet(inplances=inplances, block=ConvBlock,
                  layers=[1, 3, 3, 5, 5], backbone=backbone)
    return net


def darknet53backbone(inplances=3, pretrained=False):
    net = DarkNet(inplances=inplances, block=DarkResidualBlock,
                  layers=[1, 2, 8, 8, 4], backbone=True)
    if pretrained:
        # FIXME need a path to pretrained weights
        net.load_state_dict(torch.load('weights/darknet53.pt'), strict=False)
        print('loaded darknet53 weights')
    return net


if __name__ == '__main__':
    net = darknet53(inplances=3)
    print(net)
    print(net(torch.randn([1, 3, 256, 256])).shape)

    # convert
    # from toolbox.application.Detection.YOLO.utils.convert_weights import *
    # state_dict = copy_weight(net, torch.load('darknet53-diy.pth'))
    # torch.save(state_dict, 'darknet53.pt')

    net = darknet53(inplances=3, pretrained=True, backbone=True)
