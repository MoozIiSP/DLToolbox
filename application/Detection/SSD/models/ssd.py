import torch
import torch.nn as nn


class SSDSegmentation(nn.Module):
    def __init__(self):
        super(SSDSegmentation, self).__init__()
        self.backbone = None
        self.classifiers = None
        self.head = None

    def forward(self, x):
        x = self.backbone(x)

        out = [x]
        for classifier in self.classifiers:
            out.append(classifier(out[-1]))

        return x


class SSDHeadBlock(nn.Sequential):
    pass


class SSDExtraFeatLayer(nn.Sequential):
    def __init__(self):
        super(SSDExtraFeatLayer, self).__init__(
            nn.Conv2d(1024, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 1),
            nn.ReLU(inplace=True),
        )
