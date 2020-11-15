from __future__ import absolute_import

import torch
import torch.nn as nn

import toolbox.core.nn as tnn

__all__ = ['AlexNetA', 'AlexNetB', 'AlexNetC']


class AlexNetA(nn.Module):
    """Alexnet with SPPool"""

    def __init__(
            self,
            num_classes,
            inplanes=3,
            kernel_width=32,
            sppool_divs=(2, 1)
    ):
        super(AlexNetA, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                inplanes, kernel_width,
                kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(
                kernel_width, kernel_width * 3,
                kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(
                kernel_width * 3, kernel_width * 6,
                kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                kernel_width * 6, kernel_width * 4,
                kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                kernel_width * 4, kernel_width * 4,
                kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.sppool = tnn.SPPool(sppool_divs, 'adaptive_avgpool')
        self.classifier = nn.Sequential(
            nn.Dropout(),
            # div-256d
            nn.Linear(kernel_width * 4 * sum(map(lambda x: x ** 2, sppool_divs)), 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        """

        :param x: 

        """
        x = self.features(x)
        x = self.sppool(x)
        x = self.classifier(x)
        return x


class AlexNetB(nn.Module):
    """Alexnet with BiReLU"""

    def __init__(
            self,
            num_classes,
            inplanes=3,
            kernel_width=32
    ):
        super(AlexNetB, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                inplanes, kernel_width,
                kernel_size=11, stride=4, padding=2),
            tnn.BiReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(
                2 * kernel_width, kernel_width * 3,
                kernel_size=5, padding=2),
            tnn.BiReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(
                2 * kernel_width * 3, kernel_width * 6,
                kernel_size=3, padding=1),
            tnn.BiReLU(inplace=True),
            nn.Conv2d(
                2 * kernel_width * 6, kernel_width * 4,
                kernel_size=3, padding=1),
            tnn.BiReLU(inplace=True),
            nn.Conv2d(
                2 * kernel_width * 4, kernel_width * 4,
                kernel_size=3, padding=1),
            tnn.BiReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            # div-256d
            nn.Linear(2 * kernel_width * 4 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        """

        :param x:

        """
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class AlexNetC(nn.Module):
    """Alexnet with LeakyReLU"""

    def __init__(
            self,
            num_classes,
            inplanes=3,
            kernel_width=32
    ):
        super(AlexNetC, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                inplanes, kernel_width,
                kernel_size=11, stride=4, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(
                kernel_width, kernel_width * 3,
                kernel_size=5, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(
                kernel_width * 3, kernel_width * 6,
                kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(
                kernel_width * 6, kernel_width * 4,
                kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(
                kernel_width * 4, kernel_width * 4,
                kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(kernel_width * 4 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        """

        :param x:

        """
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
