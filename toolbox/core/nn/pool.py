from __future__ import absolute_import

import torch
import torch.nn as nn
import numpy as np


class SPPool(nn.Module):
    """Spatial Pyramid Pooling from
    `"Spatial Pyramid Pooling in Deep Convolutional Networks for Visual
    Recognition"<https://arxiv.org/abs/1406.4729>`_

      Args:
        pass
    """

    __constants__ = ['output_size', 'pooling_type']

    def __init__(self, output_size, pooling_type='max'):
        """
        output_size: the height and width after spp nn
        pooling_type: the type of pooling
        """
        super(SPPool, self).__init__()
        self.output_size = output_size
        self.pooling_type = pooling_type
        self.spp = None

    def forward(self, x):
        """

        :param x:

        """
        N, C, H, W = x.size()

        sppool = []
        for div in self.output_size:
            assert div <= H or div <= W, \
                'div={} vs (H={}, W={}): too big'.format(div, H, W)
            # the size of pooling window
            size = np.ceil(H / div)
            # the strides of pooling
            stride = np.floor(H / div)
            if self.pooling_type == 'maxpool':
                self.spp = nn.MaxPool2d(
                    kernel_size=size, stride=stride)
            elif self.pooling_type == 'avgpool':
                self.spp = nn.AvgPool2d(
                    kernel_size=size, stride=stride)
            elif self.pooling_type == 'adaptive_maxpool':
                self.spp = nn.AdaptiveMaxPool2d(div)
            else:
                self.spp = nn.AdaptiveAvgPool2d(div)
            sppool.append(self.spp(x))
            
        x = sppool[0].view(sppool[0].size(0), -1)
        for _ in sppool[1:]:
            x = torch.cat([x, _.view(_.size(0), -1)], dim=1)

        return x

    def extra_repr(self):
        """ """
        return 'output_size={output_size}, pooling_type={pooling_type}'.format(**self.__dict__)
