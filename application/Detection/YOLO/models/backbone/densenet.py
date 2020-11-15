import torch
import torch.nn as nn
from torchvision.models.densenet import *
from torchvision.models.densenet import model_urls
import torchvision.models.densenet as dn


class DenseNetBackbone(DenseNet):
    def __init__(self, index, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, memory_efficient=False):
        super(DenseNetBackbone, self).__init__(growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, memory_efficient=False)
        # block
        self.classifier = None
        # cache
        self.index = index
        self.cache = {}

    def forward(self, x):
        x = self.features[:self.index[0]+1](x)
        self.cache['scale1'] = x
        x = self.features[self.index[0]+1:self.index[1]+1](x)
        self.cache['scale2'] = x
        x = self.features[self.index[1]+1:](x)

        return x


def _densenet(index, arch, growth_rate, block_config, num_init_features, pretrained, progress,
              **kwargs):
    model = DenseNetBackbone(index, growth_rate, block_config, num_init_features, **kwargs)
    if pretrained:
        dn._load_state_dict(model, model_urls[arch], progress)
    return model


def densenet121(pretrained=False, progress=True, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet([6, 8], 'densenet121', 32, (6, 12, 24, 16), 64, pretrained, progress,
                     **kwargs)


if __name__ == '__main__':
    net = densenet121()
    print(net)
    print(net(torch.randn([1, 3, 512, 512])).shape)
    print(net.cache['scale1'].shape, net.cache['scale2'].shape)
