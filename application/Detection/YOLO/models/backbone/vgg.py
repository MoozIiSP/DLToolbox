import torch
import torch.nn as nn
from torchvision.models.vgg import *
from torchvision.models.vgg import make_layers, cfgs, model_urls
from torchvision.models.utils import load_state_dict_from_url


class VGGBackbone(VGG):
    def __init__(self, index, features, num_classes=1000, init_weights=True):
        super(VGGBackbone, self).__init__(features, num_classes=1000, init_weights=True)
        # block
        self.classifier = None
        self.avgpool = None
        # cache
        self.cache = {}
        self.index = index

    def forward(self, x):
        x = self.features[:self.index[0]](x) # 20
        self.cache['scale1'] = x
        x = self.features[self.index[0]:self.index[1]](x) # 20:27
        self.cache['scale2'] = x
        x = self.features[self.index[1]:](x) # 27:

        return x


def _vgg(index, arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGGBackbone(index, make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


def vgg11(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg([15, 20], 'vgg11', 'A', False, pretrained, progress, **kwargs)


def vgg11_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg([14, 21], 'vgg11_bn', 'A', True, pretrained, progress, **kwargs)


def vgg13(pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg([19, 24], 'vgg13', 'B', False, pretrained, progress, **kwargs)


def vgg13_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg([20, 27], 'vgg13_bn', 'B', True, pretrained, progress, **kwargs)


if __name__ == '__main__':
    net = vgg11()
    print(net)
    print(net(torch.randn([1, 3, 512, 512])).shape)
    print(net.cache['scale1'].shape, net.cache['scale2'].shape)
