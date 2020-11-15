import torch
import torch.nn as nn
from torchvision.models.squeezenet import *
from torchvision.models.squeezenet import model_urls
from torchvision.models.utils import load_state_dict_from_url


class SqueezeNetBackbone(SqueezeNet):
    def __init__(self, version='1_0', num_classes=1000):
        super(SqueezeNetBackbone, self).__init__(version='1_0', num_classes=1000)
        # block
        self.classifier = None
        # cache
        self.cache = {}

    def forward(self, x):
        x = self.features[:6](x)
        self.cache['scale1'] = x
        x = self.features[6:11](x)
        self.cache['scale2'] = x
        x = self.features[11:](x)

        return x


def _squeezenet(version, pretrained, progress, **kwargs):
    model = SqueezeNetBackbone(version, **kwargs)
    if pretrained:
        arch = 'squeezenet' + version
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


def squeezenet1_0(pretrained=False, progress=True, **kwargs):
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _squeezenet('1_0', pretrained, progress, **kwargs)


def squeezenet1_1(pretrained=False, progress=True, **kwargs):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _squeezenet('1_1', pretrained, progress, **kwargs)


if __name__ == '__main__':
    net = squeezenet1_1()
    print(net)
    print(net(torch.randn([1, 3, 512, 512])).shape)
    print(net.cache['scale1'].shape, net.cache['scale2'].shape)
