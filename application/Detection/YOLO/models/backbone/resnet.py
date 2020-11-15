import torch
import torch.nn as nn
from torchvision.models.resnet import *
from torchvision.models.resnet import BasicBlock, Bottleneck, model_urls
from torchvision.models.utils import load_state_dict_from_url


class ResnetBackbone(ResNet):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResnetBackbone, self).__init__(
            block, layers, num_classes=1000, zero_init_residual=False,
            groups=1, width_per_group=64, replace_stride_with_dilation=None,
            norm_layer=None)
        # block
        self.avgpool = None
        self.fc = None
        self.cache = {}

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        self.cache['scale1'] = x
        x = self.layer3(x)
        self.cache['scale2'] = x
        x = self.layer4(x)

        return x


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResnetBackbone(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                                            progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


if __name__ == '__main__':
    net = resnet18()
    print(net(torch.randn([1, 3, 512, 512])).shape)
    print(net.cache['scale1'].shape, net.cache['scale2'].shape)
