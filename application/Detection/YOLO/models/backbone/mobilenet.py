import torch
import torch.nn as nn
from torchvision.models.mobilenet import *
from torchvision.models.mobilenet import model_urls
from torchvision.models.utils import load_state_dict_from_url


class MobileNetBackbone(MobileNetV2):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        super(MobileNetBackbone, self).__init__(
            num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8)
        # block
        self.classifier = None
        # cache
        self.cache = {}

    def forward(self, x):
        x = self.features[:7](x)
        self.cache['scale1'] = x
        x = self.features[7:14](x)
        self.cache['scale2'] = x
        x = self.features[14:17](x)

        return x


def mobilenet_v2(pretrained=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetBackbone(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                                          progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


if __name__ == '__main__':
    net = mobilenet_v2()
    print(net)
    print(net(torch.randn([1, 3, 512, 512])).shape)
    print(net.cache['scale1'].shape, net.cache['scale2'].shape)
