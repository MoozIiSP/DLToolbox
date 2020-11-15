from collections import deque
import torch
import torch.nn as nn
import torch.nn.init as init
import math

from torch.nn.parameter import Parameter


class MLPConv2d(nn.Module):
    pass


# TODO need to be generic
class ConvBNReLU(nn.Sequential):
    def __init__(self, inp, oup, kernel_size=3, stride=1, groups=1):
        # Trick: auto set padding to zero.
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(inp, oup, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
        )


class SeparableConv2d(nn.Module):
    def __init__(self, inp, oup) -> None:
        super(SeparableConv2d, self).__init__()
        self.dwise = nn.Conv2d(inp, inp, kernel_size=3, stride=1, groups=inp)
        self.point = nn.Conv2d(inp, oup, kernel_size=1, stride=1)

    def forward(self, x):
        return self.point(self.dwise(x))


class NestedSeparableConv2d(nn.Sequential):
    def __init__(self, inp, k=1) -> None:
        layer = deque()
        kinp = inp
        for i in range(k):
            print(inp, kinp)
            layer.appendleft(nn.Conv2d(kinp, inp, kernel_size=3, stride=1, groups=inp))
            layer.append(nn.Conv2d(inp, kinp, kernel_size=1, stride=1))
            inp, kinp = kinp, kinp * 2
        super(NestedSeparableConv2d, self).__init__(*layer)


# FIXME: please to refer to _Convkd
class SkipConv1x1(nn.Module):
    """
    After freezing, its behaviour likes a simple add arithmetic operation to add
    two tensor by value-wise.
    """

    def __init__(self, inp, requires_grad=True):
        super(SkipConv1x1, self).__init__()
        assert inp % 2 == 0, f'inp must be even number: {inp}'
        self.weight = None

        # initialize their weight to skip connection
        with torch.no_grad():
            w = [[1 if in_dim == out_dim else 0
                  for in_dim in range(inp // 2)] * 2
                 for out_dim in range(inp // 2)]
            self.weight = Parameter(torch.FloatTensor(w).reshape(inp // 2, inp, 1, 1))

        # freeze or not
        self.weight.requires_grad = requires_grad

    # def reset_parameters(self) -> None:
    #     init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, tensors):
        assert tensors[0].shape[1] == tensors[1].shape[1], \
            f"tensor1.shape[1] != tensor2.shape[1]: " \
            f"{tensors[0].shape[1]} vs {tensors[1].shape[1]}"
        return self.conv(torch.cat((tensors[0], tensors[1]), dim=1))
