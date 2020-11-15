import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiReLU(nn.Module):

    __constants__ = ['inplace']

    def __init__(self, inplace=False, concat=True):
        super(BiReLU, self).__init__()
        self.inplace = inplace
        self.concat = concat

    def forward(self, input):
        pos = F.relu(input, self.inplace)
        neg = -F.relu(-input, self.inplace)
        if self.concat:
            return torch.cat([pos, neg], dim=1)
        else:
            return pos, neg

    def extra_repr(self):
        inplace_str = 'inplace=True' if self.inplace else ''
        inplace_str += 'concat=True' if self.concat else ''
        return inplace_str