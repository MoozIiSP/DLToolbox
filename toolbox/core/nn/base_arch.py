import torch
import torch.nn as nn
from torch.nn.modules.linear import Identity


# TODO:
class Nested(nn.Sequential):
    def __init__(self, seq1, nested, seq2) -> None:
        super(Nested, self).__init__(
            seq1,
            nested,
            seq2
        )


class NestedWPath(nn.Module):
    def __init__(self, seq1, nested, seq2, path=None) -> None:
        super(NestedWPath, self).__init__()
        self.seq1 = seq1
        self.nested = nested
        self.seq2 = seq2
        self.path = path

    def forward(self, x):
        identity = x

        out = self.seq1(x)
        out = self.nested(x)

        # TODO: deal with transfrom data from nested to seq2
        if self.path:
            pass
        else:
            out = self.seq2(x)
            out += identity

        return out


# TODO:
# self-ref architecture just nested sub-architecture.
class SelfArch(nn.Module):
    def __init__(self, seq) -> None:
        super(SelfArch, self).__init__()
        self.seq = seq

    def forward(self, x):
        identity = x

        out = self.seq(x)
        out += identity
        return out


if __name__ == "__main__":
    base = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1))
    for i in range(3):
        base = SelfArch(base)
    print(base)

    mod = base
    print(mod(torch.randn(1, 32, 64, 64)))
