import torch
import torch.nn as nn


def inspect_gradient(net):
    for tag, val in net.named_parameters():
        print(tag, val.grad.sum())
