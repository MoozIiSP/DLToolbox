import copy

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

module = nn.Sequential(
    nn.Conv2d(1, 2, 3, 1, 1),
    nn.Conv2d(2, 2, 3, 1, 1),
    nn.Conv2d(2, 1, 1)
)

h = 0

class HookTriggered(Exception):
    def __init__(self, message):
        super(HookTriggered, self).__init__(message)


def forward_hook(module, input, output):
    print(f'{input[0].shape} -> {output.shape}')
    if not isinstance(module, nn.Sequential):
        print(module.weight.shape)
    else:
        print('the module is Sequential')
    # print(module._buffers['buf'])
    # raise HookTriggered('The Hook triggered and stop forward and backward')


def backward_hook(module, grad_input, grad_output):
    fig = plt.figure()
    print(f'{grad_input[2].shape} -> ({len(grad_output)}) {grad_output[0].shape}')
    if not isinstance(module, nn.Sequential):
        print(module.weight.shape)
    else:
        print('the module is Sequential')
    # plt.hist(grad_output[0].flatten(1, -1)[0].numpy())
    # plt.savefig('grad_hist.png')
    # raise HookTriggered('The Hook triggered and stop forward and backward')


# inject buffer, parameter and hook into module
# module.register_forward_hook(forward_hook)
module[0].register_forward_hook(forward_hook)
module[0].register_buffer('buf', torch.ones([1, 10]))
module[2].register_backward_hook(backward_hook)

# add submodule into the module, but do not better to add it into the non-module module. It's will be strange.
# module[0].add_module('conv_i', nn.Conv2d(32, 32, 3, 1, 1))
# print(module[0]._modules)

# apply fn - execute a function to all parameter of module
# typically apply an initialization strategy to every module of the net by batch-by-batch
# def fn(m):
#     if m == nn.Conv2d:
#         print(m.weight.shape)
# module.apply(fn)

try:
    input = torch.randn([10, 1, 5, 5])
    x = module(input)
except HookTriggered as e:
    print(e)
criterion = nn.L1Loss()
loss = criterion(x, input)
loss.backward()
print(loss.grad, module[2].weight.grad)

x = module[0](x)
x = module[1](x)
grad = torch.zeros([10, 2, 5, 5])
grad[:, 0, :, :] = 1
x.backward(grad, retain_graph=True)

# print('====')
# for p in module.parameters():
#     print(p.shape)
# print(module[2].weight.grad)
# print('====')
# for p in module.parameters():
#     p.grad /= torch.sqrt(torch.mean(p.grad ** 2) + 1e-5)
# print(module[2].weight.grad)

