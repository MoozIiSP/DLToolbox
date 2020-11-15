# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# %%
sample = np.array([3.])
xs = np.arange(0, 10)
ys = 9 * xs * xs + 3 * 2 * xs
print(xs, ys)

x = torch.tensor(sample)
w1 = torch.tensor([1e-5], requires_grad=True)
z = x * w1 * 3

w2 = torch.tensor([1e-5], requires_grad=True)
z2 = x * x * w2 + z


optimizer = optim.SGD([{'params': w1}, {'params': w2}], lr=1e-1)
# optimizer.zero_grad()

# backward函数的工作原理
#  当未提供grad_tensor的时候，默认行为是[None] * len(tensors)来得到梯度tensors，
#  接着才是_make_grads(tensors, grad_tensors)
# When grad is [None], PyTorch will call function _make_grads to multiply tensors by same length identity matrix grad.
# If grad provided, then multiply tensors with the grad, and backward to update grad. So, default behavior typically is
# directly backward to pass the value of the loss. And you can pass a mask grad to update the weight of the specific
# layer.
# Update weight by theta := theta - alpha * gradient of theta, so it is global update.
# Optimizer will control every layer to update its weight by specific learning rate.

# z2.backward(torch.tensor([1.]))
#
# print(x.requires_grad, z.requires_grad, w1.requires_grad)
# print(x, x.grad)
# print(w1, w1.grad)
# print(z2, z2.grad)
#
# optimizer.step()
# print(w1, w1.grad)
# print(z2, z2.grad)

lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [50, 100, 150, 200, 250, 300, 350, 400, 450, 500])

for i in range(500):
    x = torch.tensor(np.random.randint(-10, 10))
    z = x * x * w2 + x * w1 * 3
    y = torch.tensor(9 * x.item() * x.item() + 3 * 2 * x.item(), requires_grad=False)
    loss = torch.abs(z - y)
    loss.backward()
    if i % 2 == 0:
        # Accumulates gradient before each step
        optimizer.step()
        optimizer.zero_grad()
    print('lr', optimizer.state_dict()['param_groups'][0]['lr'])
    print(f'I{i} Loss {loss.item()}', x.item())
    print(f'\t w1 {w1.item()} w1.grad {w1.grad.item()}')
    print(f'\t w2 {w2.item()} w2.grad {w2.grad.item()}')
    lr_scheduler.step(i)

# %%
