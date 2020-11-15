import torch
import torch.nn as nn
import torch.optim as optim


# refer to _ConvNd
class CustomModule(nn.Module):
    def __init__(self) -> None:
        super(CustomModule, self).__init__()
        self.weight = None

    def forward(self, x):
        if not self.weight:
            self.weight = nn.Parameter(torch.randn(1, 1, 3, 3))
        return x * self.weight


module = CustomModule()

def forward_pre_hook(mod, inp):
    print(f"This is forward pre hook. {id(mod)}")
    print(f"input: {id(inp)}")


def forward_hook(mod, inp, out):
    print(f"This is forward hook. {id(mod)}")
    print(f"input: {id(inp)}")
    print(f"output: {id(out)}")


def backward_hook(mod, inp, out):
    print(f"This is backward hook. {id(mod)}")
    print(f"input: {id(inp)}, {inp}")
    print(f"output: {id(out)}, {out}")


module.register_forward_pre_hook(forward_pre_hook)
module.register_forward_hook(forward_hook)
module.register_backward_hook(backward_hook)
module.requires_grad = True
# for i in range(2):
#     module[i].register_forward_pre_hook(forward_pre_hook)
#     module[i].register_forward_hook(forward_hook)
#     module[i].register_backward_hook(backward_hook)
#     module.requires_grad = True


inp = torch.randn(1, 1, 3, 3)
x = module(inp)
print(x)

print(module.weight)
criterion = nn.L1Loss()
optimizer = optim.SGD(module.parameters(), lr=1e-2)

optimizer.zero_grad()
loss = criterion(x, inp)
loss.backward()
optimizer.step()
print(module.weight)
# x.backward(torch.ones(1, 1, 3, 3), retain_graph=True)
