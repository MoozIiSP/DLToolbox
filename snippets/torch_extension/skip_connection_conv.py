import torch
import torch.nn as nn


class skip_conv1x1(nn.Module):
    def __init__(self, inp):
        super(skip_conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels=inp, out_channels=inp // 2,
                              kernel_size=1, stride=1, padding=0, bias=False)

        # initialize their weight to skip connection
        with torch.no_grad():
            w = [[1 if in_dim == out_dim else 0
                  for in_dim in range(inp // 2)] * 2
                 for out_dim in range(inp // 2)]
            self.conv.weight = torch.nn.Parameter(torch.FloatTensor(w).reshape(inp // 2, inp, 1, 1))

    def forward(self, tensors):
        assert tensors[0].shape[1] == tensors[1].shape[1], \
            f"tensor1.shape[1] != tensor2.shape[1]: " \
            f"{tensors[0].shape[1]} vs {tensors[1].shape[1]}"
        return self.conv(torch.cat((tensors[0], tensors[1]), dim=1))


x1 = torch.ones((1, 16, 5, 5))
x2 = torch.zeros((1, 8, 5, 5))

conv = skip_conv1x1(32)

y = conv((x1, x2))