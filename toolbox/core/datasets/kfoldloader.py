import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


class kFoldDataset(ImageFolder):
    def __init__(self, root):
        super(kFoldDataset, self).__init__()

    def __getitem__(self, index):
        print(f'This is getitem method for return data ang its label. - {index}')
        return torch.zeros(5, 5, 3), torch.ones(1)

    def __len__(self):
        print('This is len method.')
        return 100
