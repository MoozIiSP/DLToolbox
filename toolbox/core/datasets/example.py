import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


class ExampleDataset:

    def __init__(self):
        print('This is init method of ExampleDataset.')

    def __getitem__(self, index):
        print(f'This is getitem method for return data ang its label. - {index}')
        return torch.zeros(5, 5, 3), torch.ones(1)

    def __len__(self):
        print('This is len method.')
        return 100

    # def __repr__(self):
    #     print('This is repr method.')
    #     return 'repr'

def collate_fn(batch):
    print('This is collate method.')
    return batch


if __name__ == '__main__':
    dataloader = DataLoader(ExampleDataset(), batch_size=8, collate_fn=collate_fn)
    loader = iter(dataloader)

    _ = next(loader)
