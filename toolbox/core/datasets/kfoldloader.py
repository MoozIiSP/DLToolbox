import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split


class kFoldDataset:
    def __init__(self):
        super(kFoldDataset, self).__init__()
        self.cnt = 0
        self.dataset = [self.cnt for i in range(10)]

    def reset(self):
        self.cnt += 1
        self.dataset = [self.cnt for i in range(10)]

    def __getitem__(self, index):
        print(f'This is getitem method for return data ang its label. - {index}')
        # return torch.zeros(5, 5, 3), torch.ones(1)
        if index == len(self.dataset) - 1:
            self.reset()
        return self.dataset[index]

    def __len__(self):
        print('This is len method.')
        return len(self.dataset)


# def get_kflod_dataset(data_dir, k=1):
#     # Get all key pair of data
#     class_names = os.listdir(data_dir)

#     for class_name in class_names:
#         for fname in os.listdir(os.path.join(data_dir, class_name)):
#             pass
#     # and split into k