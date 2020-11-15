"""Main code from brain-segmentation-pytorch"""
from __future__ import absolute_import

import torch
import torch.optim as optim

from dataset import CustomSegDataset as Dataset
from models import UNet
from nn import DiceLoss


def main():
    # Define the net
    net = UNet(in_channels = Dataset.in_channels, out_channels = Dataset.out_channels) #.cuda()
    criterion = DiceLoss()
    optimizer = optim.Adam(net.parameters(), lr = 1e-4)

    # Parpered the dataset
    dataloader = torch.utils.data.DataLoader(
        Dataset('/home/mooziisp/GitRepos/unet/data/membrane/train/image'),
        batch_size = 1,
        shuffle = True,
        num_workers = 2,
        pin_memory = True)

    # TODO validation
    # TODO acc and loss record
    # TODO intergated with tensorboard
    for epoch in range(100):
        for i, (images, targets) in enumerate(dataloader):
            #images, targets = images.cuda(), targets.cuda()

            preds = net(images)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'{epoch*30+i}it, loss: {loss.item():.3f}')

    return net

            
if __name__ == '__main__':
    net = main()
    # TODO save the net to file
    print('save the net')
    torch.save(net.state_dict(), "unet.pth")
