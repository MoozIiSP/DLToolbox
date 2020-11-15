import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import AccuracyCallback
from catalyst import utils

# toolbox root directary
sys.path.append(os.path.abspath('../..'))
from toolbox.core.datasets.imagenet import ImageNetK

def main(args):
    logdir = "./logdir"
    num_epochs = 42

    # detect gpu
    device = utils.get_device()
    utils.fp
    print(f"device: {device}")

    # dataset
    trainset = ImageNetK(
        '/run/media/mooziisp/仓库/datasets/Kaggle-ILSVRC/ILSVRC',
        split='train',
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()])
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=64,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    loaders = {
        "train": trainloader
    }

    # define net
    net = models.resnet18(
        pretrained=False,
        num_classes=1000
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    # trainer
    runner = SupervisedRunner(device=device)
    runner.train(
        model=net,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir=logdir,
        callbacks=[
            AccuracyCallback(num_classes=1000)
        ],
        num_epochs=num_epochs,
        verbose=True
    )


if __name__ == '__main__':
    main(None)
