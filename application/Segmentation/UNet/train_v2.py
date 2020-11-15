import torch
import torch.nn as nn
import torch.optim as optim
from catalyst.dl.runner import SupervisedRunner

from .models import UNet
from .dataset import CustomSegDataset
from .nn import DiceLoss

logdir = "./logdir"
num_epochs = 42

# Parpered the dataset
dataloader = torch.utils.data.DataLoader(
    CustomSegDataset('/home/mooziisp/GitRepos/unet/data/membrane/train/image'),
    batch_size=1,
    shuffle=True,
    num_workers=2,
    pin_memory=True)

loaders = {
    "train": dataloader
}

# Define the net
net = UNet(in_channels=CustomSegDataset.in_channels,
           out_channels=CustomSegDataset.out_channels)
criterion = DiceLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-4)

runner = SupervisedRunner()

runner.train(
    model=net,
    criterion=criterion,
    optimizer=optimizer,
    loaders=loaders,
    logdir=logdir,
    num_epochs=num_epochs,
    verbose=True
)
