import sys
sys.path.append("..")

import torch
import torch.nn as nn
import torch.optim as optim
from utils import freeze
from torchvision import models, transforms, datasets

from classifier.NetTrainer import Trainer
from classifier.__main__ import plot_statictis

if __name__ == "__main__":
    # Hyper-parameters Setting
    LR = 1e-3
    BATCH_SIZE = 8
    NUM_CLASSES = 10
    WEIGHT_DECAY = 5e-5

    print(f"Currect Memory Allocated: {torch.cuda.memory_allocated()}")

    # Create Model
    net = models.resnet34(pretrained=True)

    print(f"Currect Memory Allocated: {torch.cuda.memory_allocated()}")

    # Transfer
    filter = [
        # 'resnet18.layer1',
        # 'resnet18.layer2',
        # 'resnet18.layer3',
        'resnet34.layer4',
        'resnet34.fc'
    ]
    freeze(net, filter, prefix='resnet34', verbose=True)
    net.fc = nn.Linear(in_features=512, out_features=10, bias=True)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(net.parameters(),
                          LR,
                          momentum=0.9,
                          weight_decay=WEIGHT_DECAY)

    print(f"Currect Memory Allocated: {torch.cuda.memory_allocated()}")

    transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    print(f"Currect Memory Allocated: {torch.cuda.memory_allocated()}")

    # Pytorch datasets
    trainset = datasets.STL10(
        root='./data',
        split='train',
        download=True,
        transform=transform,
    )
    testset = datasets.STL10(root='./data', download=True, transform=transform)

    print(f"Currect Memory Allocated: {torch.cuda.memory_allocated()}")

    print(f"Number of trainset: {len(trainset)}")
    # print(f"Number of validset: {len(validset.targets)}")
    print(f"Number of testset: {len(testset)}")

    trainer = Trainer(net, criterion, optimizer, trainset, testset)

    # Evaluation
    trainer.eval()

    for epoch in range(50):
        # trainer.adjust_learning_rate(lr)

        # train for one epoch
        trainer.train()
        trainer.log['epoch'] += 1

        # evaluate on validation set
        trainer.eval()
        # acc1 = trainer.log['valid_log']['acc1'].avg
        # save best model

    for k in trainer.log['train_log'].__dict__.keys():
        ys = (trainer.log['train_log'].__dict__[k],
              trainer.log['valid_log'].__dict__[k])
        plot_statictis(
            f'{net.__class__.__name__} Training & Validation on {trainset.__class__.__name__}',
            k, ys)
    trainer.save(False)
