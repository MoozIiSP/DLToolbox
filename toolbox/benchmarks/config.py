import torch.optim as optim
from torchvision.models import *

from toolbox.core.models import *
from toolbox.core.models.vae import MixtureNet
from toolbox.core.nn import *

# GPU: RTX 2080 8G
EPOCH = 50
BATCH_SIZE = 32
LR = 1e-3
MILESTONES = [15, 45]
NUM_CLASSES = 10

GRADIENT_STAT = False
ARCH_GRAPH = True

SAVE = False


# NET train setting
def call_fn(fn, **kwargs):
    return {"fn": fn, "kwargs": kwargs}


NETS = [
    {
        "net": call_fn(resnet18, num_classes=NUM_CLASSES),
        "criterion": call_fn(nn.CrossEntropyLoss),
        "optimizer": call_fn(optim.Adam)},
    {
        "net": call_fn(LeNet5p, num_classes=NUM_CLASSES),
        "criterion": call_fn(nn.CrossEntropyLoss),
        "optimizer": call_fn(optim.Adam)},
]

VAES = [
    {
        "net": call_fn(MixtureNet),
        "criterion": [call_fn(VAELoss), call_fn(nn.BCELoss)],
        "optimizer": call_fn(optim.Adam, weight_decay=1e-5)}
    # {
    #     "net": call_fn(LeNet5VAE, num_classes=NUM_CLASSES),
    #     "criterion": call_fn(nn.MSELoss),
    #     "optimizer": call_fn(optim.Adam)}
]

MULTI_LOSS_NETS = [
    {
        "net": call_fn(LeNet5wAeA, num_classes=NUM_CLASSES),
        "criterion": [call_fn(nn.BCELoss), call_fn(nn.CrossEntropyLoss)],
        "optimizer": call_fn(optim.Adam)},
    # {
    #     "net": call_fn(LeNet5wAeB, num_classes=NUM_CLASSES),
    #     "criterion": [call_fn(nn.BCELoss), call_fn(nn.CrossEntropyLoss)],
    #     "optimizer": call_fn(optim.Adam)},
    # {
    #     "net": call_fn(LeNet5wAeC, num_classes=NUM_CLASSES),
    #     "criterion": [call_fn(nn.BCELoss), call_fn(nn.CrossEntropyLoss)],
    #     "optimizer": call_fn(optim.Adam)}
]

# Analysis
MEASURE_DATALOAD = False
