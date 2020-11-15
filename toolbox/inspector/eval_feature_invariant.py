# %%
# Convolutional Neural Net almost don't learn anything from the object except some low- or high-level visual feature which has few effects,
# just remember which position the object appeared on the image.
# Here just are setup for the following works.
import json
import torch
import torch.nn as nn

from torchvision import transforms, models
from PIL import Image

# %%
# Load a pretrained model and dataset to evaluate.
model = models.resnet18(pretrained=True)
model.eval()

im = Image.open('cat2.jpg')
labels = json.load(open('imagenet_class_index.json', 'r'))

# %%
# Display image via matplotlib
# TODO we will make this cell to merge into first cell.
import matplotlib.pyplot as plt
import numpy as np


# view this image, need to convert float32 to int32
def view_data(im):
    x = np.asarray(im, dtype=np.uint8)
    if x.shape[0] == 3:
        plt.imshow(np.transpose(x, (1, 2, 0)))  # FIXME
    else:
        plt.imshow(x)


view_data(im)

# %%
# Display histgram of the image
import cv2

cvim = cv2.imread('cat.jpg')


def show_hist(im):
    '''im: Tensor Obejct.'''
    if not type(im) == np.ndarray:
        im = np.transpose(np.asarray(im, dtype=np.uint8), (1, 2, 0))
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([im], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()


show_hist(cvim)
# TODO
show_hist(np.asarray(cvim**0.5, dtype=np.uint8))


# %%
# Magic Transform
def magic_image(im, thrd):
    '''im: PIL Image Object.
    returen: PIL Image Object.'''
    im = np.asarray(im, dtype=np.uint8).copy()  # Get own memory
    ir, ic = im.shape[:2]

    im.flags.writeable = True
    for r in range(ir):
        for c in range(ic):
            if np.sum((im[r, c, :] - np.array([255, 255, 255]))**2) < thrd:
                im[r, c, :] = np.random.rand(3) * 255

    return Image.fromarray(im)


# %%
import random


def salt_and_pepper(im, n, mode='random'):
    # set image to writeable
    im = np.asarray(im, dtype=np.uint8).copy()  # Get own memory
    im.flags.writeable = True

    for k in range(n):
        i = random.randint(0, im.shape[0] - 1)
        j = random.randint(0, im.shape[1] - 1)

        if mode == 'random':
            im[i, j, :] = [255, 255, 255
                           ] if random.random() < 0.5 else [0, 0, 0]
        elif mode == 'salt':
            im[i, j, :] = [255, 255, 255]
        elif mode == 'pepper':
            im[i, j, :] = [0, 0, 0]

    # set image to unwriteable
    im.flags.writeable = False

    return Image.fromarray(im)


# %%
# Image preprocessing - Resize
prep = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor()])

x = prep(im)
x = x.unsqueeze(0)

view_data(x[0] * 255)

v, idx = [x.detach().numpy()[0] for x in nn.Softmax(dim=1)(model(x)).topk(5)]
for prob, pred in zip(v, idx):
    Id, n = labels[str(pred)]
    print(f'{pred:8} - {Id:10} - {prob:.2f} - {n} ')

# %%
# RGB Histgram
show_hist(x[0] * 255)

# %%
# Center Crop
prep = transforms.Compose(
    [transforms.CenterCrop((224, 224)),
     transforms.ToTensor()])

x = prep(im)
x = x.unsqueeze(0)

view_data(x[0] * 255)

v, idx = [x.detach().numpy()[0] for x in nn.Softmax(dim=1)(model(x)).topk(5)]
for prob, pred in zip(v, idx):
    Id, n = labels[str(pred)]
    print(f'{pred:8} - {Id:10} - {prob:.2f} - {n} ')

show_hist(x[0] * 255)

# %%
# ColorJit and Resize
prep = transforms.Compose([
    transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
    transforms.Grayscale(3),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

x = prep(im)
x = x.unsqueeze(0)

view_data(x[0] * 255)

v, idx = [x.detach().numpy()[0] for x in nn.Softmax(dim=1)(model(x)).topk(5)]
for prob, pred in zip(v, idx):
    Id, n = labels[str(pred)]
    print(f'{pred:8} - {Id:10} - {prob:.2f} - {n} ')

# %%
# Five-Crop
prep = transforms.Compose([
    transforms.FiveCrop((224, 224)),
    transforms.Lambda(lambda crops: torch.stack(
        [transforms.ToTensor()(crop) for crop in crops]))
])

x = prep(im)
# x = x.unsqueeze(0)

entries = [[
    x.detach().numpy()[i] for x in nn.Softmax(dim=1)(model(x)).topk(5)
] for i in range(x.shape[0])]
for v, idx in entries:
    print('---------------------------')
    for prob, pred in zip(v, idx):
        Id, n = labels[str(pred)]
        print(f'{pred:8} - {Id:10} - {prob:.2f} - {n} ')
    print('---------------------------')

# Multiple Image Display
fig = plt.figure(figsize=(8, 8))
for i in range(1, 6):
    fig.add_subplot(2, 3, i)
    plt.imshow(
        np.transpose(np.asarray(x[i - 1] * 255, dtype=np.uint8), (1, 2, 0)))
plt.show()

# %%
# Grayscale
prep = transforms.Compose([
    transforms.Grayscale(3),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

x = prep(im)
x = x.unsqueeze(0)

view_data(x[0] * 255)

v, idx = [x.detach().numpy()[0] for x in nn.Softmax(dim=1)(model(x)).topk(5)]
for prob, pred in zip(v, idx):
    Id, n = labels[str(pred)]
    print(f'{pred:8} - {Id:10} - {prob:.2f} - {n} ')

# %%
# RandomAffine
prep = transforms.Compose([
    transforms.RandomAffine(degrees=random.randrange(0, 180),
                            translate=None,
                            scale=(1, 2),
                            shear=random.randrange(0, 30),
                            resample=False,
                            fillcolor=0),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

x = prep(im)
x = x.unsqueeze(0)

view_data(x[0] * 255)

v, idx = [x.detach().numpy()[0] for x in nn.Softmax(dim=1)(model(x)).topk(5)]
for prob, pred in zip(v, idx):
    Id, n = labels[str(pred)]
    print(f'{pred:3} - {Id:10} - {prob:.2f} - {n} ')

# %%
# Flip
prep = transforms.Compose([
    transforms.RandomAffine(degrees=(270, 270)),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

x = prep(im)
x = x.unsqueeze(0)

view_data(x[0] * 255)

v, idx = [x.detach().numpy()[0] for x in nn.Softmax(dim=1)(model(x)).topk(5)]
for prob, pred in zip(v, idx):
    Id, n = labels[str(pred)]
    print(f'{pred:3} - {Id:10} - {prob:.2f} - {n} ')

# %%
# Magic Transform
# mim = magic_image(im, 2e4)
mim = salt_and_pepper(im, 300 * 300)

prep = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor()])

x = prep(mim)
x = x.unsqueeze(0)

view_data(x[0] * 255)

v, idx = [x.detach().numpy()[0] for x in nn.Softmax(dim=1)(model(x)).topk(5)]
for prob, pred in zip(v, idx):
    Id, n = labels[str(pred)]
    print(f'{pred:8} - {Id:10} - {prob:.2f} - {n} ')

# %%
prep = transforms.Compose(
    [transforms.CenterCrop((224, 224)),
     transforms.ToTensor()])

x = prep(mim)
x = x.unsqueeze(0)

view_data(x[0] * 255)

v, idx = [x.detach().numpy()[0] for x in nn.Softmax(dim=1)(model(x)).topk(5)]
for prob, pred in zip(v, idx):
    Id, n = labels[str(pred)]
    print(f'{pred:8} - {Id:10} - {prob:.2f} - {n} ')

# %%
prep = transforms.Compose([
    transforms.CenterCrop((333, 333)),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

x = prep(mim)
x = x.unsqueeze(0)

view_data(x[0] * 255)

v, idx = [x.detach().numpy()[0] for x in nn.Softmax(dim=1)(model(x)).topk(5)]
for prob, pred in zip(v, idx):
    Id, n = labels[str(pred)]
    print(f'{pred:8} - {Id:10} - {prob:.2f} - {n} ')
