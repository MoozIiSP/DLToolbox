import os
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image

# TODO: imshow image of Tensor
def imshow(tensor, w, h, save = False):
    """

    :param tensor: 
    :param w: 
    :param h: 
    :param save:  (Default value = False)

    """
    plt.figure(figsize=(10, 6))
    for i, img in enumerate(tensor):
        plt.subplot(w,h,i+1)
        plt.axis('off')
        # print labels
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.tight_layout()
    if save:
        plt.savefig('example.png')
    plt.show()

# architecture of ResNeXt50_32x4d is same as ResNet50
net = models.resnext50_32x4d(pretrained = True)
net.eval()

# output activition of layer:
# resnet.conv1
# resnet.layer1[1:2]
# resnet.layer2[1:3]
# resnet.layer3[1:5]
# resnet.layer4[1:2]

# TODO: Load Image and make one-shot 4D Tensor for testing Model
dir = '/home/mooziisp/GitRepos/toolbox/object_detection/R-CNN'
oneshot = Image.open(os.path.join(dir, 'bird.jpg'))
classes = json.load(open(os.path.join(dir, 'imagenet_class_index.json'), 'r'))

transformer = transforms.Compose([transforms.RandomCrop(224, 4),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
                                  lambda im: torch.stack((im, ), dim = 0)])

oneshot = transformer(oneshot)
# TODO: Show this is Image for one-shot testing
# You need to check docs about transforms.ToTensor
# whether rescale image into [0, 1].
#plt.imshow(oneshot, [0], ['TEST'])
imshow(oneshot, 1, 1)

# TODO: make layer pipeline to generate respond feature map
# FIXME: draft
layers = list(net.children())
first_conv1 = layers[:3]
layer1_conv = layers[:4]
layer2_conv = layers[:5]
layer3_conv = layers[:6]
layer4_conv = layers[:7]

# TODO: Random to show respond feature map from result
# For exmaple, torch.Size([1, 64, 112, 112]) from torch.Size([1, 3, 224, 224])
def imshow_respond(tensor, W, H):
    """

    :param tensor: 
    :param W: 
    :param H: 

    """
    batch_size, output_size, w, h = tensor.shape
    # Generate random indice to get related to respond feature map
    indices = np.random.randint(0, output_size, size = W*H)
    # top=0.95, bottom=0.05, left=0.005, right=0.995, hspace=0.0, wspace=0.05
    plt.figure(figsize=(6, 6))
    for id, indice in enumerate(indices):
        plt.subplot(W, H, id + 1)
        # TODO: Plot your respond feature map
        plt.axis('off')
        plt.title(f'#{indice}', loc = 'left', fontsize = 'small')
        img = tensor[0, indice, :, :].detach() / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(npimg, cmap = 'viridis')
    plt.show()

# TODO: Extracting respond feature map
for pipe in (first_conv1, layer1_conv, layer2_conv, layer3_conv, layer4_conv):
    x = oneshot
    for layer in pipe:
        x = layer(x)
    imshow_respond(x, 5, 10)

# TODO: second part - extract respond pattern from layer
