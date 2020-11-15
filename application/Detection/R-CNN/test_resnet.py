#!/usr/bin/env python

import json
import glob

import utils

import torch
import torch.nn as nn

import numpy as np
from PIL import Image
from torchvision import models, transforms

preprocessor = transforms.Compose([ 
    transforms.Resize((224, 224)),
    transforms.ToTensor()])

#files = glob.glob('./cat*.*')

for f in range(1):
    im = Image.open('cat8.jpg')
    #print(np.array(im).shape)
    labels = json.load(open('imagenet_class_index.json', 'r'))

    model = models.resnet18(pretrained = True)
    model.eval()

    softmax = nn.Softmax()

    # 0 45 357 455
    im = np.array(im)[0:357, 45:455, :]
    x = torch.Tensor(np.expand_dims(preprocessor(Image.fromarray(im)), axis = 0))
    #print(x.shape)
    output = model(x)
    pred = output.argsort(descending = True)
    prob = softmax(output)
    print(f, pred[0][:5])
    for idx in pred[0][:5]:
        print('{} - {:.3f}%  - {}'.format(idx, prob.flatten()[idx], labels[str(int(idx))][1]))
    print('\n')
