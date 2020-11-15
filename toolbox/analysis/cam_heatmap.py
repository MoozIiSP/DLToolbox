# TODO not implemented yet
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torchvision import models

im = np.transpose(plt.imread('el.png'), (2, 0, 1))
x = torch.tensor(im).unsqueeze(0)

net = models.vgg11(pretrained=True)
print(net)

pred = net(x)
print(pred.argmax().item())

one_hot = torch.zeros([1, 1000], dtype=torch.float32)
one_hot[0][386] = 1
# backward to compute gradient
pred.backward(one_hot, retain_graph=True)

# print(pred.shape, net.classifier[6].weight.shape)
last_conv = net.features[18]

conv_output = net.features[:19](x).data.numpy()

grads = last_conv.weight.grad.data.numpy()

pooled_grads = np.mean(grads, axis=(1,2))

for i in range(512):
    conv_output[0, i, :, :] *= pooled_grads[i]

fig = plt.figure()
heatmap = np.transpose(np.mean(conv_output, axis=1), (1,2,0))[:,:,0]
plt.imshow(heatmap)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()

import cv2
heatmap = cv2.resize(heatmap, (224, 224))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_im = heatmap * 0.4 + cv2.imread('el.png')
cv2.imwrite('heat.png', superimposed_im)

