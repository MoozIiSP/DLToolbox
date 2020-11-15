import json
import numpy as np

import torch.nn
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import models, transforms
from torchvision.utils import make_grid

from .utils import np_make_grid

im = Image.open('/home/mooziisp/bird.png')

one_shot_prec = transforms.Compose([
    #lambda im: Image.fromarray(im),
    #transforms.Resize((224, 224)),
    transforms.CenterCrop((224,224)),
    lambda im: np.transpose(np.array(im)[:, :, :3], (2, 0, 1)),
    lambda im: np.expand_dims(im, axis = 0),
    lambda im: torch.Tensor(im)])


im = one_shot_prec(im)

model = models.alexnet(pretrained = True)
model.eval()

def heatmap(im):
    if np.max(im) - np.min(im) != 0:
        return (im - np.min(im)) / (np.max(im) - np.min(im)) * 255
    else:
        return im

x = model.features[0](im)
x = model.features[1](x)
x_vis = [x.data[:, i, :, :].numpy() for i in range(x.shape[1])]
x_heat = [heatmap(x) for x in x_vis]

#%matplotlib qt5
plt.imshow(np_make_grid(x_heat, 8)[:,:,0], cmap='hot')

x = model.maxpool(x)
x_vis = [x.data[:, i, :, :].numpy() for i in range(x.shape[1])]
x_heat = [heatmap(x) for x in x_vis]
plt.imshow(np_make_grid(x_heat, 8)[:,:,0], cmap='hot')

plt.imsave('layer_conv1_para_output.png', np_make_grid(x_vis, 8)[:, :, 0])


# FIXME layer_maxpool_para = models.maxpool.data.numpy()
# it not have weight
x = model.bn1(x)
x_vis = [x.data[:, i, :, :].numpy() for i in range(x.shape[1])]
plt.imsave('layer_bn1_output.png', np_make_grid(x_vis, 8)[:, :, 0])

x = model.relu(x)
x_vis = [x.data[:, i, :, :].numpy() for i in range(x.shape[1])]
plt.imsave('layer_relu_output.png', np_make_grid(x_vis, 8)[:, :, 0])

x = model.maxpool(x)
x_vis = [x.data[:, i, :, :].numpy() for i in range(x.shape[1])]
plt.imsave('layer_maxpool_output.png', np_make_grid(x_vis, 8)[:, :, 0])

x = model.layer1(x)
x_vis = [x.data[:, i, :, :].numpy() for i in range(x.shape[1])]
plt.imsave('layer1_output.png', np_make_grid(x_vis, 8)[:, :, 0])

x = model.layer2(x)
x_vis = [x.data[:, i, :, :].numpy() for i in range(x.shape[1])]
plt.imsave('layer2_output.png', np_make_grid(x_vis, 8)[:, :, 0])

x = model.layer3(x)
x_vis = [x.data[:, i, :, :].numpy() for i in range(x.shape[1])]
plt.imsave('layer3_output.png', np_make_grid(x_vis, 8)[:, :, 0])

x = model.layer4(x)
x_vis = [x.data[:, i, :, :].numpy() for i in range(x.shape[1])]
plt.imsave('layer4_output.png', np_make_grid(x_vis, 8)[:, :, 0])

x = model.avgpool(x)
x_vis = [x.data[:, i, :, :].numpy() for i in range(x.shape[1])]
plt.imsave('layer_avgpool_output.png', np_make_grid(x_vis, 8)[:, :, 0])

x = model.fc(x.view(x.size(0), 512))

labels = json.load(open('imagenet_class_index.json', 'r'))
print(labels[str(int(x.argmax().numpy()))][1])
