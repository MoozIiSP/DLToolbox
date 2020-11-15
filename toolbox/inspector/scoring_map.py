# %%
import cv2
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms

# import torch2trt

dev = torch.device('cuda:0')

input_shape = (224, 224)
net = models.vgg11_bn(pretrained=True).cuda().eval()

# Group nn into convolution layer and full-connection layer
# features = [
#     net.conv1,
#     net.bn1,
#     net.relu,
#     net.maxpool,
#     net.layer1,
#     net.layer2,
#     net.layer3,
#     net.layer4,
#     # net.avgpool,
# ]
# classifier = [
#     net.avgpool,
#     net.fc
# ]

# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToTensor(),
                                normalize])

im = Image.open('tiger-cat-1.jpg')
# im = Image.open('SAM_3280-960x640.jpg')
w, h = im.size
im = im.resize((w//2, h//2), resample=Image.BILINEAR)
# im = im.resize((112, 112), resample=Image.NEAREST)

padding_size = [x // 2 for x in input_shape]
im_padded = np.pad(im, (padding_size, padding_size, (0, 0)), mode='constant')
plt.imshow(im_padded)
print(im.size, im_padded.shape)


def generator(im, padding_size, transform, stride=0, batch_size=8, sliding_size=112):
    cnt = 0
    batch = None
    h, w, c = im.shape
    for i in range(padding_size, h - padding_size):
        for j in range(padding_size, w - padding_size):
            im_trans = transform(im_padded[i - sliding_size:i + sliding_size, j - sliding_size:j +
                                           sliding_size, :]).unsqueeze(0)
            # init
            if cnt == 0:
                batch = im_trans
                cnt = 1
                continue
            if cnt == batch_size:
                yield batch
                # init count and batch container
                batch = im_trans
                cnt = 1
            batch = torch.cat((batch, im_trans), 0)
            cnt += 1


def generator_v1(im, padding_size):
    h, w, c = im.shape
    for i in range(padding_size, h - padding_size):
        for j in range(padding_size, w - padding_size):
            yield im_padded[i - padding_size:i + padding_size,
                            j - padding_size:j + padding_size,
                            :]


# scoring_map = np.zeros(439 * 780)
scoring_map = []
acc_map = np.zeros(im.size[1] * im.size[0])

BATCH_SIZE = 256
# gen = generator(im_padded, padding_size=112, transform=transform, batch_size=BATCH_SIZE)
gen = generator_v1(im_padded, padding_size=input_shape[0]//2)

# TEST
# x = transform(im).unsqueeze(0).cuda()
# for fea in features:
#     print(x.shape)
#     x = fea(x)
# print(x.shape)

for i, image in enumerate(gen):
    outputs = net(transform(image).unsqueeze(0).cuda())
    #st = time.time()
    # outputs = net(images.cuda())
    score = nn.Softmax(dim=1)(outputs)
    _, label = score.topk(1)
    if 282 <= label.item() <= 285:
        acc_map[i] += 1
    #print(f'{i} - {(time.time() - st):.2f}ms')
    # flag = 1
    # for j, score in enumerate(scores):
    #     avg = (score[282].item() + score[283].item() + score[284].item() +
    #            score[285].item()) / 4.
    #     if flag == 1 and i % 10 == 0:
    #         print(f'{i/(439*780/BATCH_SIZE)*100:3.0f}% -> {j} Cat predict avg {avg:.2f}')
    #         flag = 0
    #     # scoring_map[i] = avg
    #     scoring_map.append(avg)
    if i % 10000 == 0:
        print(i)
    scoring_map.append((score[0][282].item()+score[0][283].item()+score[0][284].item()+score[0][285].item())/4)

w, h = im.size
scoring_map = np.asarray(scoring_map).reshape(h,w)
plt.figure(figsize=(12,12))
plt.imshow(scoring_map, cmap='hot')

acc_map = acc_map.reshape(h,w)
plt.figure(figsize=(12,12))
plt.imshow(acc_map, cmap='hot')

plt.figure(figsize=(12,12))
score_map_int = cv2.cvtColor((scoring_map*255*5).astype(np.uint8), cv2.COLOR_GRAY2RGBA)
im_int = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2RGBA)
plt.imshow(cv2.addWeighted(im_int, 0.2,
                           score_map_int, 0.8,
                           0))