import json
import time

from operator import itemgetter

import cv2
import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from PIL import Image
from torchvision import models, transforms
# NOTE You need clone and compile by yourself
# to install the branch of nn-v2 of vision.
from torchvision.ops import nms

from utils import selective_search, non_max_suppression_fast

if __name__ == '__main__':
    im = np.array(Image.open('bird.jpg'))
    labels = json.load(open('imagenet_class_index.json', 'r'))

    # Using default paras
    # orignal paper is ks = [50, 100, 150, 300]
    k, inc = 50, 50
    sigma = 0.8
    minLen = 20
    maxLen = 300

    # compute selective search candidates
    tic = time.time()
    # for k in ks:
    rects = selective_search(im, k, inc, sigma, mode='f')
    print('Generated {} regions (in {:.3f}s)'.format(len(rects),
                                                     time.time() - tic))

    tic = time.time()
    proposals = set()
    for r in rects:
        x, y, w, h = r
        x1, y1, x2, y2 = x, y, x + w, y + h
        # BoxRemoveDuplicates - excluding same rectangle (with different segments)
        if set(r.tolist()) in proposals:
            continue
        # excluding regions smaller than 2000 pixels
        # if (x2 - x1) * (y2 - y1) < 10000:
        #     continue
        # if (x2 - x1) * (y2 - y1) > 250000:
        #     continue
        # FilterBoxesWidth
        if w < minLen and h < minLen:
            continue
        if w > maxLen and h > maxLen:
            continue
        if w / h > 1.2 or h / w > 1.2:
            continue
        #print(x1, y1, x2, y2)
        proposals.add((x1, y1, x2, y2))
    print('Filtered {} propesals (in {:.3f}s).'.format(len(proposals),
                                                       time.time() - tic))

    # extract features from candidates (one row per candidate box)
    tic = time.time()
    device = torch.device('cpu')
    model = models.resnet18(pretrained=True).to(device)
    # Set to Eval mode
    model.eval()
    preprocessor = transforms.Compose([
        # FIXME need to refactor
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    print('Initilized CNN (in {:.3f}s)'.format(time.time() - tic))

    tic = time.time()
    scored_boxes = []
    thresh = 0.5
    log = []
    for i, (x1, y1, x2, y2) in enumerate(proposals):
        # FIXME there is a bug
        x = preprocessor(Image.fromarray(im[y1:y2, x1:x2, :]))
        x = torch.Tensor(np.expand_dims(x, axis=0)).to(device)
        outputs = model(x)
        score, label = nn.Softmax(dim=1)(outputs).topk(1)
        # NOTE debug info
        # print(i, labels[str(int(label.data.max()))], float(score.data.max()))
        log.append(
            [labels[str(int(label.data.max()))],
             float(score.data.max())])
        # NOTE filter some boxes
        if score > thresh:
            # if 'cat' in labels[str(int(label.data.max()))][1] and score > thresh:
            scored_boxes.append(
                np.array([
                    x1, y1, x2, y2,
                    score.flatten().data.numpy() * 100,
                    label.flatten().data.numpy()
                ]))
    # nms only support float32
    scored_boxes = np.asarray(scored_boxes, dtype=np.float32)
    print('Compute scores (in {:.3f}s)'.format(time.time() - tic))

    tic = time.time()
    boxes = torch.tensor(scored_boxes[:, :4])
    scores = torch.tensor(scored_boxes[:, 4])
    # res = non_max_suppression_fast(scored_boxes, 0.3)
    res = [scored_boxes[x] for x in nms(boxes, scores, 0.3)]
    print('done (in {:.3f}s)'.format(time.time() - tic))

    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(im)
    for x1, y1, x2, y2, _, pred in res:
        # print(_, pred)
        pred_text = '{} {:.3f}'.format(labels[str(int(pred))][1], _)
        rect = mpatches.Rectangle((x1, y1),
                                  x2 - x1 + 1,
                                  y2 - y1 + 1,
                                  fill=False,
                                  edgecolor='red',
                                  linewidth=1)
        # print(pred_text)
        ax.annotate(pred_text, (x1, y1),
                    transform=ax.transAxes,
                    bbox=dict(facecolor='red', alpha=0.5, edgecolor='red'))
        ax.add_patch(rect)

    plt.show()

    print('\nDEBUG INFO:\n')
    log.sort(key=itemgetter(1))
    for _, __ in log[-20:]:
        print('{}\t{:8.3f}'.format(_[1], __))
