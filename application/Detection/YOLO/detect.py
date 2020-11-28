import sys, os
sys.path.append("/home/aliclotho/GitRepos/DLToolbox/")
import random
import time

from toolbox.core.utils import AverageMeter
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from matplotlib.ticker import NullLocator
from .models.yolo import *
from .models.darknet import darknet53
# from torchvision.ops import nms
from torchvision.datasets import ImageFolder
from torchvision import transforms
from .datasets import custom
from .utils.utils import *


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    size = (3, 512, 512)
    b = 1

    # data
    dataset = ImageFolder('data/',
                          transform=transforms.Compose([
                              transforms.Resize(512),
                              transforms.ToTensor(),
                          ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             shuffle=False, num_workers=1)
    # if os.path.exists('/home/mooziisp'):
    #     data_root = '/home/mooziisp/Downloads/人体脊椎MRI图像/part3/labels-coco/train/'
    # else:
    #     data_root = '/home/aliclotho/GitRepos/yolact/data/custom/train/'
    #
    # dataset = custom.SpineDetection(
    #     data_root,
    #     data_root + 'annotations.json',
    #     target_size=size,
    #     bg_class=False
    # )
    # dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=b,  # FIXME
    #     shuffle=True,
    #     collate_fn=dataset.collate_fn,
    #     num_workers=1
    # )

    net = yolov3(backbone=darknet53,
                inp_dim=(3, 512, 512),
                num_classes=80)
    # net.to(device)
    # load existed weights
    net.load_state_dict(torch.load('weights/spine-512-80.pt'), strict=False)
    # net.load_state_dict(torch.load('weights/test.pt', map_location='cpu'))

    net.eval()

    inference = AverageMeter('cost')
    print("Performing object detection:")
    # One image
    ims = []
    im_detections = []
    for it, (im, _) in enumerate(dataloader):
        ims.extend(im.permute(0, 2, 3, 1).numpy())

        tik = time.perf_counter()
        with torch.no_grad():
            # detections, _ = net(im, _)
            detections = net(im)
            inference.update(time.perf_counter() - tik)
            detections = non_max_suppression(detections, 0.5, 0.3)
        im_detections.extend(detections)
        print('inference time: {:.2f}s'.format(inference.avg))

    # Color bbox
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    # Save the result
    for it, (im, detections) in enumerate(zip(ims, im_detections)):
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(im)

        print(it)
        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, 512, im.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                print("\t+ Label: %s, Conf: %.5f" % (int(cls_pred), cls_conf.item()))

                box_w = x2 - x1
                box_h = y2 - y1
                print(x1, y1, box_w, box_h)

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=1, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(
                    x1,
                    y1,
                    s=int(cls_pred),
                    color="white",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                )

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        plt.savefig(f"output/{it}-result.png", bbox_inches="tight", pad_inches=0.0)
        plt.close()
