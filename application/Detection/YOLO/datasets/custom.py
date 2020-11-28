import os
import os.path
import random
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw, ImageFilter
from pycocotools.coco import COCO
from pycocotools.mask import encode, toBbox
from torchvision.datasets.vision import VisionDataset


def polygon_to_mask(polygon, target_size):
    # L - 8bit pixels
    mask = Image.new('1', target_size)
    draw = ImageDraw.Draw(mask)
    draw.polygon(polygon, outline=1, fill=1)
    return np.asarray(mask, dtype=np.bool)


class SpineDetection(object):
    def __init__(
        self, root: str, annFile: str, target_size: Tuple[int, int, int],
        # max_objects=10,
        bg_class: Optional[bool]=False,
        multiscale: Optional[bool]=True
    ):
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.root = root
        self.target_size = target_size
        # self.max_objects = max_objects
        self.bg_class = bg_class
        self.multiscale = multiscale
        self.min_size = self.target_size[1] - 3 * 32
        self.max_size = self.target_size[1] + 3 * 32
        self.im_size = target_size[1:]
        self.batch_count = 0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        ann_target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        # Image
        ch, target_size = self.target_size[:2]
        if ch != 1:
            im = Image.open(os.path.join(self.root, path)).convert('RGB')
        else:
            im = Image.open(os.path.join(self.root, path))

        # adjust colorjitter
        im = T.ColorJitter(.1, .1, .1, .1)(im)

        im_height, im_width = im.size
        im_tensor = F.to_tensor(F.resize(im, target_size))
        # height_scale, width_scale = target_size[0] / im_height, target_size[1] / im_width

        # Generate mask and bbox from segmentation of coco
        bboxes = []
        masks = []
        labels = []
        for cat in ann_target:
            # convert polygon into mask and then into bbox
            for p in cat['segmentation']:
                points = [(p[i], p[i+1]) for i in range(0, len(p), 2)]
                # Smooth the mask using PIL.ImageFilter.SMOOTH
                smooth_mask = Image.fromarray(polygon_to_mask(points, im.size)).filter(ImageFilter.SMOOTH)
                masks.append(np.asarray(smooth_mask))

                # left top (x, y), width and height
                x, y, w, h = toBbox(encode(np.asfortranarray(smooth_mask).astype(np.uint8)))
                # normalization - absolution scale
                x = (x + w/2) / im_width
                y = (y + h/2) / im_height
                w /= im_width
                h /= im_height
                bboxes.append([x, y, w, h])
                labels.append(cat['category_id'])

        # convert everything into a torch.Tensor
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([index])
        area = bboxes[:, 2] * bboxes[:, 3]
        # suppose al instances are not crowd
        iscrowd = torch.zeros((len(bboxes), ), dtype=torch.int64)

        # padding zero (dummy value) to make bounding boxes have same size
        # offset = self.max_objects - bboxes.size(0)
        # bboxes = torch.cat((bboxes, torch.zeros([offset, 4])), dim=0)
        # masks = torch.cat((masks, torch.zeros([offset, im_height, im_width], dtype=torch.uint8)), dim=0)
        # labels = torch.cat((labels, torch.zeros([offset], dtype=torch.int64)), dim=0)

        # Transforms - Data Augmentation
        im_tensor, bboxes = hflip(im_tensor, bboxes)
        im_tensor, bboxes = vflip(im_tensor, bboxes)

        # no batch idx
        bg_class = 1 if self.bg_class else 0
        target = {
            'boxes': bboxes,
            'labels': labels + bg_class,
            'masks': masks,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd
        }

        return im_tensor, target  # tuple(zip(*target))  # collate_fn

    # TODO
    def collate_fn(self, batch):
        ims, targets = list(zip(*batch))
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 100 == 0:
            self.im_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        im = torch.stack([resize(im, self.im_size) for im in ims])
        self.batch_count += 1
        return im, targets

    def __len__(self):
        return len(self.ids)


def resize(image, size):
    image = torch.nn.functional.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def hflip(im, bbox, p=0.5):
    if np.random.random() < p:
        # hflip
        im = torch.flip(im, [-1])
        bbox[:, 0] = 1 - bbox[:, 0]
    return im, bbox


def vflip(im, bbox, p=0.5):
    if np.random.random() < p:
        # vflip
        im = torch.flip(im, [-2])
        bbox[:, 1] = 1 - bbox[:, 1]
    return im, bbox


if __name__ == '__main__':
    data_root = '/home/mooziisp/Downloads/人体脊椎MRI图像/part3/labels-coco/train/'

    dataset = SpineDetection(
        data_root,
        data_root + 'annotations.json',
        target_size=(1, 512, 512),
        # max_objects=10,
        bg_class=True,
        multiscale=False
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,  # FIXME
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=1
    )

    boxes = []
    for i, (_, target) in enumerate(dataloader):
        print(i)
        if i == len(dataloader) - 1:
            break
        for t in target:
            boxes.append(t['boxes'][..., 2:])
        plt.axis('off')
        plt.imsave(f'{i}.png', _[0][0], cmap='gray')
    boxes = torch.stack(boxes, dim=0)

    # import matplotlib.pyplot as plt
    # from PIL import Image, ImageDraw
    # im_w, im_h = (512, 512)
    # fig = plt.figure()
    # for i, (input, target) in enumerate(dataloader):
    #     print(input.shape)
    #     im = Image.fromarray(np.asarray(input[0, 0]))
    #     draw = ImageDraw.Draw(im)
    #     for t in target:
    #         for b in t['boxes']:
    #             xywh = b.tolist()
    #             x1 = xywh[0] * im_w - xywh[2] / 2 * im_w
    #             y1 = xywh[1] * im_h - xywh[3] / 2 * im_h
    #             x2 = xywh[0] * im_w + xywh[2] / 2 * im_w
    #             y2 = xywh[1] * im_h + xywh[3] / 2 * im_h
    #             draw.rectangle([x1, y1, x2, y2], outline=1)
    #     im = np.asarray(im)
    #     plt.imshow(im)
    #     break
    # plt.show()

