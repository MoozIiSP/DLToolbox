import os

import numpy as np
import torch
from PIL import Image, ImageFilter
from labelme.utils import shape_to_mask
from pycocotools.mask import toBbox, encode
from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset
import torchvision.transforms.functional as F


class SpineDetection(VisionDataset):
    """the spine dataset labeled by Labelme for detection task."""
    def __init__(self, root, annFile,
                 transform=None, target_transform=None, transforms=None):
        super(SpineDetection, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, idx):
        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        ann_target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        # Image
        im = F.to_tensor(Image.open(os.path.join(self.root, path)).convert('RGB'))
        _, height, width = im.shape

        # Generate mask and bbox from segmentation of coco
        bboxes = []
        masks = []
        labels = []
        for cat in ann_target:
            # convert polygon into mask and then into bbox
            for p in cat['segmentation']:
                points = [(p[i], p[i+1]) for i in range(0, len(p), 2)]
                # Smooth the mask using PIL.ImageFilter.SMOOTH
                smooth_mask = Image.fromarray(shape_to_mask(im.shape[1:], points, 'polygon')).filter(ImageFilter.SMOOTH)
                masks.append(np.asarray(smooth_mask))

                # left top (x, y), width and height
                x, y, w, h = toBbox(encode(np.asfortranarray(smooth_mask).astype(np.uint8)))
                # normalization
                x1 = x
                y1 = y
                x2 = x + w
                y2 = y + h
                bboxes.append([x1, y1, x2, y2])

                labels.append(cat['category_id'])

        # convert everything into a torch.Tensor
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        labels = torch.as_tensor(labels, dtype=torch.int64)  # float32

        image_id = torch.tensor([idx])
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(bboxes), ), dtype=torch.int64)

        # no batch idx
        target = {}
        target["boxes"] = bboxes
        target["labels"] = labels + 1  # add background class
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return im, target

    def __len__(self):
        return len(self.ids)


class PennFudanDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs, ), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs, ), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    import sys
    sys.path.append('/home/mooziisp/GitRepos/DLToolbox/third_party/vision/references/detection')
    import utils
    import transforms as T

    def get_transform(train):
        transforms = []
        transforms.append(T.ToTensor())
        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))
        return T.Compose(transforms)

    data_root = '/home/mooziisp/Downloads/人体脊椎MRI图像/part3/labels-coco/train/'
    # data_root = '/home/mooziisp/Downloads/PennFudanPed/'

    dataset = SpineDetection(
        data_root,
        data_root + 'annotations.json',
    )
    # dataset = PennFudanDataset(data_root, get_transform(train=True))

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=1,
        collate_fn=utils.collate_fn  # tuple(zip(*batch))
    )

    for i, (im, target) in enumerate(dataloader):
        # print(im.shape)

        # print(target['boxes'].shape)
        print(target[0]["labels"])
        # print(target["masks"].shape)
        # print(target["image_id"].shape)
        # print(target["area"].shape)
        # print(target["iscrowd"].shape)
        break

