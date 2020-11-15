import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


class CustomSegDataset(Dataset):
    """Custom MRI/CT dataset for abnormality segmentation"""

    in_channels = 1
    out_channels = 1

    def __init__(
        self,
        images_dir,
    ):
        if not os.path.isdir(images_dir):
            raise RuntimeError('Dataset not found or corrupted.')
        self.images_dir = images_dir
        
        # read path of images into the memory
        self.images = []
        self.masks = []
        for (dirpath, dirnames, filenames) in os.walk(images_dir):
            for filename in sorted(
                    filter(lambda f: '.png' in f, filenames),
                    key = lambda x: int(x.split('.')[0].split('-')[1])):
                filepath = os.path.join(dirpath, filename)
                if 'mask' in filename:
                    self.masks.append(filepath)
                else:
                    self.images.append(filepath)

        assert len(self.images) == len(self.masks), \
            "length: images != masks"

    def _transform(self, img, mask):
        # TODO crop - discard some pixel under specify threshold and crop
        #   remaining non zero pixels to save. Warning: np.max(im, axis = n),
        #   when n is negative, implies that np.max will choose axis
        #   reversely. And -1 is x-axis, -2 is y-axis.
        # TODO pad - check if its height is same as width. If not, this function
        #   will pad zero into the data to make its height to equals width.
        # TODO resize - just resize the data.
        # TODO normalize - rescale pixel intensity into the range of the data
        #   , and compute mean and std to normalize.

        return F.to_tensor(img), F.to_tensor(mask)
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        mask = Image.open(self.masks[index])

        return self._transform(img, mask)
