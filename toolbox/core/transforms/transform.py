import torch
import math
import sys
import random
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
import collections
import warnings

import torchvision.transforms.functional as F


class ResizedFiveCrop(object):

    def __init__(self, size, vertical_flip=False):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size
        self.vertical_flip = vertical_flip

    def __call__(self, img: Image.Image):
        target_size = np.array(img.size)
        # import numpy as np
        #
        # o_size = np.array((80., 120.))
        # t_size = np.array((224., 224.))
        #
        # def crop(o_size, t_size):
        #     if o_size[0] < t_size[0] or o_size[1] < t_size[1]:
        #         raise RuntimeError("Crop failed.")
        #     return True
        #
        # def resizedCrop(o_size, t_size):
        #     if o_size[0] < t_size[0] or o_size[1] < t_size[1]:
        #         ratio = max(t_size) / min(o_size)
        #         o_size = o_size * ratio
        #     return True
        #
        # resizedCrop(o_size, t_size)
        if min(target_size) > min(self.size):
            ratio = min(target_size) / min(self.size)
        return F.ten_crop(img, self.size, self.vertical_flip)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, vertical_flip={1})'.format(self.size, self.vertical_flip)
