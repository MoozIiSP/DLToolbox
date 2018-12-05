from __future__ import absolute_import

import os

import numpy as np

from ..project import load_pk


def load_weights(model_name, grayscale=False):
    root = os.path.split(__file__)[0]
    files = os.listdir(root)

    for f in files:
        # FIXME
        if model_name.lower() in f:
            layer_name, weights = load_pk(os.path.join(root, f)).values()
            if len(weights) == 2:
                w, bias = weights
            else:
                w = weights[0]

            if grayscale:
                r, g, b = w[:, :, 0, :], w[:, :, 1, :], w[:, :, 2, :]
                w = ((r * 30 + g * 59 + b * 11 + 50) / 100)[:, :, np.newaxis, :]

            if len(weights) == 1:
                return layer_name, [w]
            else:
                return layer_name, (w, bias)

    raise ValueError('Please check whether model exists.')
