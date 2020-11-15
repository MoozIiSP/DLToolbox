import io
import itertools

import torch
import matplotlib.pyplot as plt
import numpy as np


__all__ = ['ConfusionMatrix']


class ConfusionMatrix(object):
    def __init__(self, size, classnames):
        if type(size) is list:
            self.cm = np.zeros(size)
        elif type(size) is int:
            self.cm = np.zeros([size, size])
        self.classnames = classnames

    def update(self, output, target):
        # NOTE convert i, j into integers
        for i, j in zip(*[target, output]):
            self.cm[int(i), int(j)] += 1

    def _plot(self, normalize=False, ticks=False):
        # Turn interactive plotting off
        plt.ioff()

        # Normalize the confusion matrix.
        try:
            cm = np.around(self.cm.astype('float') / self.cm.sum(axis=1)[:, np.newaxis], decimals=2)
        except ZeroDivisionError:
            cm = np.around(self.cm.astype('float') / 1.0, decimals=2)

        figure = plt.figure(figsize=(6, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()

        if ticks:
            tick_marks = np.arange(len(self.classnames))
            plt.xticks(tick_marks, self.classnames, rotation=45)
            plt.yticks(tick_marks, self.classnames)

        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return figure

    def plot_to_tensorborad(self, **kwargs):
        """
        Returns a matplotlib figure containing the plotted confusion matrix.

        Args:
          nomalize:
        """
        figure = self._plot(**kwargs)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        return torch.Tensor(np.transpose(plt.imread(buf), (2,0,1)))
