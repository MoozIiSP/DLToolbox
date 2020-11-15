import torch
import torch.nn as nn
import torch.nn._reduction as _Reduction
import torch.nn.functional as F
from torch.nn.modules.module import Module


class _Loss(Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class VAELoss(_Loss):
    """Reconstruction + KL divergence losses summed over all elements and batch"""
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(VAELoss, self).__init__(size_average, reduce, reduction)

    def forward(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy_with_logits(recon_x.flatten(1, -1), x.flatten(1, -1), reduction=self.reduction)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc
