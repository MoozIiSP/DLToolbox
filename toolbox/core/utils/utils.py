"""
compute_mean_and_std
imshow
plot_statictis
"""
from __future__ import absolute_import

import datetime
import os
from functools import reduce
from types import FunctionType

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision
from thop import profile, clever_format
from torchvision.utils import save_image

from . import colors


# TODO for custom layer count
def arch_stat(net, input_shape, device, custom_ops):
    """Tools from `"sovrasov/flops-counter.pytorch"
    <https://github.com/sovrasov/flops-counter.pytorch>`

    Returns:
      Params
      Flops
    """
    input = torch.randn(1, 3, input_shape, input_shape).to(device)
    flops, params = profile(net, inputs=(input,), custom_ops=custom_ops, verbose=False)
    return clever_format([params, flops], '%.3f')


def get_eta_time(max_iter, cur, period):
    return str(datetime.timedelta(seconds=(max_iter - cur) * period)).split('.')[0]


# TODO
def create_base_net(filepath, net_fn, **kwargs):
    if type(net_fn) is FunctionType:
        net_name = net_fn.__name__
    else:
        net_name = net_fn.__class__.__name__

    try:
        net = net_fn(**kwargs)
        torch.save(
            net.state_dict(),
            os.path.join(filepath, 'weights/{}-base-weight.pth').format(net_name))
    except:
        raise ValueError("{} doesn't existed.".format(net_name))


# NOTE Data Pre-Processing
def compute_mean_and_std(dataloader):
    """Computing the mean and standard deviation for every pixel of
    all images from dataset.

    Args:
      dataloader (DataLoader): Data loader. Combines a dataset and
        a sampler, and provides an iterable over the given dataset.

    Returns:
      mean (float): the mean of all images
      std (float): the standard deviation of all images
    """
    mean = 0.
    std = 0.
    nb_samples = 0.
    for _, (data, __) in enumerate(dataloader):
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    return mean, std


# TODO control parameters: figsize, size of subplot
# NOTE Analysis
def imshow(tensor, labels, nclass):
    """Display images from tensor data, and show its label correspond
    to categories. Figure will exhibit 144 images by 12 rows and 12
    columns.

    Args:
      tensor (Tensor): tensor contains images which its amount is
        batch_size.
      labels (Tensor): tensor contains labels which its amount is
        batch_size as same as images.
      nclass (list): category name correspond to labels.

    Returns:
      None

    """
    plt.figure(figsize=(14, 12))
    for i, img in enumerate(tensor):
        plt.subplot(12, 12, i + 1)
        plt.axis('off')
        # print labels
        plt.title(nclass[labels[i]])
        # img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.tight_layout()
    plt.show()


# NOTE Analysis
def plot_statictis(title, item, y_vals, path=None):
    """y_vals = [train, val]

    Args:
      title (str): figure title
      item (str): indicates statistic type
      y_vals (tuple): y_vals = [train, val]
      path (str): path to save the figure

    Returns:
      None
    """
    plt.figure()
    plt.title(title)
    for label, y_val in zip([f'train_{item}', f'val_{item}'], y_vals):
        x_val = np.arange(len(y_val))
        plt.plot(x_val, y_val, label=label)
    plt.xlabel('epoch')
    plt.ylabel(f'{item}')
    plt.legend(loc='upper left', shadow=True)
    if path:
        plt.savefig(f'{title}-{item}-log.png')
    else:
        plt.show()


# NOTE Analysis
def plot_recall_and_precision():
    """ """
    pass


def gen_image(device, net, z_length, shape, epoch):
    with torch.no_grad():
        sample = torch.randn(64, z_length).to(device)
        sample = net.decode(sample).cpu()
        save_image(sample.view(64, shape[2], shape[0], shape[1]),
                   'results/sample_' + str(epoch) + '.png')


# NOTE Transfer
def freeze(modules, filter, requires_grad=False, prefix='', verbose=False):
    res = {'unchange': []}
    for name, module in modules.named_children():
        path = f"{prefix}.{name}"
        if path in filter:
            if verbose:
                try:
                    print(f"{path}: SKIPPED - {module.weight.requires_grad}")
                except:
                    print(f"{path}: SKIPPED")
            res['unchange'].append(f"{path}")
            continue
        if type(module) is torch.nn.modules.container.Sequential:
            res['unchange'].extend(
                freeze(module, filter, requires_grad, f"{path}",
                       verbose)['unchange'])
            continue
        elif type(module) is torchvision.models.resnet.BasicBlock:
            res['unchange'].extend(
                freeze(module, filter, requires_grad, f"{path}",
                       verbose)['unchange'])
            continue
        elif type(module) is torchvision.models.resnet.Bottleneck:
            res['unchange'].extend(
                freeze(module, filter, requires_grad, f"{path}",
                       verbose)['unchange'])
            continue
        try:
            if verbose:
                print(
                    f"{path}: {module.weight.requires_grad} -> {requires_grad}"
                )
            module.weight.requires_grad_(requires_grad)
        except AttributeError:
            res['unchange'].append(f"{path}")
    return res


# NOTE Trainer Helper
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        """ """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """

        Args:
          val: param n:  (Default value = 1)
          n: Default value = 1)

        Returns:

        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# NOTE Trainer Helper
class ProgressMeter(object):

    def __init__(self, losses, acc, time, color=None):
        self.losses_meters = losses
        self.acc_meters = acc
        self.time_meters = time
        self.color = color

    def status(self, epoch, max_epoch, it, max_it, mode: str) -> str:
        progress = '{} [{:3d}/{:3d}]'.format(mode, epoch, max_epoch)
        progress += ' {:3d}'.format(int(it / max_it * 100)) if mode == 'T' else '    '
        if len(self.losses_meters):
            losses = (' {} {:4.2f}' * len(self.losses_meters)).format(
                *reduce(lambda x, y: x + y,
                        [(m.name, m.avg) for m in self.losses_meters]))
        else:
            losses = ' NULL'
        if len(self.acc_meters):
            acc = (' {} {:5.1f}' * len(self.acc_meters)).format(
                *reduce(lambda x, y: x + y,
                        [(m.name, m.avg) for m in self.acc_meters]))
        else:
            acc = ' NULL'
        time = (' {} {:6.2f}s/it ' * len(self.time_meters)).format(
            *reduce(lambda x, y: x + y,
                    [(m.name, m.avg) for m in self.time_meters]))

        # FIXME
        eta = get_eta_time(max_epoch * max_it,
                           epoch * max_it + it,
                           self.time_meters[0].avg)

        msg = f'{progress} |{losses} |{acc} | ETA {eta}{time}'
        if self.color is None:
            return msg
        elif self.color == 'blue':
            return colors.blue(msg)
        elif self.color == 'red':
            return colors.red(msg)
        else:
            return colors.random(msg)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified
    values of k. The code refer to pytorch offical tutorials.

    Args:
      output (Tensor): the number of tensor are batch_size
      target (Tensor): true labels correspond to output
      topk (Tuple): Default value = (1,) indicates that will get top1 acc
        and (1, 5) then is to get top1 and top5 acc

    Returns:
      res (list):
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        # row to row compare pred with target
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
