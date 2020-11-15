import types
import typing
import argparse


def build_parser() -> argparse.ArgumentParser:
    """
    parser will accept the following argument:
      - data :: data dir
      -

    :return: parser
    """
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR', nargs='*',
                        help='path(s) to dataset (if one path is provided, it is assumed\n' +
                        'to have subdirectories named "train" and "val"; alternatively,\n' +
                        'train and val paths can be specified directly by providing both paths as arguments)')
    # parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
    #                     choices=model_names,
    #                     help='model architecture: ' +
    #                     ' | '.join(model_names) +
    #                     ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--fp16', action='store_true',
                        help='Run model fp16 mode.')
    parser.add_argument('--dali_cpu', action='store_true',
                        help='Runs CPU based version of DALI pipeline.')
    parser.add_argument('--static-loss-scale', type=float, default=1,
                        help='Static loss scale, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--dynamic-loss-scale', action='store_true',
                        help='Use dynamic loss scaling.  If supplied, this argument supersedes ' +
                        '--static-loss-scale.')
    parser.add_argument('--prof', dest='prof', action='store_true',
                        help='Only run 10 iterations for profiling.')
    parser.add_argument('-t', '--test', action='store_true',
                        help='Launch test mode with preset arguments')

    parser.add_argument("--local_rank", default=0, type=int)

    return parser
