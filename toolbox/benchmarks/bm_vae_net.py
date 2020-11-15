import argparse
import os
import sys
import time

# toolbox root directary
sys.path.append(os.path.abspath('../..'))

from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from toolbox.core.utils.logger import setup_logger
from toolbox.core.utils.utils import *
from toolbox.benchmarks.config import *


def main(args):
    logger = setup_logger(__name__)
    logger.disabled = args.logger

    logger.info('AE/VAE train')

    # NOTE Define a transform for dataset to increase number of dataset and enhance data
    logger.info('define transform')
    transform = transforms.Compose([
        # transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.RandomAffine(0, (3/224, 3/224)),
        transforms.ToTensor()
    ])
    transform_t = transforms.Compose([
        # transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])

    # NOTE define data loader to load data for training
    logger.info('define dataset')
    trainset = datasets.CIFAR10(
        root=args.data,
        train=True,
        download=True,
        transform=transform)
    testset = datasets.CIFAR10(
        root=args.data,
        train=False,
        download=True,
        transform=transform_t)
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True)

    device = get_device(logger, cudnn_benchmark=True)
    # Benchmark multi net once
    for model in VAES:
        # unpack model
        net_fn = model['net']['fn']
        net_params = model['net']['kwargs']
        criterions_fn = [x['fn'] for x in model['criterion']]
        criterions_params = [x['kwargs'] for x in model['criterion']]
        optimizer_fn = model['optimizer']['fn']
        optimizer_params = model['optimizer']['kwargs']

        # FIXME
        if not model.get('pretrained'):
            logger.info(f'create {net_fn.__name__} net')
            net = net_fn(**net_params)
        else:
            logger.info(f'load existed weight of {net_fn.__name__}')
            net = torch.load(args.weights, map_location=torch.device(device))
        net.to(device)

        # compute FLOPs and MACs
        params, flops = arch_stat(net, 32, device, None)
        logger.info(f'Params {params}, FLOPs {flops}')

        if not args.tensorboard:
            # enable tensorboard
            message = f"run/{time.strftime('%b%d_%H%M%S', time.gmtime())}" + \
                      f"_{net_fn.__name__}_{trainset.__class__.__name__}" + \
                      f"_P{params}_F{flops}"
            logger.info("tensorboard hooked and log to " + message)
            writer = SummaryWriter(message)
        else:
            writer = None

        criterions = [fn().to(device) for fn in criterions_fn]
        optimizer = optimizer_fn(net.parameters(), **optimizer_params)
        # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=0.5)

        for epoch in range(args.epoch):
            train(epoch, train_loader, net, criterions, optimizer,
                  writer, logger, device)

            # NOTE change learning rate
            # lr_scheduler.step()

            eval(epoch, train_loader, net, criterions, optimizer,
                 writer, logger, device)

            # NOTE if TensorBoard don't automatically update, please refer to
            #  https://github.com/pytorch/pytorch/issues/24234#issuecomment-521024418
            if not args.tensorboard:
                writer.flush()

        # Save the model?
        if args.save:
            if not os.path.exists('weights'):
                os.mkdir('weights')
            logger.info('save weights of net')
            torch.save(net.state_dict(),
                       f"weights/{time.strftime('%b%d_%H%M%S', time.gmtime())}" +
                       f"_{net_fn.__name__}_{trainset.__class__.__name__}" +
                       f"_P{params}_F{flops}.pth")

        if not args.tensorboard:
            writer.close()


def train(epoch, dataloader, net, criterion, optimizer,
          writer, logger, device):
    # Meter Params
    losses1 = AverageMeter('Lv')
    losses2 = AverageMeter('Lb')
    losses = AverageMeter('Loss')
    tiktok = AverageMeter('Tiktok')
    loader = AverageMeter('Data')
    pregress = ProgressMeter([losses, losses1, losses2],
                             [],
                             [tiktok, loader])

    net.train(True)

    end = time.perf_counter()
    interval = int(len(dataloader) * 0.3 + 0.5)
    for it, (inputs, _) in enumerate(dataloader):
        loader.update(time.perf_counter() - end)
        inputs, target = inputs.to(device), _.to(device)

        tik = time.perf_counter()

        # zero the optimizer gradient
        optimizer.zero_grad()

        # forward, compute pred and loss
        # NOTE Compute loss
        output, recon_x, x, mu, logvar = net(inputs)
        loss1 = criterion[0](recon_x, x, mu, logvar)
        loss2 = criterion[1](output, inputs)
        loss = loss1 + loss2

        # NOTE measure accuracy and loss
        losses1.update(loss1.item(), inputs.size(0))
        losses2.update(loss2.item(), inputs.size(0))
        losses.update(loss.item(), inputs.size(0))

        # backward, compute gradient and do optimizer step
        loss1.backward()
        optimizer.step()

        tiktok.update(time.perf_counter() - tik)
        # NOTE Logging & Analyzer
        if args.tensorboard:
            # plot_gradient(writer, net)
            # for tag, value in filter(lambda p: 'conv' in p[0],
            #                          net.named_parameters()):
            #     tag = tag.replace('.', '/')
            #     writer.add_histogram(tag, value.data.cpu().numpy(),
            #                          epoch * len(train_loader) + it)
            #     writer.add_histogram(tag+'/grad', value.grad.data.cpu().numpy(),
            #                          epoch * len(train_loader) + it)
            writer.add_scalar(f'loss/train', losses.avg, epoch * len(dataloader) + it)
        if it % interval == 0:
            logger.info(pregress.status(epoch, args.epoch, it, len(dataloader), mode='T'))
            # logger.info(f'T [{epoch:2d}/{args.epoch}] {it / len(dataloader) * 100:02.0f}' +
            #             f' | Loss {losses.avg:02.2f}' +
            #             f' | Top@1 {acc1.avg:03.2f} Top@5 {acc5.avg:03.2f}' +
            #             f' | ETA {get_eta_time(args.epoch * len(dataloader), cur, tiktok.avg)}' +
            #             f' ({tiktok.avg * interval:.2f} {loader.avg:.2f})')
            # dataload time measure
        end = time.perf_counter()


def eval(epoch, dataloader, net, criterion, optimizer, writer, logger, device):
    # Meter Params
    losses1 = AverageMeter('Lv')
    losses2 = AverageMeter('Lb')
    losses = AverageMeter('Loss')
    tiktok = AverageMeter('Tiktok')
    loader = AverageMeter('Data')
    pregress = ProgressMeter([losses, losses1, losses2],
                             [],
                             [tiktok, loader],
                             color='random')

    # NOTE Plot Confusion Matrix
    # cm = ConfusionMatrix(len(trainset.classes), trainset.classes)

    net.train(False)

    end = time.perf_counter()
    with torch.no_grad():
        for it, (inputs, _) in enumerate(dataloader):
            loader.update(time.perf_counter() - end)
            inputs, target = inputs.to(device), _.to(device)

            tik = time.perf_counter()

            # zero the optimizer gradient
            optimizer.zero_grad()

            # forward, compute pred and loss
            # NOTE Compute loss
            output, recon_x, x, mu, logvar = net(inputs)
            loss1 = criterion[0](recon_x, x, mu, logvar)
            loss2 = criterion[1](output, inputs)
            loss = loss1 + loss2

            # NOTE measure accuracy and loss
            losses1.update(loss1.item(), inputs.size(0))
            losses2.update(loss2.item(), inputs.size(0))
            losses.update(loss.item(), inputs.size(0))

            tiktok.update(time.perf_counter() - tik)
            # NOTE Logging & Analyzer
            if args.tensorboard:
                writer.add_scalar('loss/valid', losses.avg,
                                  epoch * len(dataloader) + it)
            if it == len(dataloader) - 1:
                logger.info(pregress.status(epoch, args.epoch, it, len(dataloader), mode='V'))
                # logger.info(red(f'T [{epoch:2d}/{args.epoch}] {it / len(dataloader) * 100:02.0f}' +
                #                 f' | Loss {losses.avg:02.2f}' +
                #                 f' | Top@1 {acc1.avg:03.2f} Top@5 {acc5.avg:03.2f}' +
                #                 f' | ETA {get_eta_time(args.epoch * len(dataloader), cur, tiktok.avg)}' +
                #                 f' ({tiktok.avg * len(dataloader):.2f} {loader.avg:.2f})'))

    # logger.info('plot confusion matrix to tensorboard')
    # cm_image = cm.plot_to_tensorborad()
    # writer.add_image('Confusion_Matrix', cm_image, epoch * len(dataloader))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='benchmarks of neural networks.')

    parser.add_argument('data', metavar='DATA',
                        help='path to data')
    parser.add_argument('--batch_size', metavar='batch_size', default=BATCH_SIZE,
                        type=int)
    parser.add_argument('--epoch', metavar='epoch', default=EPOCH,
                        type=int)
    parser.add_argument('--no-logger', dest='logger', action='store_true',
                        help='enable logger')
    parser.add_argument('--no-tensorboard', dest='tensorboard', action='store_true',
                        help='enable tensorboard')
    parser.add_argument('--save', dest='save', action='store_true',
                        help='save weight of net')

    args = parser.parse_args()
    main(args)
