import math
import sys, os
import time



sys.path.append("/home/aliclotho/GitRepos/DLToolbox/")

import torch
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from toolbox.application.Detection.YOLO.models.yolo import *
from toolbox.application.Detection.YOLO.models.backbone import resnet, vgg, mobilenet, densenet
from toolbox.application.Detection.YOLO.models.darknet import darknet53
from toolbox.application.Detection.YOLO.datasets import custom
from toolbox.application.Detection.YOLO.models import *
from toolbox.application.Detection.YOLO.utils.utils import *
from toolbox.core.utils import AverageMeter, arch_stat

if os.path.exists('/home/mooziisp'):
    data_root = '/home/mooziisp/Downloads/人体脊椎MRI图像/part3/labels-coco/'
else:
    data_root = '/home/aliclotho/GitRepos/yolact/data/custom/'


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def tarin_one_epoch(net, optimizer, dataloader, device, logger, epoch):
    # Meter Params
    losses = AverageMeter('Loss')
    lr = AverageMeter('Lr')

    net.train()

    # warmup lr
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 10000
        warmup_iters = min(10000, len(dataloader) - 1)
        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
        print(f'warmup learning rate, factor {warmup_factor}, iters {warmup_iters}')

    for i, (inputs, targets) in tqdm.tqdm(enumerate(dataloader), desc="Train"):
        inputs = inputs.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs, loss = net(inputs, targets)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        losses.update(loss_value, 1)
        lr.update(optimizer.param_groups[0]['lr'])
        logger.add_scalar(f'train/loss', losses.avg, epoch * len(dataloader) + i)
        logger.add_scalar(f'train/lr', lr.avg, epoch * len(dataloader) + i)
        print(f'\nepoch {epoch} iters {epoch * len(dataloader) * b + i * b} Loss {loss.item():.2f}')
        print('lr ', [pg['lr'] for pg in optimizer.state_dict()['param_groups']])
        metrics = [net.detection1.metrics.items(),
                   net.detection2.metrics.values(),
                   net.detection3.metrics.values()
                   ]
        for pos1, pos2, pos3 in zip(*metrics):
            if pos1[0] == 'grid_size':
                print(f'{pos1[0]} {pos1[1]} {pos2} {pos3}')
                continue
            print(('{:8s} ' + (' {:4.2f}' * 3)).format(pos1[0], pos1[1], pos2, pos3))


@torch.no_grad()
def evaluate(net, dataloader, conf_thres, iou_thres, nms_thres, device, logger):
    net.eval()

    labels = []
    sample_metrics = []  # (TP, confs, pred)
    for i, (inputs, targets) in tqdm.tqdm(enumerate(dataloader), desc="Eval"):
        inputs = inputs.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        detections = net(inputs).cpu()  # (cx, cy, w, h, conf, class)
        detections = non_max_suppression(detections, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(detections, targets,
                                               iou_threshold=iou_thres, device=device)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [torch.cat(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, mAP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    evaluation_metrics = {
        "val_precision": precision.mean().item(),
        "val_recall": recall.mean().item(),
        "val_mAP": mAP.mean().item(),
        "val_f1": f1.mean().item(),
    }

    print('\nEval')
    for k, v in evaluation_metrics:
        print(f'{k} {v:.2f}')


if __name__ == '__main__':
    size = (3, 512, 512)
    b = 4

    trainset = custom.SpineDetection(
        data_root + 'train/',
        data_root + 'train/annotations.json',
        target_size=size,
        bg_class=False,
        multiscale=False  # FIXME BUG will lead to out of memory
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=b,  # FIXME
        shuffle=True,
        collate_fn=trainset.collate_fn,
        num_workers=1
    )

    validset = custom.SpineDetection(
        data_root + 'valid/',
        data_root + 'valid/annotations.json',
        target_size=size,
        bg_class=False,
        multiscale=False
    )
    validloader = torch.utils.data.DataLoader(
        validset,
        batch_size=b//2,  # FIXME
        shuffle=False,
        collate_fn=validset.collate_fn,
        num_workers=1
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # net = YOLO(masks=((6, 7, 8), (3, 4, 5), (0, 1, 2)),
    #            anchors=((10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
    #                       (59, 119), (116, 90), (156, 198), (373, 326)),
    #            inp_dim=size, num_classes=2)
    net = yolov(backbone=darknet53,
                pretrained=True,
                inp_dim=(3, 512, 512),
                num_classes=2)
    net.to(device)

    params, flops = arch_stat(net, size[1], device, None)
    tag = f"{time.strftime('%b%d_%H%M%S', time.gmtime())}" + \
          f"_{net.__class__.__name__}:{net.backbone.__class__.__name__}_{trainset.__class__.__name__}" + \
          f"_P{params}_F{flops}"
    writer = SummaryWriter('run/' + tag)
    print(tag)

    # net.apply(weights_init_normal)
    optimizer = optim.SGD(net.parameters(), lr=5e-3, momentum=0.9, weight_decay=5e-4)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                                step_size=3,
    #                                                gamma=0.1)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                  milestones=[5, 25],
                                                  gamma=0.5)
    # torch.autograd.set_detect_anomaly(True)

    for epoch in range(75):
        tarin_one_epoch(net, optimizer, trainloader, device, writer, epoch)

        lr_scheduler.step()

        #evaluate(net, validloader,
        #         iou_thres=0.5, conf_thres=0.5, nms_thres=0.5, device=device)

    if not os.path.exists('weights'):
        os.makedirs('weights')
    torch.save(net.state_dict(), f'weights/{tag}-test.pt')
    print('save weights.')
