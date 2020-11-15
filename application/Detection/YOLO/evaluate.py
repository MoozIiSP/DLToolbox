import os
import sys

from torch.autograd import Variable

from toolbox.application.Detection.YOLO.utils.bbox import bbox_iou, xywh2xyxy

sys.path.append("/home/aliclotho/GitRepos/DLToolbox/")

import torch
import numpy as np
import tqdm

from toolbox.application.Detection.YOLO.datasets import custom
from toolbox.application.Detection.YOLO.models.darknet import darknet53
from toolbox.application.Detection.YOLO.models.yolo import yolov, yolov3
from toolbox.application.Detection.YOLO.utils.utils import non_max_suppression


@torch.no_grad()
def evaluate(net, dataloader, conf_thres, iou_thres, nms_thres, device, logger):
    net.eval()

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (imgs, targets) in enumerate(tqdm.tqdm(validloader, desc="Detecting objects")):

        with torch.no_grad():
            outputs = net(imgs)
            outputs = non_max_suppression(outputs, conf_thres=0.5, nms_thres=0.3)

        labels += [target['labels'] for target in targets]
        sample_metrics += get_batch_statistics(imgs.size(-1), outputs, targets, iou_threshold=0.5)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [torch.cat(x, 0) for x in list(zip(*sample_metrics))]
    labels = torch.cat(labels, 0)
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    evaluation_metrics = {
        "val_precision": precision.mean().item(),
        "val_recall": recall.mean().item(),
        "val_mAP": AP.mean().item(),
        "val_f1": f1.mean().item(),
    }

    print('\nEval')
    for k, v in evaluation_metrics.items():
        print(f'{k} {v:.4f}')


def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.

    true_positives, pred_scores, pred_labels, labels
    """
    # Sort by objectness (scores)
    idx = torch.argsort(conf, descending=True)
    tp, conf, pred_cls = tp[idx], conf[idx], pred_cls[idx]

    # Find unique classes
    unique_classes = torch.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).flatten().cumsum(0)
            tpc = (tp[i]).flatten().cumsum(0)

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.long()


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_batch_statistics(im_size, predictions, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    FloatTensor = torch.cuda.FloatTensor if device.type == 'cuda' else torch.FloatTensor

    batch_metrics = []
    for idx, pred in enumerate(predictions):

        # if outputs[sample_i] is None:
        #     continue

        target = targets[idx]
        target_boxes = xywh2xyxy(target['boxes']) * im_size
        target_labels = target['labels']

        pred_boxes = pred[:, :4]
        pred_scores = pred[:, 4]
        pred_labels = pred[:, -1]

        # true positive table
        true_positives = torch.zeros(pred_boxes.shape[0])

        # existed predicted box
        if len(target_boxes):
            detected_boxes = []

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If all targets are found break
                if len(detected_boxes) == len(target_boxes):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                # add box according to iou_threshold
                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]

        batch_metrics.append([true_positives, pred_scores, pred_labels])

    return batch_metrics


if __name__ == '__main__':
    device = torch.device("cpu")

    if os.path.exists('/home/mooziisp'):
        data_root = '/home/mooziisp/Downloads/人体脊椎MRI图像/part3/labels-coco/'
    else:
        data_root = '/home/aliclotho/GitRepos/yolact/data/custom/'

    size = (3, 512, 512)
    b = 4

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

    net = yolov3(backbone=darknet53,
                 inp_dim=(3, 512, 512),
                 num_classes=80)
    # load existed weights
    net.load_state_dict(torch.load('weights/spine-512-80.pt', map_location='cpu'), strict=False)



