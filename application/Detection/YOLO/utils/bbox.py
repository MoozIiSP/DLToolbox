import torch


def xywh2xyxy(x):
    """x: tensor"""
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def xyxy2xywh(x):
    pass


def bbox_wh_iou(wh1, wh2):
    # FIXME temp
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


# REF:
# https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
def bbox_iou(box, boxes, xywh=True):
    """
    Returns the IoU of two bounding boxes, and the tensor
     contains any number of bounding boxes coordinate (x, y, w, h).

    :param box: (1, 4) array
    :param boxes: (n, 4) array
    :return:
    """

    assert len(box.shape) == 2 and len(boxes.shape) == 2, \
        f'box ({box.shape}) or boxes ({boxes.shape}) must be array with (n, 4) shape'

    if xywh:
        box = xywh2xyxy(box)
        boxes = xywh2xyxy(boxes)

    b1_x1, b1_y1, b1_x2, b1_y2 = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou