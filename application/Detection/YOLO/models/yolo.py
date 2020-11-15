"""大部分代码参考自https://github.com/BobLiu20/YOLOv3_PyTorch以及..."""
from __future__ import absolute_import

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from toolbox.application.Detection.YOLO.models.darknet import ConvBlock, conv1x1, conv3x3
from toolbox.application.Detection.YOLO.models.backbone import resnet, vgg, mobilenet, densenet
from toolbox.application.Detection.YOLO.models.darknet import darknet53backbone
from toolbox.application.Detection.YOLO.utils.bbox import bbox_wh_iou, bbox_iou
from toolbox.application.Detection.YOLO.utils.utils import *

# from darknet import BaseBlock, darknet53, conv1x1, conv3x3

__all__ = ['YOLO', 'yolov3', 'YOLOv', 'yolov']


# TODO need to support any size image
class YOLOPredictor(nn.Module):
    def __init__(self, mask, anchors, img_size, num_classes):
        super(YOLOPredictor, self).__init__()
        self.anchors = [anchors[i] for i in mask]
        self.num_anchors = len(self.anchors)  # default: 3
        self.num_classes = num_classes
        self.bbox_attrs = num_classes + 5
        if type(img_size) is int:
            self.img_size = (img_size, img_size)
        else:
            self.img_size = img_size

        self.ignore_thres = 0.5
        # self.obj_scale = 1
        # self.noobj_scale = 100
        # self.grid_size = 0
        self.lambda_coord = 1.  # FIXME
        self.lambda_conf = 100.  # FIXME -> .5
        self.lambda_cls = 1.0

        # Loss
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, x, targets=None):
        batch_size = x.size(0)
        grid_h = x.size(2)
        grid_w = x.size(3)
        stride_h = self.img_size[0] / grid_h
        stride_w = self.img_size[1] / grid_w
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]

        # tensor(*, anchors, bbox_attrs, cell, cell)
        x = (
            x.view(batch_size, self.num_anchors, self.bbox_attrs, grid_h, grid_w)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
        )

        # Get outputs
        c_x = torch.sigmoid(x[..., 0])  # Center x
        c_y = torch.sigmoid(x[..., 1])  # Center y
        w = x[..., 2]  # Width
        h = x[..., 3]  # Height
        pred_conf = torch.sigmoid(x[..., 4])  # Conf
        pred_cls = torch.sigmoid(x[..., 5:])  # Cls pred.

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        # Calculate offsets for each grid
        grid_x = torch.linspace(0, grid_w - 1, grid_w).repeat(grid_w, 1).repeat(
            batch_size * self.num_anchors, 1, 1).view(c_x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, grid_h - 1, grid_h).repeat(grid_h, 1).t().repeat(
            batch_size * self.num_anchors, 1, 1).view(c_y.shape).type(FloatTensor)

        # Calculate anchor w, h
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0])) \
            .repeat(batch_size, 1).repeat(1, 1, grid_h * grid_w).view(w.shape)
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1])) \
            .repeat(batch_size, 1).repeat(1, 1, grid_h * grid_w).view(h.shape)

        # add anchor to every (centre_x, centre_y)
        # NOTE tensor.repeat(dim1, dim2) indicate that repeat row or column element dim1 or dim2 times
        pred_boxes = FloatTensor(x[..., :4].shape)
        # pred_boxes = x[..., :4]
        pred_boxes[..., 0] = c_x.data + grid_x
        pred_boxes[..., 1] = c_y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        _scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)
        # print(_scale)
        output = torch.cat((
            pred_boxes.view(batch_size, -1, 4) * _scale,
            pred_conf.view(batch_size, -1, 1),
            pred_cls.view(batch_size, -1, self.num_classes),
        ), -1,
        )

        # FIXME
        if targets is None:
            return output, 0
        else:
            # FIXME - batch_size must be 1
            # print(len(targets))
            mask, noobj_mask, tx, ty, tw, th, tconf, tcls, class_mask, iou_scores = self.compute_loss(
                targets,
                pred_boxes,
                pred_cls,
                scaled_anchors,
                grid_w, grid_h,
                self.ignore_thres
            )

            # loss funtion := \lambda_{\text{coord}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}}
            # \left[ (x_i - \hat{x_i})^2 + (y_i - \hat{y}_i)^2 \right] \\ + \lambda_{\text{coord}} \sum_{i=0}^{S^2}
            # \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} \left[ (\sqrt{w_i} - \sqrt{\hat{w_i}})^2 + (\sqrt{h_i} -
            # \sqrt{\hat{h}_i})^2 \right] \\ + \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} (C_i -
            # \hat{C}_i)^2 + \lambda_{\text{noobj}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{noobj}}
            # (C_i - \hat{C}_i)^2 + \sum_{i=0}^{S^2} \mathbb{1}_{i}^{\text{noobj}} \sum_{c \in \text{classes}}
            # (p_i(c) - \hat{p}_i(c))^2
            loss_x = self.mse_loss(c_x[mask], tx[mask])
            loss_y = self.mse_loss(c_y[mask], ty[mask])
            loss_w = self.mse_loss(w[mask], tw[mask])
            loss_h = self.mse_loss(h[mask], th[mask])
            # loss_conf = self.bce_loss(pred_conf[mask], tconf[mask]) + \
            #             0.5 * self.bce_loss(pred_conf * noobj_mask, noobj_mask * 0.0)
            loss_conf_obj = self.bce_loss(pred_conf[mask], tconf[mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = loss_conf_obj + self.lambda_conf * loss_conf_noobj
            loss_cls = self.bce_loss(pred_cls[mask], tcls[mask])
            # total loss = losses * weight
            loss = (loss_x + loss_y + loss_w + loss_h) * self.lambda_coord + \
                   loss_conf + loss_cls * self.lambda_cls

            cls_acc = 100 * class_mask[mask].mean()
            conf_obj = pred_conf[mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (mask.sum() + 1e-16)

            self.metrics = {
                "grid_size": "{}x{}".format(grid_w, grid_h),
                "loss": loss.item(),
                "x": loss_x.item(),
                "y": loss_y.item(),
                "w": loss_w.item(),
                "h": loss_h.item(),
                "conf": loss_conf.item(),
                "cls": loss_cls.item(),
                "cls_acc": cls_acc.item(),
                "recall50": recall50.item(),
                "recall75": recall75.item(),
                "precision": precision.item(),
                "conf_obj": conf_obj.item(),
                "conf_noobj": conf_noobj.item(),
            }

            return output, loss

            # iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
            #     pred_boxes=pred_boxes,
            #     pred_cls=pred_cls,
            #     targets=targets,
            #     anchors=self.scaled_anchors,
            #     ignore_thres=self.ignore_thres,
            # )

            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            # loss_x = self.mse_loss(centre_x[obj_mask], tx[obj_mask])
            # loss_y = self.mse_loss(centre_y[obj_mask], ty[obj_mask])
            # loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            # loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            # print(loss_x, loss_y, loss_w, loss_h)
            # loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            # loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            # loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            # loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            # total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

    def compute_loss(self, targets, pred_boxes, pred_cls, anchors, grid_w, grid_h, ignore_threshold):
        batch_size = len(targets)

        # single gpu
        BoolTensor = torch.cuda.BoolTensor if pred_boxes.is_cuda else torch.BoolTensor
        FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor
        device = torch.device('cuda' if pred_boxes.is_cuda else 'cpu')

        # 网格中存在目标的掩码，存在则置1
        # 其中obj_mask是存在目标的掩码，而noobj_mask反之
        obj_mask = BoolTensor(batch_size, self.num_anchors, grid_h, grid_w).fill_(0)
        noobj_mask = BoolTensor(batch_size, self.num_anchors, grid_h, grid_w).fill_(1)
        # offset value
        tx = FloatTensor(batch_size, self.num_anchors, grid_h, grid_w).fill_(0)
        ty = FloatTensor(batch_size, self.num_anchors, grid_h, grid_w).fill_(0)
        tw = FloatTensor(batch_size, self.num_anchors, grid_h, grid_w).fill_(0)
        th = FloatTensor(batch_size, self.num_anchors, grid_h, grid_w).fill_(0)
        # 置信度、分类
        tcls = FloatTensor(batch_size, self.num_anchors, grid_h, grid_w, self.num_classes).fill_(0)
        # tconf
        # metrics
        cls_mask = FloatTensor(batch_size, self.num_anchors, grid_h, grid_w).fill_(0)
        iou_scores = FloatTensor(batch_size, self.num_anchors, grid_h, grid_w).fill_(0)

        # mask = torch.zeros(batch_size, self.num_anchors, grid_h, grid_w, requires_grad=False).to(device)
        # noobj_mask = torch.ones(batch_size, self.num_anchors, grid_h, grid_w, requires_grad=False).to(device)
        # tx = torch.zeros(batch_size, self.num_anchors, grid_h, grid_w, requires_grad=False).to(device)
        # ty = torch.zeros(batch_size, self.num_anchors, grid_h, grid_w, requires_grad=False).to(device)
        # tw = torch.zeros(batch_size, self.num_anchors, grid_h, grid_w, requires_grad=False).to(device)
        # th = torch.zeros(batch_size, self.num_anchors, grid_h, grid_w, requires_grad=False).to(device)
        # tconf = torch.zeros(batch_size, self.num_anchors, grid_h, grid_w, requires_grad=False).to(device)
        # tcls = torch.zeros(batch_size, self.num_anchors, grid_h, grid_w, self.num_classes, requires_grad=False).to(
        #     device)
        #
        # class_mask = torch.zeros(batch_size, self.num_anchors, grid_h, grid_w,
        #                          requires_grad=False).to(device)
        # iou_scores = torch.zeros(batch_size, self.num_anchors, grid_h, grid_w,
        #                          requires_grad=False).to(device)

        # Output tensors
        # mask = FloatTensor(batch_size, self.num_anchors, grid_h, grid_w).fill_(0)
        # noobj_mask = FloatTensor(batch_size, self.num_anchors, grid_h, grid_w).fill_(1)
        # class_mask = FloatTensor(batch_size, self.num_anchors, grid_h, grid_w).fill_(0)
        # iou_scores = FloatTensor(batch_size, self.num_anchors, grid_h, grid_w).fill_(0)
        # tx = FloatTensor(batch_size, self.num_anchors, grid_h, grid_w).fill_(0)
        # ty = FloatTensor(batch_size, self.num_anchors, grid_h, grid_w).fill_(0)
        # tw = FloatTensor(batch_size, self.num_anchors, grid_h, grid_w).fill_(0)
        # th = FloatTensor(batch_size, self.num_anchors, grid_h, grid_w).fill_(0)
        # tcls = FloatTensor(batch_size, self.num_anchors, grid_h, grid_w, self.num_classes).fill_(0)

        for b in range(batch_size):
            for bbox, label in zip(*(targets[b]['boxes'], targets[b]['labels'])):
                # FIXME dataloader should generate bounding box of equal length
                if bbox.sum() == 0:
                    continue

                # Convert [0,1] to position relative to box
                gx = bbox[0] * grid_w
                gy = bbox[1] * grid_h
                gw = bbox[2] * grid_w
                gh = bbox[3] * grid_h

                # Get grid box indices
                gi = gx.long()
                gj = gy.long()

                # Get shape of gt box
                # gt_box = FloatTensor([0, 0, gw, gh]).unsqueeze(0)
                gwh = FloatTensor([gw, gh])
                ious = torch.stack([bbox_wh_iou(FloatTensor(anchor), gwh) for anchor in anchors])
                # Where the overlap is larger than threshold set mask to zero (ignore)
                # 大于阈值，置0，意味着该网格存在目标
                noobj_mask[b, ious > ignore_threshold, gj, gi] = 0
                # Find the best matching anchor box
                best_n = torch.argmax(ious)

                # Set masks
                obj_mask[b, best_n, gj, gi] = 1
                noobj_mask[b, best_n, gj, gi] = 0

                # Coordinates
                tx[b, best_n, gj, gi] = gx - gi
                ty[b, best_n, gj, gi] = gy - gj
                # Width and height
                tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][0] + 1e-16)
                th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][1] + 1e-16)
                # One-hot encoding of label
                tcls[b, best_n, gj, gi, label] = 1
                # compute label correctness
                cls_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == label).float()
                iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi].unsqueeze(0).to(device),
                                                         FloatTensor([gx, gy, gw, gh]).unsqueeze(0),
                                                         # bbox.unsqueeze(0),
                                                         xywh=True)

        tconf = obj_mask.float()
        return obj_mask, noobj_mask, tx, ty, tw, th, tconf, tcls, cls_mask, iou_scores


class FPNDetection(nn.Module):
    def __init__(self, inplanes, planes=512, num_anchors=3, num_classes=80, upsample=None):
        super(FPNDetection, self).__init__()
        self.upsample = upsample
        self.conv1 = ConvBlock(conv1x1, inplanes, planes, stride=1, padding=0)
        self.conv2 = ConvBlock(conv3x3, planes, planes * 2, stride=1, padding=1)
        self.conv3 = ConvBlock(conv1x1, planes * 2, planes, stride=1, padding=0)
        self.conv4 = ConvBlock(conv3x3, planes, planes * 2, stride=1, padding=1)
        self.conv5 = ConvBlock(conv1x1, planes * 2, planes, stride=1, padding=0)
        self.conv6 = ConvBlock(conv3x3, planes, planes * 2, stride=1, padding=1)
        self.conv7 = conv1x1(planes * 2, num_anchors * (num_classes + 5), stride=1, padding=0, bias=True)  # Linear
        self.x_cache = {}

    def forward(self, x, cache=None):
        if self.upsample:
            x = self.upsample(x)
        if cache is not None:
            x = torch.cat((x, cache), 1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)  # pass to next FPN

        # cache
        self.x_cache['next_fpn'] = x

        x = self.conv6(x)
        x = self.conv7(x)
        return x


class FPNv(nn.Module):
    def __init__(self, feat_dim=(256, 512, 1024), planes=128, num_anchors=3, num_classes=80):
        super(FPNv, self).__init__()
        # upsample
        self.upsample1 = nn.Sequential(
            ConvBlock(conv1x1, 2 * feat_dim[1], feat_dim[0], 1, 0),  # conv1x1(256, 128, 1, 0),
            nn.Upsample(scale_factor=2, mode="nearest"))
        self.upsample2 = nn.Sequential(
            ConvBlock(conv1x1, feat_dim[2], feat_dim[1], 1, 0),  # conv1x1(256, 128, 1, 0),
            nn.Upsample(scale_factor=2, mode="nearest"))
        # Union
        self.union_conv1 = ConvBlock(conv1x1, feat_dim[0] * 2, planes * 2, stride=1, padding=0)
        self.union_conv2 = ConvBlock(conv1x1, feat_dim[1] * 2, planes * 2, stride=1, padding=0)
        self.union_conv3 = ConvBlock(conv1x1, feat_dim[2], planes * 2, stride=1, padding=0)
        # Conv Group 1
        self.conv1_1 = ConvBlock(conv1x1, 2 * planes, 2 * planes, stride=1, padding=0)
        self.conv1_2 = ConvBlock(conv3x3, 2 * planes, 2 * planes, stride=1, padding=1)
        self.conv1_3 = ConvBlock(conv1x1, 2 * planes, 2 * planes, stride=1, padding=0)
        self.conv1_4 = ConvBlock(conv3x3, 2 * planes, 2 * planes, stride=1, padding=1)
        self.conv1_5 = ConvBlock(conv1x1, 4 * planes, 2 * planes, stride=1, padding=0)
        self.conv1_6 = ConvBlock(conv3x3, 2 * planes, 2 * planes, stride=1, padding=1)
        self.conv1_7 = conv1x1(planes * 2, num_anchors * (num_classes + 5), stride=1, padding=0, bias=True)  # Linear
        # Conv Group 2
        self.conv2_1 = ConvBlock(conv1x1, 4 * planes, 2 * planes, stride=1, padding=0)
        self.conv2_2 = ConvBlock(conv3x3, 2 * planes, 2 * planes, stride=1, padding=1)
        self.conv2_3 = ConvBlock(conv1x1, 2 * planes, 2 * planes, stride=1, padding=0)
        self.conv2_4 = ConvBlock(conv3x3, 2 * planes, 2 * planes, stride=1, padding=1)
        self.conv2_5 = ConvBlock(conv1x1, 4 * planes, 2 * planes, stride=1, padding=0)
        self.conv2_6 = ConvBlock(conv3x3, 2 * planes, 2 * planes, stride=1, padding=1)
        self.conv2_7 = conv1x1(planes * 2, num_anchors * (num_classes + 5), stride=1, padding=0, bias=True)  # Linear
        # Conv Group 3
        self.conv3_1 = ConvBlock(conv1x1, 2 * planes, 2 * planes, stride=1, padding=0)
        self.conv3_2 = ConvBlock(conv3x3, 2 * planes, 2 * planes, stride=1, padding=1)
        self.conv3_3 = ConvBlock(conv1x1, 4 * planes, 2 * planes, stride=1, padding=0)
        self.conv3_4 = ConvBlock(conv3x3, 2 * planes, 2 * planes, stride=1, padding=1)
        self.conv3_5 = ConvBlock(conv1x1, 2 * planes, 2 * planes, stride=1, padding=0)
        self.conv3_6 = ConvBlock(conv3x3, 2 * planes, 2 * planes, stride=1, padding=1)
        self.conv3_7 = conv1x1(planes * 2, num_anchors * (num_classes + 5), stride=1, padding=0, bias=True)  # Linear

    def forward(self, x3, x2, x1, cache=None):
        # print(x3.shape, x2.shape, x1.shape)
        # first layer compute
        x = self.upsample2(x3)
        x2 = torch.cat((x, x2), 1)
        x = self.upsample1(x2)
        x1 = torch.cat((x, x1), 1)
        # second layer union
        x1 = self.union_conv1(x1)
        x2 = self.union_conv2(x2)
        x3 = self.union_conv3(x3)
        # third layer
        x1 = self.conv1_1(x1)
        x1 = self.conv1_2(x1)
        x3 = self.conv3_1(x3)
        x3 = self.conv3_2(x3)
        x2 = self.conv2_1(torch.cat((
            x2,
            nn.Upsample(scale_factor=0.5, mode='nearest')(x1)), 1))
        x2 = self.conv2_2(x2)
        # four layer
        x1 = self.conv1_3(x1)
        x1 = self.conv1_4(x1)
        x2 = self.conv2_3(x2)
        x2 = self.conv2_4(x2)
        x3 = self.conv3_3(torch.cat((
            x3,
            nn.Upsample(scale_factor=0.5, mode='nearest')(x2)), 1))
        x3 = self.conv3_4(x3)
        # five layer
        x3 = self.conv3_5(x3)
        x3 = self.conv3_6(x3)
        x2 = self.conv2_5(torch.cat((
            x2,
            nn.Upsample(scale_factor=2, mode='nearest')(x3)), 1))
        x2 = self.conv2_6(x2)
        x1 = self.conv1_5(torch.cat((
            x1,
            nn.Upsample(scale_factor=2, mode='nearest')(x2)), 1))
        x1 = self.conv1_6(x1)
        # detection
        x1 = self.conv1_7(x1)
        x2 = self.conv2_7(x2)
        x3 = self.conv3_7(x3)

        return x3, x2, x1


class YOLOv(nn.Module):
    def __init__(self, backbone, masks, anchors, inp_dim, feat_dim, num_classes):
        super(YOLOv, self).__init__()
        # net info
        self.inp_dim = inp_dim
        self.num_classes = num_classes
        self.num_anchors = len(masks[0])

        self.backbone = backbone

        self.fpn = FPNv(feat_dim, 128, self.num_anchors, self.num_classes)
        self.detection1 = YOLOPredictor(masks[0], anchors, inp_dim[1:], num_classes)
        self.detection2 = YOLOPredictor(masks[1], anchors, inp_dim[1:], num_classes)
        self.detection3 = YOLOPredictor(masks[2], anchors, inp_dim[1:], num_classes)

    def forward(self, x, targets=None):
        x = self.backbone(x)

        x3, x2, x1 = self.fpn(x,
                              self.backbone.cache['scale2'],
                              self.backbone.cache['scale1'])

        detection1, loss1 = self.detection1(x3, targets)

        detection2, loss2 = self.detection2(x2, targets)
        detection2 = torch.cat((detection1, detection2), dim=1)

        detection3, loss3 = self.detection3(x1, targets)
        detections = torch.cat((detection2, detection3), dim=1)

        if targets is None:
            return detections
        return detections, loss1 + loss2 + loss3


class YOLO(nn.Module):
    def __init__(self, backbone, masks, anchors, inp_dim, num_classes):
        super(YOLO, self).__init__()
        # net info
        self.inp_dim = inp_dim
        self.num_classes = num_classes
        self.num_anchors = len(masks[0])

        self.backbone = darknet53backbone(inplances=inp_dim[0],
                                          pretrained=True)
        # Detection Module #1
        self.fpn1 = FPNDetection(inplanes=1024, planes=512,
                                 num_anchors=self.num_anchors,
                                 num_classes=self.num_classes,
                                 upsample=None)
        self.detection1 = YOLOPredictor(masks[0], anchors, inp_dim[1:], num_classes)
        # Detection Module #2
        self.fpn2 = FPNDetection(
            inplanes=512 + 256, planes=256,
            num_anchors=self.num_anchors,
            num_classes=self.num_classes,
            upsample=nn.Sequential(
                ConvBlock(conv1x1, 512, 256, 1, 0),  # conv1x1(512, 256, 1, 0),
                nn.Upsample(scale_factor=2, mode="nearest"), )
        )
        self.detection2 = YOLOPredictor(masks[1], anchors, inp_dim[1:], num_classes)
        # Detection Module #3
        self.fpn3 = FPNDetection(
            inplanes=256 + 128, planes=128,
            num_anchors=self.num_anchors,
            num_classes=self.num_classes,
            upsample=nn.Sequential(
                ConvBlock(conv1x1, 256, 128, 1, 0),  # conv1x1(256, 128, 1, 0),
                nn.Upsample(scale_factor=2, mode="nearest"), )
        )
        self.detection3 = YOLOPredictor(masks[2], anchors, inp_dim[1:], num_classes)

    def forward(self, x, targets=None):
        x = self.backbone(x)

        # route 1
        pred = self.fpn1(x)
        detection1, loss1 = self.detection1(pred, targets)
        # route 2
        pred = self.fpn2(self.fpn1.x_cache['next_fpn'],
                         self.backbone.cache['scale2'])
        detection2, loss2 = self.detection2(pred, targets)
        detection2 = torch.cat((detection1, detection2), dim=1)
        # route 3
        pred = self.fpn3(self.fpn2.x_cache['next_fpn'],
                         self.backbone.cache['scale1'])
        detection3, loss3 = self.detection3(pred, targets)
        detections = torch.cat((detection2, detection3), dim=1)

        if targets is None:
            return detections
        return detections, loss1 + loss2 + loss3


def yolov3(backbone, inp_dim, num_classes):
    feat_dim = (256, 512, 1024)
    if backbone.__name__ == 'densenet121':
        feat_dim = (512, 1024, 1024)
    # elif backbone.__name__ == 'vgg11':
    #     faet_dim = (512, 512, 512)
    # elif backbone.__name__ == 'vgg13':
    #     faet_dim = (256, 512, 512)
    elif backbone.__name__ == 'resnet18':
        feat_dim = (128, 256, 512)
    elif backbone.__name__ == 'resnet34':
        feat_dim = (128, 256, 512)
    elif backbone.__name__ == 'resnet50':
        feat_dim = (128, 256, 512)
    elif backbone.__name__ == 'mobilenet':
        feat_dim = (128, 256, 512)
    else:  # default: darknet53
        feat_dim = (256, 512, 1024)

    net = YOLO(backbone,
               masks=((6, 7, 8), (3, 4, 5), (0, 1, 2)),
               anchors=((10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                        (59, 119), (116, 90), (156, 198), (373, 326)),
               inp_dim=inp_dim, num_classes=num_classes)
    return net


def yolov(backbone, pretrained, inp_dim, num_classes):
    # darknet53: 256, 512, 1024
    # resnet18: 128, 256, 512
    # resnet34: 256, 512, 1024
    # resnet50: 512, 1024, 2048
    # vgg11: 512, 512, 512
    # vgg13: 256, 512, 512
    # mobilenet v2: 32, 96, 160
    # squeezenet: 256, 512, 512
    feat_dim = (256, 512, 1024)
    if backbone.__name__ == 'densenet121':
        feat_dim = (512, 1024, 1024)
    # elif backbone.__name__ == 'vgg11':
    #     faet_dim = (512, 512, 512)
    # elif backbone.__name__ == 'vgg13':
    #     faet_dim = (256, 512, 512)
    elif backbone.__name__ == 'resnet18':
        feat_dim = (128, 256, 512)
    elif backbone.__name__ == 'resnet34':
        feat_dim = (128, 256, 512)
    elif backbone.__name__ == 'resnet50':
        feat_dim = (512, 1024, 2048)
    elif backbone.__name__ == 'mobilenet_v2':
        feat_dim = (32, 96, 160)
    else:  # default: darknet53
        feat_dim = (256, 512, 1024)

    net = YOLOv(backbone(pretrained=pretrained),
                masks=((6, 7, 8), (3, 4, 5), (0, 1, 2)),
                anchors=((10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)),
                inp_dim=inp_dim, feat_dim=feat_dim, num_classes=num_classes)
    return net


def weights_init_normal(m):
    def convblock_init_normal(module):
        torch.nn.init.normal_(m.conv.weight.data, 0.0, 0.02)
        torch.nn.init.normal_(m.bn.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bn.bias.data, 0.0)

    classname = m.__class__.__name__
    print(classname)
    if classname.find("ConvBlock") != -1:
        # print('init', classname)
        convblock_init_normal(m)
    elif classname.find("Conv2d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


if __name__ == '__main__':
    import sys, os

    sys.path.append(os.path.join(os.environ['HOME'], 'GitRepos/DLToolbox'))
    from toolbox.core.utils import arch_stat

    net = yolov(backbone=resnet.resnet50,
                pretrained=False,
                inp_dim=(3, 512, 512),
                num_classes=2)
    # net = yolov(backbone=densenet.densenet121,
    #             inp_dim=(3, 512, 512),
    #             num_classes=2)
    # print(net)
    net.apply(weights_init_normal)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(net(torch.randn([1, 3, 512, 512])).shape)
    params, flops = arch_stat(net, 32, device, None)
    print(f'Params {params}, FLOPs {flops}')
