import torch
import torch.nn as nn


class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, x, target):
        # FIXME
        iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
            pred_boxes=pred_boxes,
            pred_cls=pred_cls,
            target=targets,
            anchors=self.scaled_anchors,
            ignore_thres=self.ignore_thres,
        )

        # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
        loss_x = self.mse_loss(centre_x[obj_mask], tx[obj_mask])
        loss_y = self.mse_loss(centre_y[obj_mask], ty[obj_mask])
        loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
        loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
        loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
        loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
        loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
        loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
        total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

        # # Metrics
        # cls_acc = 100 * class_mask[obj_mask].mean()
        # conf_obj = pred_conf[obj_mask].mean()
        # conf_noobj = pred_conf[noobj_mask].mean()
        # conf50 = (pred_conf > 0.5).float()
        # iou50 = (iou_scores > 0.5).float()
        # iou75 = (iou_scores > 0.75).float()
        # detected_mask = conf50 * class_mask * tconf
        # precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
        # recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
        # recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)
        #
        # self.metrics = {
        #     "loss": total_loss.item(),
        #     "x": loss_x.item(),
        #     "y": loss_y.item(),
        #     "w": loss_w.item(),
        #     "h": loss_h.item(),
        #     "conf": loss_conf.item(),
        #     "cls": loss_cls.item(),
        #     "cls_acc": cls_acc.item(),
        #     "recall50": recall50.item(),
        #     "recall75": recall75.item(),
        #     "precision": precision.item(),
        #     "conf_obj": conf_obj.item(),
        #     "conf_noobj": conf_noobj.item(),
        #     "grid_size": grid_size,
        # }

        return total_loss