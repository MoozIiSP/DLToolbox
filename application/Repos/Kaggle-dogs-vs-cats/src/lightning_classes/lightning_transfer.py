from torch.optim import optimizer
from src.utils.dataset import get_test_dataset, get_trainval_datasets
from omegaconf.dictconfig import DictConfig
import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict
import pytorch_lightning as pl

from src.utils.utils import collate_fn, freeze, load_obj


class LitTransferLearning(pl.LightningModule):

    def __init__(self, cfg: DictConfig) -> None:
        super(LitTransferLearning, self).__init__()
        self.hparams = cfg
        self.model = self._build_model(cfg)

    def _build_model(self) -> nn.Sequential:
        model_fn = getattr(models, self.hparams.models.backbone.class_name)
        backbone = model_fn(pretrained=self.hparams.models.backbone.params.pretrained)

        _layer = list(backbone.children())[:self.hparams.models.depth]
        backbone = nn.Sequential(**_layer)
        freeze(_layer, train_bn=self.hparams.models.bn_trainable)

        # get last layer output shape
        with torch.no_grad():
            last = torch.prod(self.backbone(torch.randn(1, 3, 224, 224)).shape[1:]).detach().item()
        head = nn.Sequential(
            nn.Linear(last, 2)
        )

        return nn.Sequential(OrderedDict([
            (f'{self.hparams.backbone.class_name}_backbone', backbone),
            ('detection_head', head)
        ]))

    def forward(self, x) -> torch.Tensor:
        return self.model(x)

    def _prepare_data(self) -> None:
        trainval = get_trainval_datasets(self.hparams)
        self.trainset = trainval['train']
        self.validset = trainval['valid']
        self.testset = get_test_dataset(self.hparams)

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(self.trainset,
                                                   batch_size=self.hparams.data.batch_size,
                                                   num_workers=self.hparams.num_workers,
                                                   shuffle=True,
                                                   collate_fn=collate_fn)
        return train_loader

    def val_dataloader(self):
        valid_loader = torch.utils.data.DataLoader(self.validset,
                                                   batch_size=self.hparams.data.batch_size,
                                                   num_workers=self.hparams.num_workers,
                                                   shuffle=True,
                                                   collate_fn=collate_fn)
        return valid_loader

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(self.testset,
                                                  batch_size=self.hparams.data.batch_size,
                                                  num_workers=self.hparams.num_workers,
                                                  shuffle=True,
                                                  collate_fn=collate_fn)
        return test_loader

    def configure_optimizers(self):
        optimizer = load_obj(self.hparams.optimizer.class_name)(self.model.parameters(),
                                                                **self.hparams.optimizer.params)
        scheduler = load_obj(self.hparams.scheduler.class_name)(optimizer, **self.hparams.scheduler.params)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y, image_ids = batch
        y_hat = self.model(x)
        losses = self.hparams.loss.class_name(y_hat, y)

        self.log('train_loss', losses, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return losses

    def validation_step(self, batch, batch_idx):
        x, y, image_ids = batch
        y_hat = self.model(y)
        losses = self.hparams.loss.class_name(y_hat, y)

        self.log('valid_loss', losses, on_step=True, on_epoch=True, prog_bar=True, logger=True)


if __name__ == '__main__':
    cfg = DictConfig()

