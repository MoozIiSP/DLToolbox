import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from src.models import get_unet_model
from src.utils.dataset import get_training_datasets
from src.utils.loss import SoftDiceLoss
from src.utils.utils import collate_fn, load_obj
from torch.utils.data import Dataset


class LitBCDI(pl.LightningModule):

    def __init__(self, hparams: DictConfig = None):
        super(LitBCDI, self).__init__()
        self.hparams = hparams
        self.model = get_unet_model(self.hparams)
        self.loss = SoftDiceLoss()

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def prepare_data(self):
        datasets = get_training_datasets(self.hparams)
        self.train_dataset = datasets['train']
        self.valid_dataset = datasets['valid']

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=self.hparams.data.batch_size,
                                                   num_workers=self.hparams.data.num_workers,
                                                   shuffle=True,
                                                   #collate_fn=collate_fn
                                                   )
        return train_loader

    def val_dataloader(self):
        valid_loader = torch.utils.data.DataLoader(self.valid_dataset,
                                                   batch_size=self.hparams.data.batch_size,
                                                   num_workers=self.hparams.data.num_workers,
                                                   shuffle=False,
                                                   #collate_fn=collate_fn
                                                   )

        return valid_loader

    def configure_optimizers(self):
        optimizer = load_obj(self.hparams.optimizer.class_name)(self.model.parameters(),
                                                                **self.hparams.optimizer.params)
        scheduler = load_obj(self.hparams.scheduler.class_name)(optimizer, **self.hparams.scheduler.params)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        images, targets, image_ids = batch
        outputs = self.model(images)
        losses = self.loss(outputs, targets, ignore_index=[])

        loss_dict = {'train_loss': losses }

        return {'loss': losses, 'log': loss_dict}

    def validation_step(self, batch, batch_idx):
        images, targets, image_ids = batch
        outputs = self.model(images)
        losses = self.loss(outputs, targets, ignore_index=[])

        return {'val_loss': losses}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}
