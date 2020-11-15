import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from argparse import ArgumentParser
from pytorch_lightning import Trainer

import pytorch_lightning as pl
import resnet


class LitModelContainer(pl.LightningModule):
    def __init__(self, hparams):
        super(LitModelContainer, self).__init__()
        self.hparams = hparams
        self.net = resnet.__dict__[self.hparams.net](
            num_classes=self.hparams.num_classes)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)

        tb_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tb_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def optimizer_step(
            self,
            epoch,
            batch_idx,
            optimizer,
            optimizer_idx,
            second_order_closure
    ) -> None:
        # warm up lr
        # global_step is not epoch just every iteration
        if self.hparams.warmup and self.trainer.global_step < 1000:
            lr_scale = min(1., float(self.trainer.global_step + 1) / 1000.)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.learning_rate

        optimizer.step()
        optimizer.zero_grad()

    def train_dataloader(self):
        return DataLoader(datasets.CIFAR100(self.hparams.dataset,
                                            train=True,
                                            download=False,
                                            transform=transforms.Compose([
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomVerticalFlip(),
                                                transforms.ToTensor()
                                            ])),
                          batch_size=self.hparams.batch_size, shuffle=True, pin_memory=True,
                          num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(datasets.CIFAR100(self.hparams.dataset,
                                            train=False,
                                            download=False,
                                            transform=transforms.Compose([
                                                transforms.ToTensor()
                                            ])),
                          batch_size=self.hparams.batch_size, shuffle=False, pin_memory=True,
                          num_workers=self.hparams.num_workers)

    # def test_dataloader(self):
    #     raise NotImplementedError

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=0.9,
                                    nesterov=self.hparams.nesterov,
                                    weight_decay=self.hparams.weight_decay)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams.milestones, gamma=0.1)
        return [optimizer], [scheduler]


if __name__ == '__main__':
    # optimize gpu
    torch.backends.cudnn.deterministic = True  # Enable reproducible
    torch.backends.cudnn.benchmark = True

    # Hyper parameters setting
    parser = ArgumentParser()
    parser.add_argument('dataset', help='path to dataset')
    # Model Hyper parameters
    model_group = parser.add_argument_group('Model')
    model_group.add_argument('--net', type=str, default='UNet',
                             help='resnet18|resnet34|resnet50|resnet101\n'
                                  'resnet152|resnext50_32x4d|resnext101_32x8d\n'
                                  'wide_resnet50_2|wide_resnet101_2')
    model_group.add_argument('--num-classes', type=int, default=10)
    # model_group.add_argument('--input-shape',
    #                          type=lambda l: eval(l), default=(512, 512))
    # model_group.add_argument('--init-features', type=int, default=16)
    # model_group.add_argument('--alpha', type=float, default=1,
    #                          help='Control features of ResidualBlock: ' +
    #                               'standard residual (1), inverted residual (<1), bottleneck (>1)')
    # model_group.add_argument('--beta', type=int, default=1,
    #                          help='Control depth of ResidualBlock in module.')
    # Optimizer Hyper parameters
    optim_group = parser.add_argument_group('Optimizer')
    optim_group.add_argument('--weight-decay', type=float, default=0)
    optim_group.add_argument('--nesterov', action='store_true')
    optim_group.add_argument('--warmup', action='store_true')
    # Learning rate parameters
    lr_group = parser.add_argument_group('Learning Rate')
    lr_group.add_argument('--milestones',
                          type=lambda l: eval(l), default=list(range(1, 100, 5)))
    # Data Aug
    aug_group = parser.add_argument_group('Data Augmentation')
    # aug_group.add_argument('--normalize',
    #                        type=lambda l: eval(l), default=[])
    # aug_group.add_argument('--adjust-brightness',
    #                        type=lambda l: eval(l), default=[1, 1])
    # aug_group.add_argument('--flip', action='store_true')
    # aug_group.add_argument('--crop-batch',
    #                        type=int, default=8)
    aug_group.add_argument('--num-workers', type=int, default=1)
    # Trainer Hyper parameters
    trainer_group = parser.add_argument_group('Trainer')
    trainer_group.add_argument('--batch-size', type=int, default=1)
    trainer_group.add_argument('--learning-rate', type=float, default=1e-1)
    trainer_group.add_argument('--max-epochs', type=int, default=1000)
    trainer_group.add_argument('--amp-level', type=str, default='O1')
    trainer_group.add_argument('--fp16', action='store_true')
    # Pytorch lightning parameters
    # pl_param_group = parser.add_argument_group('PyTorch Lightning')
    parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()

    model = LitModelContainer(hparams)
    trainer = Trainer(
        max_epochs=hparams.max_epochs,
        gpus=1,  # CUDA support
        amp_level=hparams.amp_level,  # Nvidia Apex
        precision=16 if hparams.fp16 else 32,  # FP16
    )
    trainer.fit(model)
