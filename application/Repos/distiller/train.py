import torch
from torch import (
    nn, optim
)
from torchvision import (
    datasets, models, transforms
)
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer
from torchvision.models.mobilenet import mobilenet_v2


class Distiller(LightningModule):
    def __init__(self) -> None:
        super(Distiller, self).__init__()
        self.T = models.resnet50(pretrained=True)
        self.T.eval()
        self.S = models.mobilenet_v2()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_ind):
        x, _ = batch
        with torch.no_grad():
            y_T = self.T(x)
        y_S = self.S(x)

        nb = y_T.size(0)
        loss = nn.functional.l1_loss(y_T, y_S)
        tb_logs = {
            'train_loss': loss,
        }

        return {'loss': loss, 'log': tb_logs}

    def validation_step(self, batch, batch_ind):
        x, _ = batch
        with torch.no_grad():
            y_T = self.T(x)
        y_S = self.S(x)

        loss = nn.functional.l1_loss(y_T, y_S)

        nb = y_T.size(0)

        return {
            'val_loss': loss,
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        
        tb_logs = {
            'val_loss': avg_loss,
        }

        return {
            'avg_val_loss': avg_loss,
            'log': tb_logs,
        }

    def train_dataloader(self):
        return DataLoader(datasets.FakeData(size=1000, image_size=(3, 224, 224), 
                                            transform=transforms.Compose([transforms.ToTensor()])), 
                                            batch_size=16, num_workers=4)
    
    def val_dataloader(self):
        return DataLoader(datasets.FakeData(size=1000, image_size=(3, 224, 224), 
                                            transform=transforms.Compose([transforms.ToTensor()])), 
                                            batch_size=16, num_workers=4)
    
    def configure_optimizers(self):
        return optim.SGD(self.S.parameters(), lr=1e-2)


if __name__ == "__main__":
    model = Distiller()
    trainer = Trainer(
        max_epochs=100,
        #gpus=1,
    )
    trainer.fit(model)