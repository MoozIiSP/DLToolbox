import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['LeNet5', 'LeNet5t', 'AlexNetAe', 'LeNet5VAE', 'LeNet5p',
           'LeNet5wAeA', 'LeNet5wAeB', 'LeNet5wAeC']


class LeNet5(nn.Module):
    def __init__(self, kwidth=32, num_classes=10):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, kwidth, 5, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(kwidth, kwidth*2, 5, 1),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(kwidth*2 * 5 * 5, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1, -1)
        x = self.classifier(x)
        return x


class LeNet5t(nn.Module):
    def __init__(self, kwidth=32, num_classes=10):
        super(LeNet5t, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, kwidth, 3, stride=1),
            nn.Conv2d(kwidth, kwidth, 3, stride=1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(kwidth, kwidth*2, 3, 1),
            nn.Conv2d(kwidth*2, kwidth*2, 3, 1),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(kwidth*2 * 5 * 5, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1, -1)
        x = self.classifier(x)
        return x


class AlexNetAe(nn.Module):
    def __init__(self, kwidth=32, num_classes=10, z_length=16):
        super(AlexNetAe, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 192, 3, 1, 1),
            nn.Conv2d(192, 192, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(192, 384, 3, 1, 1),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.Conv2d(384, 192, 3, 1, 1),
            nn.ConvTranspose2d(192, 192, 2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, 3, 1, 1),
            nn.Conv2d(192, 64, 3, 1, 1),
            nn.ConvTranspose2d(64, 64, 2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.Conv2d(64, 3, 3, 1, 1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return torch.sigmoid(x)


class LeNet5VAE(nn.Module):
    def __init__(self, kwidth=32, num_classes=10, z_length=16):
        super(LeNet5VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, kwidth*3, 3, 1, 1),
            nn.BatchNorm2d(kwidth*3),
            nn.ReLU(inplace=True),
            nn.Conv2d(kwidth*3, kwidth*3, 3, 1, 1),
            nn.BatchNorm2d(kwidth*3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(kwidth*3, kwidth*2, 3, 1, 1),
            nn.BatchNorm2d(kwidth*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(kwidth*2, kwidth, 3, 1, 1),
            nn.BatchNorm2d(kwidth),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.mlp1 = nn.Linear(kwidth*8*8, z_length*20)
        self.mlp21 = nn.Linear(z_length*20, z_length)
        self.mlp22 = nn.Linear(z_length*20, z_length)
        self.mlp3 = nn.Linear(z_length, z_length*10)
        self.mlp4 = nn.Linear(z_length*10, kwidth*8*8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(kwidth, kwidth, 2, 2),
            nn.Conv2d(kwidth, kwidth*2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(kwidth*2, kwidth*3, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(kwidth*3, kwidth*3, 2, 2),
            nn.Conv2d(kwidth*3, kwidth*2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(kwidth*2, 3, 3, 1, 1)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        x = self.encoder(x)
        target = x

        out = self.mlp1(x.flatten(1, -1))
        mu = self.mlp21(F.relu(out))
        logvar = self.mlp22(F.relu(out))
        z = self.reparameterize(mu, logvar)
        out = F.relu(self.mlp3(z))
        recon_target = torch.sigmoid(self.mlp4(out))
        print(torch.max(mu), torch.max(logvar), torch.max(recon_target))

        x = recon_target.reshape_as(x)
        x = self.decoder(x)
        return x, target, recon_target, mu, logvar


class LeNet5p(nn.Module):
    def __init__(self, kwidth=32, num_classes=10):
        super(LeNet5p, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, kwidth, 3, 1),
            nn.PReLU(),
            nn.Conv2d(kwidth, kwidth, 3, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(kwidth, kwidth*2, 3, 1),
            nn.PReLU(),
            nn.Conv2d(kwidth*2, kwidth*2, 3, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(kwidth*2 * 5 * 5, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1, -1)
        x = self.classifier(x)
        return x


class LeNet5wAeA(nn.Module):
    """利用LeNet5p中的特征提取层作为编码层，对输入图像进行编码，生成隐变量表征。
    然后，将其输入到与编码层相反结构的解码层中，对隐变量表征进行解码，以还原成原来的图像。
    与此同时，也需将隐变量表征输入到全连接层中，进行分类训练。该网络最后会返回多个Loss。"""
    def __init__(self, kwidth=64, num_classes=10):
        super(LeNet5wAeA, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, kwidth*3, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(kwidth*3, kwidth*3, 3, 1, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(kwidth*3, kwidth*2, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(kwidth*2, kwidth, 3, 1, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(kwidth, kwidth, 2, 2),
            nn.Conv2d(kwidth, kwidth*2, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(kwidth*2, kwidth*3, 3, 1, 1),
            nn.PReLU(),
            nn.ConvTranspose2d(kwidth*3, kwidth*3, 2, 2),
            nn.Conv2d(kwidth*3, kwidth*2, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(kwidth*2, 3, 3, 1, 1)
        )
        #self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Linear(kwidth * 8 * 8, 10)
        )

    def forward(self, x):
        x = self.encoder(x)
        match = self.decoder(x)
        pred = self.classifier(x.flatten(1, -1))
        return torch.sigmoid(match), pred


class LeNet5wAeB(nn.Module):
    def __init__(self, kwidth=64, num_classes=10):
        super(LeNet5wAeB, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, kwidth*3, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(kwidth*3, kwidth*3, 3, 1, 1),
            nn.BatchNorm2d(kwidth*3),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(kwidth*3, kwidth*2, 3, 1, 1),
            nn.BatchNorm2d(kwidth*2),
            nn.PReLU(),
            nn.Conv2d(kwidth*2, kwidth, 3, 1, 1),
            nn.BatchNorm2d(kwidth),
            nn.PReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(kwidth, kwidth, 2, 2),
            nn.Conv2d(kwidth, kwidth*2, 3, 1, 1),
            nn.BatchNorm2d(kwidth*2),
            nn.PReLU(),
            nn.Conv2d(kwidth*2, kwidth*3, 3, 1, 1),
            nn.BatchNorm2d(kwidth*3),
            nn.PReLU(),
            nn.ConvTranspose2d(kwidth*3, kwidth*3, 2, 2),
            nn.Conv2d(kwidth*3, kwidth*2, 3, 1, 1),
            nn.BatchNorm2d(kwidth*2),
            nn.PReLU(),
            nn.Conv2d(kwidth*2, 3, 3, 1, 1)
        )
        #self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Linear(kwidth * 8 * 8, 10)
        )

    def forward(self, x):
        x = self.encoder(x)
        match = self.decoder(x)
        pred = self.classifier(x.flatten(1, -1))
        return match, pred


class LeNet5wAeC(nn.Module):
    def __init__(self, kwidth=64, num_classes=10):
        super(LeNet5wAeC, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, kwidth*3, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(kwidth*3, kwidth*3, 3, 1, 1),
            nn.BatchNorm2d(kwidth*3),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(kwidth*3, kwidth*2, 3, 1, 1),
            nn.BatchNorm2d(kwidth*2),
            nn.PReLU(),
            nn.Conv2d(kwidth*2, kwidth, 3, 1, 1),
            nn.BatchNorm2d(kwidth),
            nn.PReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(kwidth, kwidth, 2, 2),
            nn.Conv2d(kwidth, kwidth*2, 3, 1, 1),
            nn.BatchNorm2d(kwidth*2),
            nn.PReLU(),
            nn.Conv2d(kwidth*2, kwidth*3, 3, 1, 1),
            nn.BatchNorm2d(kwidth*3),
            nn.PReLU(),
            nn.ConvTranspose2d(kwidth*3, kwidth*3, 2, 2),
            nn.Conv2d(kwidth*3, kwidth*2, 3, 1, 1),
            nn.BatchNorm2d(kwidth*2),
            nn.PReLU(),
            nn.Conv2d(kwidth*2, 3, 3, 1, 1)
        )
        #self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Linear(kwidth * 8 * 8, 10)
        )

    def forward(self, x):
        x = self.encoder(x)
        match = self.decoder(x)
        pred = self.classifier(x.flatten(1, -1))
        return match, pred
