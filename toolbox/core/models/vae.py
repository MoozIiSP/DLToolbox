"""Code from pytorch/examples"""
from __future__ import print_function

import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F


class MixtureNet(nn.Module):
    def __init__(self):
        super(MixtureNet, self).__init__()

        self.encoder = Encoder()
        self.VAE = VAE(size=16*1024)
        self.decoder = Decoder()

    def forward(self, x):
        out = self.encoder(x)

        x = out
        recon_x, mu, logvar = self.VAE(out)

        out = self.decoder(recon_x.reshape_as(out))

        return out, recon_x, x, mu, logvar


class VAE(nn.Module):
    def __init__(self, size=784):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(size, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.flatten(1, -1))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))

        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(16, 3, 1, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))

        return torch.sigmoid(x)
