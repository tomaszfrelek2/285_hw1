import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, image_size, latent_dim, base_channels):
        super().__init__()
        self.channels = [base_channels * 2**i for i in range(int(math.log2(image_size // 4)))]
        layers, c_in = [], 1
        for c in self.channels:
            layers += [nn.Conv2d(c_in, c, 4, 2, 1, bias=False), nn.BatchNorm2d(c), nn.LeakyReLU(0.2, inplace=True)]
            c_in = c
        self.conv = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(c_in * 16, latent_dim)
        self.fc_logvar = nn.Linear(c_in * 16, latent_dim)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim, enc_channels):
        super().__init__()
        chans = enc_channels[::-1]
        self.top = chans[0]
        self.fc = nn.Linear(latent_dim, self.top * 16)
        layers = []
        for c_in, c_out in zip(chans[:-1], chans[1:]):
            layers += [nn.ConvTranspose2d(c_in, c_out, 4, 2, 1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
        layers.append(nn.ConvTranspose2d(chans[-1], 1, 4, 2, 1))
        self.deconv = nn.Sequential(*layers)

    def forward(self, z):
        h = F.relu(self.fc(z)).view(-1, self.top, 4, 4)
        return torch.sigmoid(self.deconv(h))


class VAE(nn.Module):
    def __init__(self, image_size=64, latent_dim=128, base_channels=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(image_size, latent_dim, base_channels)
        self.decoder = Decoder(latent_dim, self.encoder.channels)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)  # reparameterization
        return self.decoder(z), mu, logvar

    @torch.no_grad()
    def sample(self, n, device):
        return self.decoder(torch.randn(n, self.latent_dim, device=device))

    @torch.no_grad()
    def reconstruct(self, x):
        return self.decoder(self.encoder(x)[0])


def vae_loss(x_hat, x, mu, logvar, beta=1.0):
    """Negative ELBO in nats per image: BCE reconstruction + beta * KL."""
    rec = F.binary_cross_entropy(x_hat, x, reduction="sum") / x.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return rec + beta * kl, rec, kl
