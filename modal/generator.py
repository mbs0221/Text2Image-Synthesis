import torch

import torch.nn.functional as F
import torch.nn as nn


class MBSGenerator(nn.Module):

    def __init__(self, latent_dim, text_dim, ngf=128, n_channels=3):
        """

        :param latent_dim: The dimensional of latent space
        :param text_dim: the dimensional of text-embedding
        :param ngf: Size of feature maps in generator
        :param n_channels: the channel of image
        """
        super(MBSGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.text_dim = text_dim
        self.generator = nn.Sequential(
            # [-, nz, 1, 1]
            nn.ConvTranspose2d(latent_dim + text_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, True),
            # [-, 1024, 4, 4]
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, True),
            # [-, 512, 8, 8]
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, True),
            # [-, 256, 16, 16]
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, True),
            # [-, 128, 32, 32]
            nn.ConvTranspose2d(ngf, n_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_channels),
            nn.Tanh()
            # [-, nc, 64, 64]
        )

    def forward(self, embedding):
        return self.generator(embedding)
