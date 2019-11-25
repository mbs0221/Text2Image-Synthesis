import torch

import torch.nn.functional as F
import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, nz):
        super(Generator, self).__init__()
        self.nz = nz
        self.generator = nn.Sequential(
            # input is Z, going into a convolution
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, input):
        # Ng=128, Nz=100
        embedding = torch.cat([input, z], 1)
        image = embedding.reshape((-1, 16, 16, 3))
        return self.generator(image)
