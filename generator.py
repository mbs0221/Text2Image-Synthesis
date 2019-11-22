import torch

import torch.nn.functional as F
import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, z_dim, text_embedding_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.text_embedding_dim = text_embedding_dim
        self.generator = nn.Sequential(
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

    def forward(self, input, z):
        embedding = torch.cat([input, z], 1)
        return self.generator(embedding)
