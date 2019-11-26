import argparse
import cv2
import os
import numpy as np
import torch.nn.functional as F
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchtext import data
from torchvision import datasets
from torchtext.vocab import Vectors, GloVe
import torchvision.models as models
from torchvision import transforms
from torchvision.datasets import Flickr8k, CocoCaptions

from generator import Generator
from discriminator import Discriminator
from text_encoder import TextEncoder


class CALayer(nn.Module):
    """
    Conditioning Augmentation (CA)
    """

    def __init__(self, input_dim, latent_dim):
        super(CALayer, self).__init__()
        self.mean = nn.Linear(input_dim, latent_dim)
        self.var = nn.Linear(input_dim, latent_dim)

    def forward(self, embedding):
        z_mean = self.mean(embedding)
        z_var = self.var(embedding)
        return torch.normal(z_mean, z_var)


def coco_captions(path, type, image_size=64):
    dataset = CocoCaptions(
        root=f"{path}/images/{type}2014",
        annFile=f"{path}/annotations/captions_{type}2014.json",
        transform=transforms.Compose([
            transforms.Resize([image_size, image_size]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    )
    return dataset


def get_annotations(dataset):
    annotations = []
    for i, (_, target) in enumerate(dataset):
        annotations.extend(target)
    return annotations


def sample_image(n_row, batches_done):
    """
    Saves a grid of generated digits ranging from 0 to n_classes
    :param n_row:
    :param batches_done:
    :return:
    """
    pass


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument('--latent_dim', type=int, default=32, help='dimensionality of the latent space')
parser.add_argument('--image_size', type=int, default=64, help='size of each image dimension')
parser.add_argument("--n_channels", type=int, default=1, help="number of image channels")
parser.add_argument('--text_dim', type=int, default=48, help='Size of the embedding for the captions')
parser.add_argument('--text_reduced_dim', type=int, default=32, help='Reduced dimension of the caption encoding')
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument('--max_length', type=int, default=50, help='the max length of input sequence')
args = parser.parse_args()
print(args)

batch_size = args.batch_size
latent_dim = args.latent_dim
text_dim = args.text_dim
text_reduced_dim = args.text_reduced_dim
image_size = args.image_size
n_channels = args.n_channels

# Initialize text-encoder, ca-layer, generator, discriminator
text_encoder = TextEncoder(32, 128, 50)
generator = nn.Sequential(
    CALayer(50, 36),
    Generator(latent_dim=latent_dim, text_dim=text_dim)
)
discriminator = Discriminator(text_embed_dim=text_dim, text_reduced_dim=text_reduced_dim)

# Data-sets
root = "../datasets/coco-2014"
train = coco_captions(root, 'train')
val = coco_captions(root, 'val')

print(f'train:{len(train)}, val:{len(val)}')

img, target = train[0]  # load 4th sample
print(f"Image Size:{img.size}")
print(target)

# Build vocabulary from dataset
# train_annotations = get_annotations(train)
# val_annotations = get_annotations(val)
# annotations = np.hstack([train_annotations, val_annotations])

vectors = Vectors('../glove/glove.6B/glove.6B.300d.txt')
TEXT = data.Field(sequential=True, lower=True, use_vocab=True, batch_first=True, eos_token='.')
TEXT.build_vocab(train, val, vectors=vectors)
TEXT.build_vocab(train, val, vectors=GloVe(name='6B', dim=300))

# Configure data-loader
data_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=False, num_workers=4)

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

# ----------
#  Training
# ----------
for epoch in range(args.n_epochs):
    for i, (images, targets) in enumerate(data_loader):
        # Adversarial ground truths
        valid = Variable(Tensor(images.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(images.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_images = Variable(imgs.type(Tensor))

        # text-embedding
        text_embedding = text_encoder(targets)

        # -----------------
        #  Train Generator
        # -----------------

        # generate fake images with text-embedding
        gen_images = generator(text_embedding)
        validity = discriminator(gen_images)
        g_loss = adversarial_loss(validity, valid)
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # TODO: Image embedding

        # Loss for real images
        validity_real = discriminator(real_images, text_embedding)
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        validity_fake = discriminator(gen_images.detach(), text_embedding)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # TODO: Loss for interpolated samples

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=10, batches_done=batches_done)
