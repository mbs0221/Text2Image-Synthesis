import argparse
import cv2
import os

import torch
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from generator import Generator
from discriminator import Discriminator
from text_encoder import TextEncoder

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument('--latent_dim', type=int, default=32, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=64, help='size of each image dimension')
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument('--text_embedding_dim', type=int, default=48, help='Size of the embedding for the captions')
parser.add_argument('--text_reduced_dim', type=int, default=32, help='Reduced dimension of the caption encoding')
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()
print(opt)


def coco_captions(path, type):
    dataset = datasets.CocoCaptions(
        root=f"{path}/images/{type}2014",
        annFile=f"{path}/annotations/captions_{type}2014.json",
        transform=transforms.Compose([
            transforms.Resize([opt.img_size, opt.img_size]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    )
    return dataset


def sample_image(n_row, batches_done):
    """
    Saves a grid of generated digits ranging from 0 to n_classes
    :param n_row:
    :param batches_done:
    :return:
    """
    pass


# Initialize generator and discriminator
text_encoder = TextEncoder(32, 32, 50)
generator = Generator(opt.latent_dim, opt.text_embedding_dim)
discriminator = Discriminator(batch_size=opt.batch_size,
                              img_size=opt.img_size,
                              text_embed_dim=opt.text_embedding_dim,
                              text_reduced_dim=opt.text_reduced_dim)

# load vectors from file
# vectors = Vectors(name='../glove/glove.6B.200d.txt')
# TEXT = data.Field(sequential=True)
# TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=300))
# TEXT.vocab


# Data-sets
root = "../datasets/coco-2014"
train = coco_captions(root, 'train')
val = coco_captions(root, 'val')

print(f'train:{len(train)}, val:{len(val)}')

img, target = train[0]  # load 4th sample
print(f"Image Size:{img.size}")
print(target)

# Configure data-loader
data_loader = torch.utils.data.DataLoader(dataset=train, batch_size=opt.batch_size, shuffle=False, num_workers=4)

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# ----------
#  Training
# ----------
for epoch in range(opt.n_epochs):
    for i, (images, targets) in enumerate(data_loader):
        # Adversarial ground truths
        valid = Variable(Tensor(images.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(images.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_images = Variable(imgs.type(Tensor))
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
