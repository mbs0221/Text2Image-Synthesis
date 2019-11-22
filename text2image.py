import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
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
parser.add_argument('--img_size', type=int, default=64, help='size of each imahe dimension')
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
opt = parser.parse_args()
print(opt)

# Initialize generator and discriminator
text_encoder = TextEncoder(32, 32, 50)
generator = Generator(opt.latent_dim, opt.text_embedding_dim)
discriminator = Discriminator()

# load vectors from file
# vectors = Vectors(name='../glove/glove.6B.200d.txt')
TEXT = data.Field(sequential=True)
TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=300))
# TEXT.vocab

# Configure data-loader
data_loader = torch.utils.data.DataLoader(
    datasets.Flickr8k(
        root="./data/Flickr8k/images/",
        ann_file="./data/Flickr8k/annotations/annotations.txt",
        transform=transforms.Compose([
            transforms.Resize(opt.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    ),
    batch_size=opt.batch_size,
    shuffle=True
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# ----------
#  Training
# ----------
for epoch in range(opt.n_epochs):
    for i, (img, anno) in enumerate(data_loader):
        pass
