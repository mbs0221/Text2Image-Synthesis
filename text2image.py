import argparse
import os
import numpy as np
import time

import torch.nn.functional as F
import torch
from torch import nn
from torch.nn import init
from torch.utils.data import DataLoader

from torchtext import data
from torchtext.vocab import Vectors, GloVe

import torchvision.models as models
from torchvision.utils import save_image

from modal import MBSGenerator, ZQHDiscriminator, MBSDiscriminator, Attn, TextEncoder, TVLoss
from utils import datasets


class CALayer(nn.Module):
    """
    Conditioning Augmentation (CA)
    """

    def __init__(self, in_features, out_features):
        super(CALayer, self).__init__()
        self.mean = nn.Linear(in_features, out_features)
        self.var = nn.Linear(in_features, out_features)

    def forward(self, input):
        x = input.squeeze()
        z_mean = self.mean(x)
        z_var = self.var(x)
        y = torch.normal(z_mean, z_var)
        return y.unsqueeze(2).unsqueeze(3)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("ConvTranspose2d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def write_log(path, batch, text, vocab):
    # get captions
    captions = []
    for item in text:
        caption = []
        for word in item:
            if word == 1:
                break
            caption.append(vocab.itos[word])
        captions.append(" ".join(caption) + '\n')
    # write file
    with open(path, 'a+') as f:
        f.write(f'batch-done: {batch}\n')
        f.writelines(captions)


has_cuda = True if torch.cuda.is_available() else False

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='../datasets/coco-2014/', type=str, help='coco-dataset folder')
    parser.add_argument('--batch_size', default=32, type=int, help='the training epochs')
    parser.add_argument('--n_epochs', default=100, type=int, help='the training epochs')
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument('--l1_coeff', type=float, default=50, help='the l1 coefficient loss')
    parser.add_argument('--latent_dim', default=12, type=int, help='the dimensional of latent space')
    parser.add_argument('--text_dim', default=96, type=int, help='the dimensional of text-embedding')
    parser.add_argument('--text_reduced_dim', default=48, type=int, help='the dimensional of text-embedding')
    parser.add_argument('--cond_dim', default=48, type=int, help='the dimensional of conditioning')
    parser.add_argument('--image_size', default=64, type=int, help='the image size')
    parser.add_argument('--n_channels', default=3, type=int, help='the number of image channels')
    parser.add_argument('--kqv_dim', default=50, type=int, help='for attention module')
    parser.add_argument('--ngf', default=24, type=int, help='Size of feature maps in generator')
    parser.add_argument('--ndf', default=24, type=int, help='Size of feature maps in discriminator')
    parser.add_argument('--num_workers', default=1, type=int, help='The number of running threads')
    parser.add_argument('--sample_interval', type=int, default=300, help="interval between image sampling")
    parser.add_argument('--use_cuda', type=bool, default=True, help='use cuda for training')
    parser.add_argument('--log_path', type=str, default='./log.txt', help='generating log')
    args = parser.parse_args()

    root = args.root
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    latent_dim = args.latent_dim
    text_dim = args.text_dim
    text_reduced_dim = args.text_reduced_dim
    cond_dim = args.cond_dim
    kqv_dim = args.kqv_dim
    image_size = args.image_size
    n_channels = args.n_channels
    ngf = args.ngf
    ndf = args.ndf

    # load glove
    glove = GloVe(name='6B', dim=100)

    # COCO-dataset
    print('load coco-caption')
    field = data.Field(sequential=True, tokenize=data.get_tokenizer(tokenizer='basic_english'), lower=True)
    TEXT = data.NestedField(field, use_vocab=True)
    train = datasets.coco_caption(root, 'train', TEXT)
    # val = datasets.coco_caption(root, 'val', TEXT)

    print('build vocabulary from dataset')
    TEXT.build_vocab(train, vectors=glove)
    TEXT.vocab.vectors.unk_init = init.xavier_uniform

    # build vocabulary from dataset
    embedding = TEXT.vocab.vectors
    vocab_size, embedding_dim = embedding.shape
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    # define modules
    print('construct text-encoder, generator, discriminator')
    text_encoder = TextEncoder(vocab_size, embedding_dim, text_dim, kqv_dim, embedding, padding_idx=PAD_IDX)
    generator = MBSGenerator(latent_dim, text_dim, ngf, n_channels)
    discriminator = MBSDiscriminator(ndf, text_dim, n_channels)
    # discriminator = QHDiscriminator(text_dim, text_reduced_dim)

    # load pre-trained modal
    print('load pre-trained modal')
    pairs = [
        (text_encoder, 'text_encoder.pkl'),
        (generator, 'generator.pkl'),
        (discriminator, 'discriminator.pkl')
    ]
    for (model, path) in pairs:
        if os.path.exists(path):
            model.load_state_dict(torch.load(path))
        else:
            model.apply(weights_init_normal)

    # create image folder
    if not os.path.exists('./images'):
        os.mkdir('./images')

    # loss function
    adversarial_loss = torch.nn.BCELoss()
    l1_loss = torch.nn.L1Loss()
    tv_loss = TVLoss()

    # CUDA
    cuda_enable = has_cuda and args.use_cuda
    if cuda_enable:
        text_encoder.cuda()
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    # ----------
    #  Training
    # ----------
    device = torch.device('cuda:2' if cuda_enable else 'cpu')
    train_iterator = data.BucketIterator.splits(
        (train),
        batch_size=batch_size,
        device=device,
        sort_key=lambda x: len(x.text),
        repeat=False
    )

    print(f'start training at: {time.asctime()}')
    for epoch in range(n_epochs):
        for i, (text, image) in enumerate(train_iterator):

            # Adversarial ground truths
            batch_size = len(image)
            valid_labels = torch.ones(size=torch.Size([batch_size]), dtype=torch.float32)
            fake_labels = torch.zeros(size=torch.Size([batch_size]), dtype=torch.float32)

            # real-images
            real_images = torch.stack(image)

            # miss-match images
            ids = np.arange(0, batch_size, 1)
            _ids = np.random.permutation(ids)
            wrong_images = real_images[_ids]
            match_labels = torch.from_numpy(ids == _ids).type(torch.float32)

            # -----------------
            #  Text-Embedding
            # -----------------

            idx = np.random.randint(5)
            texts = text[:, idx, :]
            text_embedding = text_encoder(texts)
            text_embedding = text_embedding[:, -1].unsqueeze(2).unsqueeze(3)
            text_embedding = text_embedding.detach()

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # generate fake images with text-embedding
            gen_images = generator(text_embedding)

            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_images, text_embedding)
            g_loss = adversarial_loss(validity, valid_labels) \
                     + args.l1_coeff * l1_loss(gen_images, real_images) + tv_loss(gen_images)
            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # loss for exact-match (real-image, real-text)
            validity_real = discriminator(real_images, text_embedding)
            d_real_loss = adversarial_loss(validity_real, valid_labels)

            # loss for miss-match (fake-image, real-text)
            validity_fake = discriminator(gen_images.detach(), text_embedding)
            d_fake_loss = adversarial_loss(validity_fake, fake_labels)

            # loss for miss-match (wrong-image, real-text)
            validity_match = discriminator(wrong_images, text_embedding)
            d_match_loss = adversarial_loss(validity_match, match_labels)

            d_loss = (d_real_loss + d_fake_loss + d_match_loss) / 3
            d_loss.backward()

            optimizer_D.step()

            print(
                "[%s][Epoch %3d/%d] [Batch %4d/%4d] [D loss: %f] [G loss: %f]"
                % (time.asctime(), epoch, args.n_epochs, i, len(train_iterator), d_loss.item(), g_loss.item())
            )

            batches_done = epoch * len(train_iterator) + i
            if batches_done % args.sample_interval == 0:
                save_image(gen_images.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
                write_log(args.log_path, batches_done, texts, TEXT.vocab)

        for (model, path) in pairs:
            torch.save(model.state_dict(), path)
