import argparse
import cv2
import os
import numpy as np

from PIL import Image

import torch.nn.functional as F
import torch
from torch import nn
from torch.nn import init
from torch.utils.data import DataLoader

from torchtext import data
from torchtext.vocab import Vectors, GloVe
from torchtext.data import Example

import torchvision.models as models
from torchvision import datasets
from torchvision import transforms
from torchvision.datasets import Flickr8k
from torchvision.datasets.vision import StandardTransform
from modal import Generator, Discriminator, Attn, TextEncoder


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


class CocoCaptions(data.Dataset):
    urls = [
        'http://images.cocodataset.org/zips/train2014.zip',
        'http://images.cocodataset.org/zips/val2014.zip',
        'http://images.cocodataset.org/zips/test2014.zip',
        'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
        'http://images.cocodataset.org/annotations/image_info_test2014.zip'
    ]
    name = 'coco'
    dirname = 'coco-2014'

    def __init__(self, path, ann_path, image_field, text_field, transforms=None, transform=None, target_transform=None,
                 **kwargs):
        self.path = path
        # for vision data
        has_transforms = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError("Only transforms or transform/target_transform can "
                             "be passed as argument")

        # for backwards-compatibility
        self.transform = transform
        self.target_transform = target_transform

        if has_separate_transform:
            transforms = StandardTransform(transform, target_transform)
        self.transforms = transforms

        # for text data
        from pycocotools.coco import COCO
        self.coco = COCO(ann_path)
        self.ids = list(sorted(self.coco.imgs.keys()))
        # collect examples
        coco = self.coco
        fields = [('image', image_field), ('text', text_field)]
        examples = []
        for img_id in self.ids:
            # text
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            target = [ann['caption'] for ann in anns]
            # image
            path = coco.loadImgs(img_id)[0]['file_name']
            img = Image.open(os.path.join(self.path, path)).convert('RGB')
            if self.transforms is not None:
                img, target = self.transforms(img, target)

            examples.append(Example.fromlist([img, target], fields))
        super(CocoCaptions, self).__init__(examples, fields)


def coco_captions(path, type, nested_field, image_field):
    dataset = CocoCaptions(
        path=f"{path}/resized/{type}2014",
        ann_path=f"{path}/annotations/captions_{type}2014.json",
        text_field=nested_field,
        image_field=image_field,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    )
    return dataset


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("ConvTranspose2d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


cuda = True if torch.cuda.is_available() else False

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int, help='the training epochs')
    parser.add_argument('--n_epochs', default=100, type=int, help='the training epochs')
    parser.add_argument("--lr", type=float, default=0.002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument('--latent_dim', default=48, type=int, help='the dimensional of latent space')
    parser.add_argument('--text_dim', default=48, type=int, help='the dimensional of text-embedding')
    parser.add_argument('--cond_dim', default=48, type=int, help='the dimensional of conditioning')
    parser.add_argument('--image_size', default=64, type=int, help='the image size')
    parser.add_argument('--n_channels', default=3, type=int, help='the number of image channels')
    parser.add_argument('--kqv_dim', default=50, type=int, help='for attention module')
    parser.add_argument('--ngf', default=24, type=int, help='Size of feature maps in generator')
    parser.add_argument('--ndf', default=24, type=int, help='Size of feature maps in discriminator')
    parser.add_argument('--num_workers', default=1, type=int, help='The number of running threads')
    parser.add_argument("--sample_interval", type=int, default=60, help="interval between image sampling")
    args = parser.parse_args()

    batch_size = args.batch_size
    n_epochs = args.n_epochs
    latent_dim = args.latent_dim
    text_dim = args.text_dim
    cond_dim = args.cond_dim
    kqv_dim = args.kqv_dim
    image_size = args.image_size
    n_channels = args.n_channels
    ngf = args.ngf
    ndf = args.ndf

    # load glove
    glove = GloVe(name='6B', dim=100)

    # COCO-dataset
    root = "../datasets/coco-2014/"
    print('load coco-caption')
    IMAGE = data.RawField(is_target=True)
    field = data.Field(sequential=True, lower=True)
    TEXT = data.NestedField(field, use_vocab=True)
    train = coco_captions(root, 'train', TEXT, IMAGE)
    val = coco_captions(root, 'val', TEXT, IMAGE)

    print('build vocabulary from dataset')
    TEXT.build_vocab(train, val, vectors=glove)
    TEXT.vocab.vectors.unk_init = init.xavier_uniform

    # build vocabulary from dataset
    embedding = TEXT.vocab.vectors
    vocab_size, embedding_dim = embedding.shape
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    # define modules
    print('construct text-encoder, ca-layer, generator')
    text_encoder = TextEncoder(vocab_size, embedding_dim, text_dim, kqv_dim, embedding, padding_idx=PAD_IDX)
    generator = nn.Sequential(
        CALayer(text_dim, cond_dim),
        Generator(latent_dim, cond_dim, ngf, n_channels)
    )
    discriminator = Discriminator(ndf, text_dim, n_channels)

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

    # loss function
    adversarial_loss = torch.nn.BCELoss()

    # cuda
    if cuda:
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_iterator, test_iterator = data.BucketIterator.splits(
        (train, val),
        batch_size=batch_size,
        device=device
    )

    print('start training...')
    for epoch in range(n_epochs):
        for i, (text, image) in enumerate(train_iterator):

            # Adversarial ground truths
            batch_size = len(image)
            valid = torch.ones(size=torch.Size([batch_size, 1]), dtype=torch.float32)
            fake = torch.zeros(size=torch.Size([batch_size, 1]), dtype=torch.float32)

            # real-images
            real_images = torch.stack(image)

            # text-embedding
            idx = np.random.randint(5)
            text_embedding = text_encoder(text[:, idx, :])
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
            g_loss = adversarial_loss(validity, valid)
            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # loss for (real-image, real-text), (fake-image, real-text), (wrong-image, real-text)
            validity_real = discriminator(real_images, text_embedding)
            d_real_loss = adversarial_loss(validity_real, valid)

            validity_fake = discriminator(gen_images.detach(), text_embedding)
            d_fake_loss = adversarial_loss(validity_fake, fake)

            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, args.n_epochs, i, len(train_iterator), d_loss.item(), g_loss.item())
            )

            batches_done = epoch * len(train_iterator) + i
            if batches_done % args.sample_interval == 0:
                save_image(gen_images.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

        for (model, path) in pairs:
            torch.save(model.state_dict(), path)
