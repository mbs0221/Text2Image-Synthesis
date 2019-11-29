import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class ZXHDiscriminator(nn.Module):
    def __init__(self, text_embed_dim, text_reduced_dim):
        super(ZXHDiscriminator, self).__init__()

        self.text_embed_dim = text_embed_dim
        self.text_reduced_dim = text_reduced_dim

        # d_net用于提取图像特征 假设图像大小为 64*64*3 则提取的特征为（batch_size,256,8,8）
        self.d_net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True))
        # nn.Conv2d(256, 512, 4, 2, 1, bias=False),
        # nn.BatchNorm2d(512),
        # nn.LeakyReLU(0.2, inplace=True))

        # 定义一个线性变换来减少文本嵌入的维度
        # from text_embed_dim to text_reduced_dim
        self.text_reduced_op = nn.Linear(self.text_embed_dim, self.text_reduced_dim)

        # 真正的判别器网络
        self.cat_net = nn.Sequential(
            nn.Conv2d(256 + self.text_reduced_dim, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(512, 1),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, image, text):
        """

        Arguments
        ---------
        image : torch.FloatTensor
            image.size() = (batch_size, 64, 64, 3)

        text : torch.FloatTensor

            text.size() = (batch_size, text_embed_dim)

        --------
        Returns
        --------
        output : Probability for the image being real/fake
        logit : Final score of the discriminator

        """

        d_net_out = self.d_net(image)  # (batch_size, 256, 8, 8)
        text_reduced = self.text_reduced_op(text)  # (batch_size, text_reduced_dim)
        text_reduced = text_reduced.repeat(1, 1, 8, 8)  # (batch_size, text_reduced_dim, 8, 8)
        concat_out = torch.cat((d_net_out, text_reduced), 1)  # (batch_size, 8, 8, 256+text_reduced_dim)

        logit = self.cat_net(concat_out)

        output = F.sigmoid(logit)

        return output, logit


class MBSDiscriminator(nn.Module):

    def __init__(self, ndf, text_dim, n_channels):
        super(MBSDiscriminator, self).__init__()
        self.text_dim = text_dim
        self.extractor64 = nn.Sequential(
            # [-, nc, 64, 64]
            nn.Conv2d(n_channels, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            # [-, ndf, 32, 32]
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.ReLU(True),
            # [-, ndf*2, 16, 16]
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU(True),
            # [-, ndf*4, 8, 8]
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.ReLU(True),
            # [-, ndf*8, 4, 4]
        )
        embedding_dim = ndf * 8 + text_dim
        self.discriminator = nn.Sequential(
            # [-, embedding_dim, 4, 4]
            nn.Conv2d(embedding_dim, ndf, 1, 1, 0, bias=False),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            # [-, ndf, 4, 4]
            nn.Conv2d(ndf, 1, 3, 1, 0, bias=False),
            nn.MaxPool2d(2),
            nn.Sigmoid()
            # [-, 1, 1, 1]
        )

    def forward(self, image, text):
        image_embedding = self.extractor64(image)
        text_embedding = text.repeat(1, 1, 4, 4)
        embedding = torch.cat([image_embedding, text_embedding], dim=1)
        return self.discriminator(embedding)
