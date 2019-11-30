import torch
import torch.nn as nn
from torch.autograd import Variable


class TVLoss(nn.Module):
    def __init__(self, weight=0.2):
        super(TVLoss, self).__init__()
        self.weight = weight

    def forward(self, x):
        batch_size, _, img_width, img_height = x.size()
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :img_width - 1, :]), 2).mean()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :img_height - 1]), 2).mean()
        return self.weight * 2 * (h_tv + w_tv) / batch_size
