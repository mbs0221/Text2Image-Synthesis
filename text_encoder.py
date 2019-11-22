import torch.nn as nn
from torch.nn import GRU


class TextEncoder(nn.Module):

    def __init__(self, embedding, text_len, hidden_len):
        super(TextEncoder, self).__init__()
        self.text_len = text_len
        self.hidden_len = hidden_len
        self.embedding = embedding
        self.gru = nn.GRU(text_len, embedding, hidden_len)
