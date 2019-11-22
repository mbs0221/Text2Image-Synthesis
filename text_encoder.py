import torch.nn as nn
from torch.nn import GRU
import torch
import torch.nn.functional as F
from torchtext.vocab import GloVe, Vectors
from torchtext import data


class TextEncoder(nn.Module):

    def __init__(self, hidden_size, output_size, max_length=30):
        super(TextEncoder, self).__init__()
        # Text-Embedding
        self.text_embedding = nn.Sequential(
            nn.Embedding(output_size, hidden_size),
            nn.Dropout(p=0.2)
        )
        # Attention module
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, max_length),
            nn.Softmax()
        )
        # Attention-combine
        self.attention_combine = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU()
        )
        # Text-Encoder
        self.gru = nn.GRU(hidden_size, hidden_size)
        # Output-encoding
        self.output_encoder = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, inputs, hidden, encoder_outputs):
        # text-embedding
        text_embedding = self.text_embedding(inputs)
        # calculate attention
        attn_weights = self.attention(torch.cat([text_embedding[0], hidden[0]], dim=1))
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        output = self.attention_combine(torch.cat((text_embedding[0], attn_applied[0]), 1)).unsqueeze(0)
        # apply attention
        output, hidden = self.gru(output, hidden)
        output = self.output_encoder(output[0])
        return output, hidden, attn_weights
