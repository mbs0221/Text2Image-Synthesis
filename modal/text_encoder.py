import torch.nn as nn
from torch.nn import GRU
import torch
import torch.nn.functional as F
from torchtext.vocab import GloVe, Vectors
from torchtext import data


class TextEncoder(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_size, kqv_dim, weight, rnn_type='gru', bidirectional=False,
                 batch_first=False, padding_idx=None):
        """

        :param vocab_size: The local vocabulary size
        :param embedding_dim: the dimensional of the word embedding
        :param hidden_size: the dimensional of extracted features
        :param weight: the weight of the Embedding layer
        :param rnn_type: RNN type
        :param bidirectional: if we use bi-directional RNN
        :param batch_first:
        """
        super(TextEncoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim=embedding_dim, _weight=weight, padding_idx=padding_idx)
        if rnn_type == 'rnn':
            self.rnn = RNN(embedding_dim, hidden_size, bidirectional=bidirectional, num_layers=2,
                           batch_first=batch_first)
        elif rnn_type == 'gru':
            self.rnn = GRU(embedding_dim, hidden_size, bidirectional=bidirectional, num_layers=2,
                           batch_first=batch_first)
        elif rnn_type == 'lstm':
            self.rnn = LSTM(embedding_dim, hidden_size, bidirectional=bidirectional, num_layers=2,
                            batch_first=batch_first)

    def forward(self, input_ids):
        # get word embedding
        word_vectors = self.embed(input_ids)
        # get outputs from RNN module
        # word_vectors = self.attn(word_vectors)
        output, h_c = self.rnn(word_vectors)
        # get text-embedding from RNN output
        # embedding = output[-1]
        # reshape [batch, channel, 1, 1]
        # return embedding.unsqueeze(2).unsqueeze(3)
        return output


class AttnTextEncoder(nn.Module):

    def __init__(self, hidden_size, output_size, max_length=50):
        super(AttnTextEncoder, self).__init__()
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
