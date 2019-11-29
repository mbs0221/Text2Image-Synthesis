import os
import string

import numpy as np
import torch
import torch.nn as nn
from torch.nn import RNN, GRU, LSTM, RNNCell, Conv2d, Conv1d
from torchtext.vocab import GloVe, Vectors
from torchvision.transforms import transforms
from torchtext import data
import torch.nn.functional as F

class Attn(nn.Module):
    def __init__(self, hidden_dim, kqv_dim):
        super(Attn, self).__init__()
        self.wk = nn.Linear(hidden_dim,  kqv_dim)
        self.wq = nn.Linear(hidden_dim,kqv_dim)
        self.wv = nn.Linear(hidden_dim, kqv_dim)
        self.d = kqv_dim**0.5

    def forward(self, input):
        k = self.wk(input)
        q = self.wq(input)
        v = self.wv(input)
        w = F.softmax(torch.bmm(q, k.transpose(-1, -2))/self.d, dim=-1)
        attn = torch.bmm(w, v)

        return attn

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, weight, kqv_dim, rnn_type='gru', bidirectional=False, batch_first=False, padding_idx=None):
        super(TextEncoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim=emb_dim, _weight=weight)
        if rnn_type == 'rnn':
            self.rnn = RNN(emb_dim, hidden_size, bidirectional=bidirectional, num_layers=6, batch_first=batch_first)
        elif rnn_type == 'gru':
            self.rnn = GRU(emb_dim, hidden_size, bidirectional=bidirectional, num_layers=6, batch_first=batch_first)
        elif rnn_type == 'lstm':
            self.rnn = LSTM(emb_dim, hidden_size, bidirectional=bidirectional, num_layers=6, batch_first=batch_first)

        self.attn = Attn(emb_dim, kqv_dim)
        self.linear = nn.Linear(emb_dim, 2)


    def forward(self, input_ids):
        output = self.embed(input_ids)
        output, hidden = self.rnn(output)
        output = self.attn(output)
        output = output[-1].unsqueeze(2).unsqueeze(3)
        return output


def load_glove(worddict, file_path):
    GloveVocab = {}
    f = open(file_path)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        GloveVocab[word] = coefs
    f.close()

    num_words = len(worddict)
    embedding_dim = len(list(GloveVocab.values())[0])
    embedding_matrix = np.zeros((num_words, embedding_dim))

    for word, i in worddict.items():
        if word in GloveVocab:
            embedding_matrix[i] = np.array(GloveVocab[word], dtype=float)
        else:
            if word == "_PAD_" or word == "_OOV_":
                continue
            embedding_matrix[i] = np.random.normal(size=(embedding_dim))

    return embedding_matrix

def ConstructWordDict(text_list):
    worddict = {
        '_PAD_': 0,
        '_OOV_': 1
    }
    dictnum = 2
    for sentence in text_list:
        for i in sentence:
            subsen = i.split(' ')
            for n in subsen:
                if n not in worddict.keys():
                    worddict[n] = dictnum
                    dictnum = dictnum + 1

    return worddict

if __name__ == '__main__':

    text_list = []
    with open('/Users/zhangxiaoou/PycharmProjects/DeepLearningTeamWork/venv/data/Flickr8k_text/Flickr8k.token.txt',
              'r') as f:
        for line in f:
            line = line.split('\t', 1)[1].strip('\n').split(',')
            text_list.append(line)
    f.close()

    worddict = ConstructWordDict(text_list)
    print(len(worddict))

    glove_file_path = '/Users/zhangxiaoou/PycharmProjects/DeepLearningTeamWork/venv/data/glove.6B/glove.6B.50d.example.txt'
    glove = load_glove(worddict, glove_file_path)

    embedding_matrix = torch.FloatTensor(glove)
    num_words, num_features = embedding_matrix.shape

    print(embedding_matrix.shape)
    kqv_dim = 128
    text_encoder = TextEncoder(num_words, num_features, 64, embedding_matrix, 128)
    print(text_encoder)

    """
    inputs = [
        [2, 3, 1, 0, 0],
        [1, 2, 3, 0, 0]
    ]
    inputs = torch.tensor(inputs, dtype=torch.long)
    text_embedding = textencoder(inputs)
    text_embedding = torch.transpose(text_embedding, 2, 1)
    model = nn.Sequential(
        Conv1d(64, 48, 3),
        Conv1d(48, 48, 3),
        nn.Sigmoid()
    )
    outputs = model(text_embedding)
    outputs = outputs.reshape(shape=[-1, 3, 4, 4])
    """