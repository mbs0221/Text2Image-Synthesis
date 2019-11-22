import torch
from generator import Generator

# Initialize generator and discriminator
text_encoder = TextEncoder(32, 32, 50)
generator = Generator()
discriminator = Discriminator()


dataloader = torch.utils.
# load vectors from file
# vectors = Vectors(name='../glove/glove.6B.200d.txt')
TEXT = data.Field(sequential=True)
TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=300))
# TEXT.vocab