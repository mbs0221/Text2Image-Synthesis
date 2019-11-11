import re
import torch

from torch.utils.data import Dataset
from torchtext import data
from torchtext.vocab import Vectors
from torch.nn import init
from tqdm import tqdm
import numpy as np
import os

# from pycocotools.coco import COCO

import json


class Flicker(Dataset):

    def __init__(self, path):
        self.path = path
        self.load()
        self.images = []

    def load(self):
        pass

    def __len__(self):
        return self.images.__len__()

    def __getitem__(self, item):
        sample = {}
        return sample




flicker8k = '../datasets/flickr8k/Flickr8k_split_annotation/'
os.listdir(flicker8k)


def parse_token(path):
    image_captions = {}
    file = open(path)
    lines = file.readlines()
    for line in lines:
        try:
            address_number, caption = line.strip().split('\t')
            address, number = address_number.split('#')
            if image_captions.get(address) is None:
                image_captions[address] = {}
            image_captions[address][number] = caption
        except ValueError and KeyError:
            print(line)
    return image_captions


def parse_expert_annotations(path):
    image_captions = {}
    file = open(path)
    lines = file.readlines()
    for line in lines:
        image, caption_id, rank1, rank2, rank3 = line.strip().split('\t')
        caption, id = caption_id.split('#')
        score = np.mean(np.array([int(number) for number in [rank1, rank2, rank3]]))
        if image_captions.get(image) is None:
            image_captions[image] = []
        image_captions[image].append((caption, id, score))
    return image_captions


def parse_crowd_flower_annotations(path):
    image_captions = {}
    file = open(path)
    lines = file.readlines()
    for line in lines:
        image, caption_id, yeses_percent, total_yeses, total_noes = line.strip().split('\t')
        caption, id = caption_id.split('#')
        if image_captions.get(image) is None:
            image_captions[image] = []
        image_captions[image].append((id, caption, yeses_percent, total_yeses, total_noes))
    return image_captions


# Image caption
image_captions = parse_token(flicker8k + 'Flickr8k.token.txt')
print(image_captions['1370615506_2b96105ca3.jpg'])
image_captions = parse_token(flicker8k + 'Flickr8k.lemma.token.txt')
print(image_captions['1370615506_2b96105ca3.jpg'])

# expert and crowd-flower annotations judgement
expert_annotations = parse_expert_annotations(flicker8k + 'ExpertAnnotations.txt')
print(expert_annotations['1056338697_4f7d7ce270.jpg'])
crowd_flower_annotations = parse_crowd_flower_annotations(flicker8k + 'CrowdFlowerAnnotations.txt')
print(crowd_flower_annotations['1056338697_4f7d7ce270.jpg'])
