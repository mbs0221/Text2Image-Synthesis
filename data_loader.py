import re
import torch
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchtext import data
from torchtext.vocab import Vectors
from torch.nn import init
from torchvision.datasets import VisionDataset
from tqdm import tqdm
import numpy as np
import os

import json


class GloveDB:
    def __init__(self, path):
        self.path = path

    def build(self):
        embedding = {}
        file = open(self.path)
        lines = file.readlines()
        for line in lines:
            values = line.split()
            word = values[0]
            feature = np.asarray(values[1:], dtype='float32')
            embedding[word] = feature
        file.close()
        return embedding


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
    file.close()
    return image_captions


def parse_expert_annotations(path):
    image_captions = {}
    file = open(path)
    lines = file.readlines()
    for line in lines:
        image, caption_id, rank1, rank2, rank3 = line.strip().split('\t')
        caption, id = caption_id.split('#')
        score = np.mean(np.array([float(number) / 4.0 for number in [rank1, rank2, rank3]]))
        if image_captions.get(image) is None:
            image_captions[image] = []
        image_captions[image].append((caption, id, score))
    file.close()
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


class Flicker(VisionDataset):

    def __init__(self, root, ann_file, transform=None, target_transform=None, transforms=None):
        super(Flicker, self).__init__(root, transform=transform,
                                      target_transform=target_transform, transforms=transforms)
        self.annotations = parse_token(ann_file)
        self.ids = list(sorted(self.annotations.keys()))

    def __getitem__(self, index):
        img_id = self.ids[index]

        # Image
        img = Image.open(os.path.join(self.root, img_id)).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        # Captions
        target = self.annotations[img_id]
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)


image_path = '../datasets/flickr8k/flickr8k/'
annotation_path = '../datasets/flickr8k/Flickr8k_split_annotation/'

# Image caption
dataset = Flicker(image_path, ann_file=annotation_path + 'Flickr8k.token.txt')
# dataset = Flicker(image_path, ann_file=annotation_path + 'Flickr8k.lemma.token.txt')


# expert and crowd-flower annotations judgement
expert_annotations = parse_expert_annotations(annotation_path + 'ExpertAnnotations.txt')
print(expert_annotations['1056338697_4f7d7ce270.jpg'])
crowd_flower_annotations = parse_crowd_flower_annotations(annotation_path + 'CrowdFlowerAnnotations.txt')
print(crowd_flower_annotations['1056338697_4f7d7ce270.jpg'])
