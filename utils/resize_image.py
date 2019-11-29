import argparse
import os

from PIL import Image
from torchvision import transforms

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', default=64, type=int, help='the image size')
    parser.add_argument('--source', default='../datasets/coco-2014/images/', type=str, help='source image directory')
    parser.add_argument('--target', default='../datasets/coco-2014/resized/', type=str, help='target image directory')
    args = parser.parse_args()
    image_size = args.image_size
    source = args.source
    target = args.target

    transform = transforms.Resize([image_size, image_size])
    folders = ['train2014', 'test2014', 'val2014']
    for folder in folders:
        list = os.listdir(os.path.join(source, folder))
        if not os.path.exists(os.path.join(target, folder)):
            os.mkdir(os.path.join(target, folder))
        for item in list:
            image = Image.open(os.path.join(source, folder, item))
            resized = transform(image)
            path = os.path.join(target, folder, item)
            if not os.path.exists(path):
                print(f'save: {path}')
                resized.save(path)
            else:
                print(f'pass: {path}')
