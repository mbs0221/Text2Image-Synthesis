import os
import sys
import urllib
import zipfile
import re
import argparse

if sys.version_info >= (3, 0):
    import urllib.request

URLLIB = urllib
if sys.version_info >= (3, 0):
    URLLIB = urllib.request


def download_and_extract(urls, data_dir):
    print("\tstart download!")
    for url in urls:
        data_file = os.path.basename(url)
        print(f'download file: {url}')
        URLLIB.urlretrieve(url, filename=data_file)
        with zipfile.ZipFile(file=data_file) as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove(data_file)
        print("\tCompleted!")


urls = [
    'http://images.cocodataset.org/zips/train2014.zip',
    'http://images.cocodataset.org/zips/val2014.zip',
    'http://images.cocodataset.org/zips/test2014.zip',
    'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
    'http://images.cocodataset.org/annotations/image_info_test2014.zip'
]

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='coco-2014', type=str, help='root directory')
args = parser.parse_args()

download_and_extract(urls, args.root)
