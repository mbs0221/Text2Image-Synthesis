import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
from skimage import io

data_type = 'train2014'

project_path = '/home/ubuntu/mbs0221/projects/datasets/'
coco_path = os.path.join(project_path, 'coco-2014/')
coco_caption_path = os.path.join(coco_path, f'annotations/captions_{data_type}.json')

coco = COCO(coco_caption_path)

# catIds = coco.getCatIds(catNms=['person', 'dog', 'skateboard'])
# imgIds = coco.getImgIds(catIds=catIds)
imgIds = coco.getImgIds()
print(f'len{len(imgIds)}')
img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
print(img)

annIds = coco.getAnnIds(imgIds=img['id'])
anns = coco.loadAnns(annIds)
coco.showAnns(anns)

im_path = os.path.join(coco_path, 'images', data_type, img['file_name'])
# im = cv2.imread(im_path)
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.imshow('image', im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

im = io.imread(im_path)
plt.axis('off')
plt.imshow(im)
plt.show()
