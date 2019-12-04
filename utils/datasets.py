from torchtext import data
from torchtext.data import Example

from torchvision import transforms
from torchvision import datasets
from torchvision.datasets import Flickr8k
from torchvision.datasets.vision import StandardTransform


def get_dataset(path, ann_path, text_field, transforms=None, transform=None, target_transform=None):
    # for vision data
    has_transforms = transforms is not None
    has_separate_transform = transform is not None or target_transform is not None
    if has_transforms and has_separate_transform:
        raise ValueError("Only transforms or transform/target_transform can "
                         "be passed as argument")

    transform = transform
    target_transform = target_transform

    if has_separate_transform:
        transforms = StandardTransform(transform, target_transform)
    transforms = transforms

    # for text data
    from pycocotools.coco import COCO
    coco = COCO(ann_path)
    ids = list(sorted(coco.imgs.keys()))

    # define fields
    image_field = data.RawField(is_target=True)
    fields = [('text', text_field), ('image', image_field)]

    # collect examples
    examples = []
    for img_id in ids:
        # text
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        captions = [ann['caption'] for ann in anns]
        # real-image
        img_path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(path, img_path)).convert('RGB')
        if transforms is not None:
            img, captions = transforms(img, captions)

        examples.append(Example.fromlist([captions, img], fields))
    return examples, fields


def coco_caption(path, type, nested_field):
    examples, fields = get_dataset(
        path=f"{path}/resized/{type}2014",
        ann_path=f"{path}/annotations/captions_{type}2014.json",
        text_field=nested_field,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    )
    return data.Dataset(examples, fields)
