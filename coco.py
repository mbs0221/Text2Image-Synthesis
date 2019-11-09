class coco:

    def __init__(self, data):
        self.info = data['info']
        self.licenses = data['licenses']
        self.images = data['images']
        self.annotations = data['annotations']

    def parse_info(self):
        description = self.info['description']
        url = self.info['url']
        version = self.info['version']
        year = self.info['year']
        contributor = self.info['contributor']
        date_created = self.info['date_created']
        return description, version, year, contributor, date_created, url

    def parse_licenses(self):
        list = []
        for license in self.licenses:
            url = license['url']
            id = license['id']
            name = license['name']
            list.append((id, name, url))
        return list

    def parse_images(self):
        list = []
        for image in self.images:
            license = image['license']
            file_name = image['file_name']
            coco_url = image['coco_url']
            height = image['height']
            width = image['width']
            date_captured = image['date_captured']
            flickr_url = image['flickr_url']
            id = image['id']
            list.append((id, file_name, coco_url, flickr_url))
        return list

    def parse_annotations(self):
        list = []
        for annotation in self.annotations:
            id = annotation['id']
            image_id = annotation['image_id']
            caption = annotation['caption']
            list.append((id, image_id, caption))
        return list

    def get_image_captions(self):
        image_captions = {}
        # parse images
        for image in self.images:
            id = image['id']
            file_name = image['file_name']
            image_captions[id] = {'file_name': file_name, 'captions': []}
        # parse captions
        for annotation in self.annotations:
            image_id = annotation['image_id']
            caption = annotation['caption']
            image_captions[image_id]['captions'].append(caption)
        # build image caption pairs
        return image_captions

# tokenize = lambda x: x.split()
# TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, fix_length=200)

# train_path = 'G:/coco-2014/train2014/'
# annotation_path = 'G:/coco-2014/annotations'
# file = open(os.path.join(annotation_path, 'captions_train2014.json'))
# data = json.load(file)
# obj = coco(data)
#
# image_captions = obj.get_image_captions()
# print(image_captions[57870])
#
# list = []
# for key in image_captions:
#     value = image_captions[key]
#     file_name = value['file_name']
#     for caption in value['captions']:
#         list.append((key, file_name, caption))
