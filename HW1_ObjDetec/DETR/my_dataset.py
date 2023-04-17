import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image


class MarinDataset(Dataset):
    def __init__(self, anno_path=''):
        self.anno = json.load(open(anno_path))
        self.prefix = anno_path.parent
        self.data = self.make_data()

    def make_data(self):
        data = []
        images = self.anno["images"]
        for id in range(len(images)):
            image = images[id]
            data.append({
                'image_id': id,
                'image': Image.open(os.path.join(self.prefix, image['file_name'])),
                'width': image['width'],
                'height': image['height'],
                'objects': {
                    'id': [],
                    'area': [],
                    'bbox': [],
                    'category': []
                }
            })
        
        anns = self.anno['annotations']
        for ann in anns:
            image_id = ann['image_id']
            data[image_id]['objects']['id'].append(ann['id'])
            data[image_id]['objects']['area'].append(ann['area'])
            xmin, ymin, w, h = ann['bbox']
            if not len(ann['bbox']):
                print(ann)
                input()
            data[image_id]['objects']['bbox'].append([xmin, ymin, w, h])
            data[image_id]['objects']['category'].append(ann['category_id'])
        
        return data
    
    def make_labelmaps(self):
        cats = self.anno['categories']
        id2label = {i: cats[i]['name'] for i in range(len(cats))}
        label2id = {v: k for k, v in id2label.items()}
        return id2label, label2id
    
    def tolist(self):
        return self.data

    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)
    
class MarinInferDataset(Dataset):
    def __init__(self, dir_path=''):
        self.dir = dir_path
        self.data = self.make_data()

    def make_data(self):
        data = []
        images = os.listdir(self.dir)
        for name in images:
            image = Image.open(os.path.join(self.dir, name))
            w, h = image.size
            data.append({
                'image_name': name,
                'image': image,
                'width': w,
                'height': h,
            })
        
        return data
    
    def tolist(self):
        return self.data

    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)
    
class MarinEvalDataset(Dataset):
    def __init__(self, dir_path=''):
        self.dir = dir_path
        self.anno = json.load(open(os.path.join(dir_path, "_annotations.coco.json"), 'r'))
        self.data = self.make_data()


    def make_data(self):
        data = []
        images = os.listdir(self.dir)
        for name in images:
            if ".jpg" not in name:
                continue
            image = Image.open(os.path.join(self.dir, name))
            w, h = image.size
            data.append({
                'image_name': name,
                'image': image,
                'width': w,
                'height': h,
                'target': {
                    "boxes": [],
                    "labels": [],
                }
            })
        
        anns = self.anno['annotations']
        for ann in anns:
            image_id = ann['image_id']
            xmin, ymin, w, h = ann['bbox']
            data[image_id]['target']['boxes'].append([xmin, ymin, xmin+w, ymin+h])
            data[image_id]['target']['labels'].append(ann['category_id'])
        
        
        return data
    
    def tolist(self):
        return self.data

    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)