# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import os
identity = lambda x:x
class MetaDataset:
    def __init__(self, rootdir='/mnt/fair/imagenet-256', meta='/home/bharathh/imagenet_meta/train.json', transform=transforms.ToTensor(), target_transform=identity):
        with open(meta, 'r') as f:
            self.meta = json.load(f)
        self.rootdir=rootdir
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self,i):
        image_path = os.path.join(self.rootdir, self.meta['image_names'][i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.meta['image_labels'][i])
        return img, target

    def __len__(self):
        return len(self.meta['image_names'])








