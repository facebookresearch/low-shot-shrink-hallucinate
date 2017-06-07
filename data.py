# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import myMetaDataset
import torchvision.datasets as dsets
import additional_transforms
dataset_dict = dict(MetaDataset=myMetaDataset.MetaDataset,
                    CocoDetection=dsets.CocoDetection,
                    CocoCaptions=dsets.CocoCaptions,
                    LSUN=dsets.LSUN,
                    CIFAR10=dsets.CIFAR10,
                    CIFAR100=dsets.CIFAR100,
                    ImageFolder=dsets.ImageFolder)


def parse_transform(transform_type, transform_params):
    if transform_type=='ImageJitter':
        method = additional_transforms.ImageJitter(transform_params['jitter_params'])
        return method
    method = getattr(transforms, transform_type)
    if transform_type=='RandomSizedCrop' or transform_type=='CenterCrop':
        return method(transform_params['image_size'])
    elif transform_type=='Scale':
        return method(transform_params['scale'])
    elif transform_type=='Normalize':
        return method(mean=transform_params['mean'], std=transform_params['std'])
    else:
        return method()



def get_data_loader(params):
    dataset_type=params['dataset_type']
    dataset_params=params['dataset_params']
    transform_params = params['transform_params']
    transform_list = [parse_transform(x, transform_params) for x in transform_params['transform_list']]
    transform = transforms.Compose(transform_list)


    dataset = dataset_dict[dataset_type](transform=transform, **dataset_params)
    data_loader_params = params['data_loader_params']
    data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
    return data_loader



