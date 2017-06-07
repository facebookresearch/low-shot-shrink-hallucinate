# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch.autograd import Variable
import myMetaDataset
import ResNetFeat
import yaml
import data
import os
import argparse
import numpy as np
import h5py

def save_features(model, data_loader, outfile ):

    f = h5py.File(outfile, 'w')
    max_count = len(data_loader)*data_loader.batch_size
    all_labels = f.create_dataset('all_labels',(max_count,), dtype='i')
    all_feats=None
    count=0
    for i, (x,y) in enumerate(data_loader):
        if i%10 == 0:
            print('{:d}/{:d}'.format(i, len(data_loader)))
        x = x.cuda()
        x_var = Variable(x)
        scores, feats = model(x_var)
        if all_feats is None:
            all_feats = f.create_dataset('all_feats', (max_count, feats.size(1)), dtype='f')
        all_feats[count:count+feats.size(0),:] = feats.data.cpu().numpy()
        all_labels[count:count+feats.size(0)] = y.cpu().numpy()
        count = count + feats.size(0)

    count_var = f.create_dataset('count', (1,), dtype='i')
    count_var[0] = count

    f.close()

def get_model(model_name, num_classes):
    model_dict = dict(ResNet10 = ResNetFeat.ResNet10,
                ResNet18 = ResNetFeat.ResNet18,
                ResNet34 = ResNetFeat.ResNet34,
                ResNet50 = ResNetFeat.ResNet50,
                ResNet101 = ResNetFeat.ResNet101)
    return model_dict[model_name](num_classes, False)



def parse_args():
    parser = argparse.ArgumentParser(description='Save features')
    parser.add_argument('--cfg', required=True, help='yaml file containing config for data')
    parser.add_argument('--outfile', required=True, help='save file')
    parser.add_argument('--modelfile', required=True, help='model file')
    parser.add_argument('--model', type=str, default='ResNet10', help='model')
    parser.add_argument('--num_classes', type=int,default=1000)
    return parser.parse_args()

if __name__ == '__main__':
    params = parse_args()
    with open(params.cfg,'r') as f:
        data_params = yaml.load(f)

    data_loader = data.get_data_loader(data_params)
    model = get_model(params.model, params.num_classes)
    model = model.cuda()
    model = torch.nn.DataParallel(model)
    from torch.utils.serialization import load_lua
    #tmp = load_lua('/home/bharathh/local/cachedir/from_lua.t7')
    tmp = torch.load(params.modelfile)
    if ('module.classifier.bias' not in model.state_dict().keys()) and ('module.classifier.bias' in tmp['state'].keys()):
        tmp['state'].pop('module.classifier.bias')

    model.load_state_dict(tmp['state'])
    model.eval()

    dirname = os.path.dirname(params.outfile)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    save_features(model, data_loader, params.outfile)
