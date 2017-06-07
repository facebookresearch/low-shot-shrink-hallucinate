# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.optim
import h5py
import json
import argparse
import torch.utils.data.sampler
import os
import generation
class SimpleHDF5Dataset:
    def __init__(self, file_handle):
        self.f = file_handle
        self.all_feats_dset = self.f['all_feats'][...]
        self.all_labels = self.f['all_labels'][...]
        self.total = self.f['count'][0]
        print('here')
    def __getitem__(self, i):
        return torch.Tensor(self.all_feats_dset[i,:]), int(self.all_labels[i])

    def __len__(self):
        return self.total


# a dataset to allow for category-uniform sampling of base and novel classes.
# also incorporates hallucination
class LowShotDataset:
    def __init__(self, file_handle, base_classes, novel_classes, novel_idx, max_per_label=0, generator_fn=None, generator=None):
        self.f = file_handle
        self.all_feats_dset = self.f['all_feats']
        all_labels_dset = self.f['all_labels']
        self.all_labels = all_labels_dset[...]

        #base class examples
        self.base_class_ids = np.where(np.in1d(self.all_labels, base_classes))[0]
        total = self.f['count'][0]
        self.base_class_ids = self.base_class_ids[self.base_class_ids<total]


        # novel class examples
        novel_feats = self.all_feats_dset[novel_idx,:]
        novel_labels = self.all_labels[novel_idx]

        # hallucinate if needed
        if max_per_label>0:
            novel_feats, novel_labels = generator_fn(novel_feats, novel_labels, generator, max_per_label)
        self.novel_feats = novel_feats
        self.novel_labels = novel_labels


        self.base_classes = base_classes
        self.novel_classes = novel_classes
        self.frac = float(len(base_classes)) / float(len(novel_classes)+len(base_classes))
        self.all_classes = np.concatenate((base_classes, novel_classes))

    def sample_base_class_examples(self, num):
        sampled_idx = np.sort(np.random.choice(self.base_class_ids, num, replace=False))
        return torch.Tensor(self.all_feats_dset[sampled_idx,:]), torch.LongTensor(self.all_labels[sampled_idx].astype(int))

    def sample_novel_class_examples(self, num):
        sampled_idx = np.random.choice(self.novel_labels.size, num)
        return torch.Tensor(self.novel_feats[sampled_idx,:]), torch.LongTensor(self.novel_labels[sampled_idx].astype(int))

    def get_sample(self, batchsize):
        num_base = round(self.frac*batchsize)
        num_novel = batchsize - num_base
        base_feats, base_labels = self.sample_base_class_examples(num_base)
        novel_feats, novel_labels = self.sample_novel_class_examples(num_novel)
        return torch.cat((base_feats, novel_feats)), torch.cat((base_labels, novel_labels))

    def featdim(self):
        return self.novel_feats.shape[1]


# simple data loader for test
def get_test_loader(file_handle, batch_size=1000):
    testset = SimpleHDF5Dataset(file_handle)
    data_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return data_loader

def training_loop(lowshot_dataset, num_classes, params, batchsize=1000, maxiters=1000):
    featdim = lowshot_dataset.featdim()
    model = nn.Linear(featdim, num_classes)
    model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), params.lr, momentum=params.momentum, dampening=params.momentum, weight_decay=params.wd)

    loss_function = nn.CrossEntropyLoss()
    loss_function = loss_function.cuda()
    for i in range(maxiters):
        (x,y) = lowshot_dataset.get_sample(batchsize)
        optimizer.zero_grad()

        x = Variable(x.cuda())
        y = Variable(y.cuda())
        scores = model(x)

        loss = loss_function(scores,y)
        loss.backward()
        optimizer.step()
        if (i%100==0):
            print('{:d}: {:f}'.format(i, loss.data[0]))

    return model

def perelement_accuracy(scores, labels):
    topk_scores, topk_labels = scores.topk(5, 1, True, True)
    label_ind = labels.cpu().numpy()
    topk_ind = topk_labels.cpu().numpy()
    top1_correct = topk_ind[:,0] == label_ind
    top5_correct = np.sum(topk_ind == label_ind.reshape((-1,1)), axis=1)
    return top1_correct.astype(float), top5_correct.astype(float)


def eval_loop(data_loader, model, base_classes, novel_classes):
    model = model.eval()
    top1 = None
    top5 = None
    all_labels = None
    for i, (x,y) in enumerate(data_loader):
        x = Variable(x.cuda())
        scores = model(x)
        top1_this, top5_this = perelement_accuracy(scores.data, y)
        top1 = top1_this if top1 is None else np.concatenate((top1, top1_this))
        top5 = top5_this if top5 is None else np.concatenate((top5, top5_this))
        all_labels = y.numpy() if all_labels is None else np.concatenate((all_labels, y.numpy()))

    is_novel = np.in1d(all_labels, novel_classes)
    is_base = np.in1d(all_labels, base_classes)
    is_either = is_novel | is_base
    top1_novel = np.mean(top1[is_novel])
    top1_base = np.mean(top1[is_base])
    top1_all = np.mean(top1[is_either])
    top5_novel = np.mean(top5[is_novel])
    top5_base = np.mean(top5[is_base])
    top5_all = np.mean(top5[is_either])
    return np.array([top1_novel, top5_novel, top1_base, top5_base, top1_all, top5_all])


def parse_args():
    parser = argparse.ArgumentParser(description='Low shot benchmark')
    parser.add_argument('--lowshotmeta', required=True, type=str, help='set of base and novel classes')
    parser.add_argument('--experimentpath', required=True, type=str, help='path of experiments')
    parser.add_argument('--experimentid', default=1, type=int, help='id of experiment')
    parser.add_argument('--lowshotn', required=True, type=int, help='number of examples per novel class')
    parser.add_argument('--trainfile', required=True, type=str)
    parser.add_argument('--testfile', required=True, type=str)
    parser.add_argument('--testsetup', default=0, type=int, help='test setup or validation setup?')
    parser.add_argument('--numclasses', default=1000, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=0.001, type=float)
    parser.add_argument('--maxiters', default=10000, type=int)
    parser.add_argument('--batchsize', default=1000, type=int)
    parser.add_argument('--outdir', type=str, help='output directory for results')
    parser.add_argument('--max_per_label', default=0, type=int, help='number to generate')
    parser.add_argument('--generator_name', default='', type=str, help='type of generator')
    parser.add_argument('--generator_file', default='', type=str, help='file containing trained generator')

    return parser.parse_args()

if __name__ == '__main__':
    params = parse_args()
    with open(params.lowshotmeta, 'r') as f:
        lowshotmeta = json.load(f)
    accs = np.zeros(6)

    with open(params.experimentpath.format(params.experimentid),'r') as f:
        exp = json.load(f)
    novel_idx = np.array(exp)[:,:params.lowshotn]
    if params.testsetup:
        novel_classes = lowshotmeta['novel_classes_2']
        base_classes = lowshotmeta['base_classes_2']
    else:
        novel_classes = lowshotmeta['novel_classes_1']
        base_classes = lowshotmeta['base_classes_1']

    novel_idx = np.sort(novel_idx[novel_classes,:].reshape(-1))

    generator=None
    generator_fn=None

    if params.generator_name!='':
        generator_fn, generator = generation.get_generator(params.generator_name, params.generator_file)


    with h5py.File(params.trainfile, 'r') as f:
        lowshot_dataset = LowShotDataset(f, base_classes, novel_classes, novel_idx, params.max_per_label, generator_fn, generator)
        model = training_loop(lowshot_dataset, params.numclasses, params, params.batchsize, params.maxiters)

    print('trained')
    with h5py.File(params.testfile, 'r') as f:
        test_loader = get_test_loader(f)
        accs = eval_loop(test_loader, model, base_classes, novel_classes)

    modelrootdir = os.path.basename(os.path.dirname(params.trainfile))
    outpath = os.path.join(params.outdir, modelrootdir+'_lr_{:.3f}_wd_{:.3f}_expid_{:d}_lowshotn_{:d}_maxgen_{:d}.json'.format(
                                    params.lr, params.wd, params.experimentid, params.lowshotn, params.max_per_label))
    with open(outpath, 'w') as f:
        json.dump(dict(lr=params.lr,wd=params.wd, expid=params.experimentid, lowshotn=params.lowshotn, accs=accs.tolist()),f)


