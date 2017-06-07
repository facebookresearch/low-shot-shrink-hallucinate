# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import analogy_generation
import argparse
import json
import os
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='Low shot benchmark')
    parser.add_argument('--lowshotmeta', required=True, type=str, help='set of base and novel classes')
    parser.add_argument('--trainfile', required=True, type=str, help='Training set features')
    parser.add_argument('--numclasses', default=1000, type=int, help='Total number of classes')
    parser.add_argument('--outdir', type=str, help='output direcrory')
    parser.add_argument('--networkfile', default=None, type=str, help='Path to trained model file')
    parser.add_argument('--initlr', default=0.1, type=float, help='Initial learning rate')
    return parser.parse_args()



if __name__ == '__main__':
    params = parse_args()
    with open(params.lowshotmeta, 'r') as f:
        lowshotmeta = json.load(f)

    base_classes = lowshotmeta['base_classes_1']
    base_classes.extend(lowshotmeta['base_classes_2'])
    base_classes = sorted(base_classes)
    outdir = os.path.join(params.outdir, os.path.basename(os.path.dirname(params.trainfile)))
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    cachedir = os.path.join(outdir, 'cache')
    if not os.path.isdir(cachedir):
        os.makedirs(cachedir)

    generator = analogy_generation.train_analogy_regressor_main(params.trainfile, base_classes, cachedir, params.networkfile, initlr=params.initlr)

    torch.save(generator,os.path.join(outdir, 'generator.tar'))


