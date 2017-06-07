# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import ResNetBasic
import torch
import torch.nn as nn
from torch.autograd import Variable

# Modification of ResNet so that it also outputs a feature vector
class ResNetFeat(ResNetBasic.ResNet):
    def __init__(self, block, list_of_num_layers, list_of_out_dims, num_classes=1000, only_trunk=False, classifier_has_bias=True):
        super(ResNetFeat, self).__init__(block, list_of_num_layers, list_of_out_dims, num_classes, only_trunk)
        if not classifier_has_bias:
            self.classifier = nn.Linear(self.final_feat_dim, num_classes, bias=classifier_has_bias)

    def forward(self, x):
        out = self.trunk(x)
        if self.only_trunk:
            return out
        out = out.view(out.size(0),-1)
        scores = self.classifier(out)
        return scores, out

    def get_classifier_weight(self):
        return self.classifier.weight


def ResNet10(num_classes=1000, only_trunk=False):
    return ResNetFeat(ResNetBasic.SimpleBlock, [1,1,1,1],[64,128,256,512], num_classes, only_trunk, classifier_has_bias=False)

def ResNet18(num_classes=1000, only_trunk=False):
    return ResNetFeat(ResNetBasic.SimpleBlock, [2,2,2,2],[64,128,256,512],num_classes, only_trunk, classifier_has_bias=False)

def ResNet34(num_classes=1000, only_trunk=False):
    return ResNetFeat(ResNetBasic.SimpleBlock, [3,4,6,3],[64,128,256,512],num_classes, only_trunk, classifier_has_bias=False)

def ResNet50(num_classes=1000, only_trunk=False):
    return ResNetFeat(ResNetBasic.BottleneckBlock, [3,4,6,3], [256,512,1024,2048], num_classes, only_trunk, classifier_has_bias=False)

def ResNet101(num_classes=1000, only_trunk=False):
    return ResNetFeat(ResNetBasic.BottleneckBlock, [3,4,23,3],[256,512,1024,2048], num_classes, only_trunk, classifier_has_bias=False)




