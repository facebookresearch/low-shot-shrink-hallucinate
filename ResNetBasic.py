# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch.autograd import Variable
import torch.nn as nn
import math

## Basic ResNet model

def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)




# Simple ResNet Block
class SimpleBlock(nn.Module):
    def __init__(self, indim, outdim, half_res):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.BN1 = nn.BatchNorm2d(outdim)
        self.C2 = nn.Conv2d(outdim, outdim,kernel_size=3, padding=1,bias=False)
        self.BN2 = nn.BatchNorm2d(outdim)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim!=outdim:
            self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
            self.parametrized_layers.append(self.shortcut)
            self.BNshortcut = nn.BatchNorm2d(outdim)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.BN2(out)
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
        out = out + short_out
        out = self.relu2(out)
        return out



# Bottleneck block
class BottleneckBlock(nn.Module):
    def __init__(self, indim, outdim, half_res):
        super(BottleneckBlock, self).__init__()
        bottleneckdim = int(outdim/4)
        self.indim = indim
        self.outdim = outdim
        self.C1 = nn.Conv2d(indim, bottleneckdim, kernel_size=1,  bias=False)
        self.relu = nn.ReLU()
        self.BN1 = nn.BatchNorm2d(bottleneckdim)
        self.C2 = nn.Conv2d(bottleneckdim, bottleneckdim, kernel_size=3, stride=2 if half_res else 1,padding=1)
        self.BN2 = nn.BatchNorm2d(bottleneckdim)
        self.C3 = nn.Conv2d(bottleneckdim, outdim, kernel_size=1, bias=False)
        self.BN3 = nn.BatchNorm2d(outdim)

        self.parametrized_layers = [self.C1, self.BN1, self.C2, self.BN2, self.C3, self.BN3]
        self.half_res = half_res


        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim!=outdim:
            self.shortcut = nn.Conv2d(indim, outdim, 1, stride=2 if half_res else 1, bias=False)
            self.parametrized_layers.append(self.shortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)


    def forward(self, x):

        short_out = x if self.shortcut_type == 'identity' else self.shortcut(x)
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu(out)
        out = self.C2(out)
        out = self.BN2(out)
        out = self.relu(out)
        out = self.C3(out)
        out = self.BN3(out)
        out = out + short_out

        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self,block,list_of_num_layers, list_of_out_dims, num_classes=1000, only_trunk=False ):
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNet,self).__init__()
        self.grads = []
        self.fmaps = []
        assert len(list_of_num_layers)==4, 'Can have only four stages'
        conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                               bias=False)
        bn1 = nn.BatchNorm2d(64)
        relu = nn.ReLU()
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        init_layer(conv1)
        init_layer(bn1)


        trunk = [conv1, bn1, relu, pool1]
        indim = 64
        for i in range(4):

            for j in range(list_of_num_layers[i]):
                half_res = (i>=1) and (j==0)
                B = block(indim, list_of_out_dims[i], half_res)
                trunk.append(B)
                indim = list_of_out_dims[i]



        self.only_trunk=only_trunk
        if not only_trunk:
            avgpool = nn.AvgPool2d(7)
            trunk.append(avgpool)

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = indim
        if not only_trunk:
            self.classifier = nn.Linear(indim, num_classes)
            self.classifier.bias.data.fill_(0)

    def forward(self,x):
        out = self.trunk(x)
        if self.only_trunk:
            return out
        out = out.view(out.size(0),-1)
        scores = self.classifier(out)
        return scores


def ResNet10(num_classes=1000, only_trunk=False):
    return ResNet(SimpleBlock, [1,1,1,1],[64,128,256,512], num_classes, only_trunk)

def ResNet18(num_classes=1000):
    return ResNet(SimpleBlock, [2,2,2,2],[64,128,256,512],num_classes)

def ResNet34(num_classes=1000):
    return ResNet(SimpleBlock, [3,4,6,3],[64,128,256,512],num_classes)

def ResNet50(num_classes=1000):
    return ResNet(BottleneckBlock, [3,4,6,3], [256,512,1024,2048], num_classes)

def ResNet101(num_classes=1000):
    return ResNet(BottleneckBlock, [3,4,23,3],[256,512,1024,2048], num_classes)




