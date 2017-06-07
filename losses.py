# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
from torch.autograd import Variable

def l2_loss(feat):
    return feat.pow(2).sum()/(2.0*feat.size(0))


def get_one_hot(labels, num_classes):

    one_hot = Variable(torch.range(0, num_classes-1)).unsqueeze(0).expand(labels.size(0), num_classes)
    if (type(labels.data) is torch.cuda.FloatTensor) or (type(labels.data) is torch.cuda.LongTensor):
        one_hot = one_hot.cuda()


    one_hot = one_hot.eq(labels.unsqueeze(1).expand_as(one_hot).float()).float()
    return one_hot

class BatchSGMLoss(nn.Module):
    def __init__(self, num_classes):
        super(BatchSGMLoss, self).__init__()
        self.softmax = nn.Softmax()
        self.num_classes = num_classes
    def forward(self,feats, scores, classifier_weight, labels):
        one_hot = get_one_hot(labels, self.num_classes)
        p = self.softmax(scores)
        if type(scores.data) is torch.cuda.FloatTensor:
            p = p.cuda()


        G = (one_hot-p).transpose(0,1).mm(feats)
        G = G.div(feats.size(0))
        return G.pow(2).sum()


class SGMLoss(nn.Module):
    def __init__(self, num_classes):
        super(SGMLoss, self).__init__()
        self.softmax = nn.Softmax()
        self.num_classes = num_classes

    def forward(self,feats, scores, classifier_weight, labels):
        one_hot = get_one_hot(labels, self.num_classes)
        p = self.softmax(scores)
        if type(scores.data) is torch.cuda.FloatTensor:
            p = p.cuda()
        pereg_wt = (one_hot - p).pow(2).sum(1)
        sqrXnorm = feats.pow(2).sum(1)
        loss = pereg_wt.mul(sqrXnorm).mean()
        return loss


class GenericLoss:
    def __init__(self,aux_loss_type, aux_loss_wt, num_classes):
        aux_loss_fns = dict(l2=l2_loss, sgm=SGMLoss(num_classes), batchsgm=BatchSGMLoss(num_classes))
        self.aux_loss_fn = aux_loss_fns[aux_loss_type]
        self.aux_loss_type = aux_loss_type
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.aux_loss_wt = aux_loss_wt

    def __call__(self, model, x_var, y_var):
        scores, feats = model(x_var)
        if self.aux_loss_type in ['l2']:
            aux_loss = self.aux_loss_fn(feats)
        else:
            classifier_weight = model.module.get_classifier_weight()
            aux_loss = self.aux_loss_fn(feats, scores, classifier_weight, y_var)
        orig_loss = self.cross_entropy_loss(scores, y_var)
        return orig_loss + self.aux_loss_wt * aux_loss



