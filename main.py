# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.



import ResNetFeat
import torch
from torch.autograd import Variable
import myMetaDataset
import data
import torch.optim
import time
import argparse
import yaml
import os
import glob
import numpy as np
import losses

def accuracy(scores, labels):
    topk_scores, topk_labels = scores.topk(5, 1, True, True)
    label_ind = labels.cpu().numpy()
    topk_ind = topk_labels.cpu().numpy()
    top1_correct = np.sum(topk_ind[:,0] == label_ind)
    top5_correct = np.sum(topk_ind == label_ind.reshape((-1,1)))
    return float(top1_correct), float(top5_correct)


def adjust_learning_rate(optimizer, epoch, params):
    lr = params.lr * (params.lr_decay ** (epoch // params.step_size))
    if epoch<params.warmup_epochs:
        lr = params.warmup_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr








def main_training_loop(train_loader, val_loader, model, loss_fn, start_epoch, stop_epoch, params):
    # init timing
    data_time = 0
    sgd_time = 0
    test_time = 0


    optimizer = torch.optim.SGD(model.parameters(), params.lr, momentum=params.momentum, weight_decay=params.weight_decay, dampening=params.dampening)
    for epoch in range(start_epoch,stop_epoch):
        adjust_learning_rate(optimizer, epoch, params)
        model.train()


        # start timing
        data_time=0
        sgd_time=0
        test_time=0
        start_data_time=time.time()
        avg_loss=0

        #train
        for i, (x,y) in enumerate(train_loader):
            data_time = data_time + (time.time()-start_data_time)
            x = x.cuda()
            y = y.cuda()
            start_sgd_time=time.time()
            optimizer.zero_grad()
            x_var = Variable(x)
            y_var = Variable(y)

            loss = loss_fn(model, x_var, y_var)
            loss.backward()
            optimizer.step()
            sgd_time = sgd_time + (time.time()-start_sgd_time)

            avg_loss = avg_loss+loss.data[0]

            if i % params.print_freq==0:
                print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d}/{:d} | Batch {:d}/{:d} | Loss {:f} | Data time {:f} | SGD time {:f}'.format(epoch,
                    stop_epoch, i, len(train_loader), avg_loss/float(i+1), data_time/float(i+1), sgd_time/float(i+1)))
            start_data_time = time.time()


        #test
        model.eval()
        data_time=0
        start_data_time = time.time()
        top1=0
        top5=0
        count = 0
        for i, (x,y) in enumerate(val_loader):
            data_time = data_time + (time.time()-start_data_time)
            x = x.cuda()
            y = y.cuda()
            start_test_time = time.time()
            x_var = Variable(x)
            scores = model(x_var)[0]
            top1_this, top5_this = accuracy(scores.data, y)
            top1 = top1+top1_this
            top5 = top5+top5_this
            count = count+scores.size(0)
            test_time = test_time + time.time()-start_test_time
            if (i%params.print_freq==0) or (i==len(val_loader)-1):
                print('Epoch {:d}/{:d} | Batch {:d}/{:d} | Top-1 {:f} | Top-5 {:f} | Data time {:f} | Test time {:f}'.format(epoch,
                    stop_epoch, i, len(val_loader), top1/float(count), top5/float(count), data_time/float(i+1), test_time/float(i+1)))



        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            if not os.path.isdir(params.checkpoint_dir):
                os.makedirs(params.checkpoint_dir)
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

    return model


def parse_args():
    parser = argparse.ArgumentParser(description='Main training script')
    parser.add_argument('--traincfg', required=True, help='yaml file containing config for data')
    parser.add_argument('--valcfg', required=True, help='yaml file containing config for data')
    parser.add_argument('--model', default='ResNet18', help='model: ResNet{10|18|34|50}')
    parser.add_argument('--lr', default=0.1, type=float, help='Initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--weight_decay', default=0.0001, type=float, help='Weight decay')
    parser.add_argument('--lr_decay', default=0.1, type=float, help='Learning rate decay')
    parser.add_argument('--step_size', default=30, type=int, help='Step size')
    parser.add_argument('--print_freq', default=10, type=int,help='Print frequecy')
    parser.add_argument('--save_freq', default=10, type=int, help='Save frequency')
    parser.add_argument('--start_epoch', default=0, type=int,help ='Starting epoch')
    parser.add_argument('--stop_epoch', default=90, type=int, help ='Stopping epoch')
    parser.add_argument('--allow_resume', default=0, type=int)
    parser.add_argument('--resume_file', default=None, help='resume from file')
    parser.add_argument('--checkpoint_dir', required=True, help='Directory for storing check points')
    parser.add_argument('--aux_loss_type', default='l2', type=str, help='l2 or sgm or batchsgm')
    parser.add_argument('--aux_loss_wt', default=0.1, type=float, help='loss_wt')
    parser.add_argument('--num_classes',default=1000, type=float, help='num classes')
    parser.add_argument('--dampening', default=0, type=float, help='dampening')
    parser.add_argument('--warmup_epochs', default=0, type=int, help='iters for warmup')
    parser.add_argument('--warmup_lr', default=0.01, type=int, help='lr for warmup')

    return parser.parse_args()




def isfile(x):
    if x is None:
        return False
    else:
        return os.path.isfile(x)


def get_model(model_name, num_classes):
    model_dict = dict(ResNet10 = ResNetFeat.ResNet10,
                ResNet18 = ResNetFeat.ResNet18,
                ResNet34 = ResNetFeat.ResNet34,
                ResNet50 = ResNetFeat.ResNet50,
                ResNet101 = ResNetFeat.ResNet101)
    return model_dict[model_name](num_classes, False)




def get_resume_file(filename):
    if isfile(filename):
        return filename
    filelist = glob.glob(os.path.join(params.checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file




if __name__=='__main__':
    np.random.seed(10)
    params = parse_args()
    with open(params.traincfg,'r') as f:
        train_data_params = yaml.load(f)
    with open(params.valcfg,'r') as f:
        val_data_params = yaml.load(f)

    train_loader = data.get_data_loader(train_data_params)
    val_loader = data.get_data_loader(val_data_params)
    model = get_model(params.model, params.num_classes)
    model = model.cuda()
    model = torch.nn.DataParallel(model)

    loss_fn = losses.GenericLoss(params.aux_loss_type, params.aux_loss_wt, params.num_classes)


    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)
    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch
    if params.allow_resume:
        resume_file = get_resume_file(params.resume_file)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch']+1
            model.load_state_dict(tmp['state'])

    model = main_training_loop(train_loader, val_loader, model, loss_fn, start_epoch, stop_epoch, params)
