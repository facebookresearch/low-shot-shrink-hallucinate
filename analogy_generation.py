# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import json
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
from sklearn.cluster import KMeans
import argparse
import os
import h5py
import pickle
import torch_kmeans
class AnalogyRegressor(nn.Module):
    def __init__(self, featdim, innerdim=512):
        super(AnalogyRegressor,self).__init__()
        self.featdim = featdim
        self.innerdim = innerdim
        self.fc1 = nn.Linear(featdim*3, innerdim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(innerdim, innerdim)
        self.fc3 = nn.Linear(innerdim, featdim)

    def forward(self, a,c,d):
        x = torch.cat((a,c,d), dim=1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        return out




def init_clusters(k, dim):
    C = np.random.randn(k,dim)
    Cnorm = np.sqrt(np.sum(C**2, axis=1, keepdims=True))
    C = C/Cnorm
    return C

def cluster_feats(filehandle, base_classes, cachefile, n_clusters=100):
    if os.path.isfile(cachefile):
        with open(cachefile, 'rb') as f:
            centroids = pickle.load(f)
    else:
        centroids = []
        all_labels = filehandle['all_labels'][...]
        all_feats = filehandle['all_feats']

        count = filehandle['count'][0]
        for j, i in enumerate(base_classes):
            print('Clustering class {:d}:{:d}'.format(j,i))
            idx = np.where(all_labels==i)[0]
            idx = idx[idx<count]
            X = all_feats[idx,:]
            # use a reimplementation of torch kmeans for reproducible results
            # TODO: Figure out why this is important
            centroids_this = torch_kmeans.kmeans(X, n_clusters, 20)
            centroids.append(centroids_this)
        with open(cachefile, 'wb') as f:
            pickle.dump(centroids, f)
    return centroids



def get_difference_vectors(c_i):
    diff_i = c_i[:,np.newaxis,:] - c_i[np.newaxis,:,:]
    diff_i = diff_i.reshape((-1, diff_i.shape[2]))
    diff_i_norm = np.sqrt(np.sum(diff_i**2,axis=1, keepdims=True))
    diff_i = diff_i / (diff_i_norm + 0.00001)
    return diff_i

def mine_analogies(centroids):
    n_clusters = centroids[0].shape[0]

    analogies = np.zeros((n_clusters*n_clusters*len(centroids),4), dtype=int)
    analogy_scores = np.zeros(analogies.shape[0])
    start=0

    I, J = np.unravel_index(np.arange(n_clusters**2), (n_clusters, n_clusters))
    # for every class
    for i, c_i in enumerate(centroids):

        # get normalized difference vectors between cluster centers
        diff_i = get_difference_vectors(c_i)
        diff_i_t = torch.Tensor(diff_i).cuda()


        bestdots = np.zeros(diff_i.shape[0])
        bestdotidx = np.zeros((diff_i.shape[0],2),dtype=int)

        # for every other class
        for j, c_j in enumerate(centroids):
            if i==j:
                continue
            print(i,j)

            # get normalized difference vectors
            diff_j = get_difference_vectors(c_j)
            diff_j = torch.Tensor(diff_j).cuda()

            #compute cosine distance and take the maximum
            dots = diff_i_t.mm(diff_j.transpose(0,1))
            maxdots, argmaxdots = dots.max(1)
            maxdots = maxdots.cpu().numpy().reshape(-1)
            argmaxdots = argmaxdots.cpu().numpy().reshape(-1)

            # if maximum is better than best seen so far, update
            betteridx = maxdots>bestdots
            bestdots[betteridx] = maxdots[betteridx]
            bestdotidx[betteridx,0] = j*n_clusters + I[argmaxdots[betteridx]]
            bestdotidx[betteridx,1] = j*n_clusters + J[argmaxdots[betteridx]]


        # store discovered analogies
        stop = start+diff_i.shape[0]
        analogies[start : stop,0]=i*n_clusters + I
        analogies[start : stop,1]=i*n_clusters + J
        analogies[start : stop,2:] = bestdotidx
        analogy_scores[start : stop] = bestdots
        start = stop

    #prune away trivial analogies
    good_analogies = (analogy_scores>0) & (analogies[:,0]!=analogies[:,1]) & (analogies[:,2]!=analogies[:,3])
    return analogies[good_analogies,:], analogy_scores[good_analogies]




def train_analogy_regressor(analogies, centroids, base_classes, trained_classifier, lr=0.1, wt=10, niter=120000, step_after=40000, batchsize=128, momentum=0.9, wd=0.0001):
    # pre-permute analogies
    permuted_analogies = analogies[np.random.permutation(analogies.shape[0])]

    # create model and init
    featdim = centroids[0].shape[1]
    model = AnalogyRegressor(featdim)
    model = model.cuda()
    trained_classifier = trained_classifier.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=wd, dampening=momentum)
    loss_1 = nn.CrossEntropyLoss().cuda()
    loss_2 = nn.MSELoss().cuda()


    num_clusters_per_class = centroids[0].shape[0]
    centroid_labels = (np.array(base_classes).reshape((-1,1)) * np.ones((1, num_clusters_per_class))).reshape(-1)
    concatenated_centroids = np.concatenate(centroids, axis=0)


    start=0
    avg_loss_1 = avg_loss_2 = count = 0.0
    for i in range(niter):
        # get current batch of analogies
        stop = min(start+batchsize, permuted_analogies.shape[0])
        to_train = permuted_analogies[start:stop,:]
        optimizer.zero_grad()

        # analogy is A:B :: C:D, goal is to predict B from A, C, D
        # Y is the class label of B (and A)
        A = concatenated_centroids[to_train[:,0]]
        B = concatenated_centroids[to_train[:,1]]
        C = concatenated_centroids[to_train[:,2]]
        D = concatenated_centroids[to_train[:,3]]
        Y = centroid_labels[to_train[:,1]]

        A = Variable(torch.Tensor(A)).cuda()
        B = Variable(torch.Tensor(B)).cuda()
        C = Variable(torch.Tensor(C)).cuda()
        D = Variable(torch.Tensor(D)).cuda()
        Y = Variable(torch.LongTensor(Y.astype(int))).cuda()

        Bhat = model(A,C,D)

        lossval_2 = loss_2(Bhat, B) # simple mean squared error loss

        # classification loss
        predicted_classprobs = trained_classifier(Bhat)
        lossval_1 = loss_1(predicted_classprobs, Y)
        loss = lossval_1 + wt * lossval_2

        loss.backward()
        optimizer.step()

        avg_loss_1 = avg_loss_1 + lossval_1.data[0]
        avg_loss_2 = avg_loss_2 + lossval_2.data[0]
        count = count+1.0


        if i % 100 == 0:
            print('{:d} : {:f}, {:f}, {:f}'.format(i, avg_loss_1/count, avg_loss_2/count, count))
            avg_loss_1 = avg_loss_2 = count = 0.0

        if (i+1) % step_after == 0:
            lr = lr / 10.0
            print(lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        start = stop
        if start==permuted_analogies.shape[0]:
            start=0

    return dict(model_state=model.state_dict(), concatenated_centroids=torch.Tensor(concatenated_centroids),
            num_base_classes=len(centroids), num_clusters_per_class=num_clusters_per_class)

def train_classifier(filehandle, base_classes, cachefile, networkfile, total_num_classes = 1000, lr=0.1, wd=0.0001, momentum=0.9, batchsize=1000, niter=10000):
    # either use pre-existing classifier or train one
    all_labels = filehandle['all_labels'][...]
    all_labels = all_labels.astype(int)
    all_feats = filehandle['all_feats']
    base_class_ids = np.where(np.in1d(all_labels, base_classes))[0]
    loss = nn.CrossEntropyLoss().cuda()
    model = nn.Linear(all_feats[0].size, total_num_classes).cuda()
    if os.path.isfile(cachefile):
        tmp = torch.load(cachefile)
        model.load_state_dict(tmp)
    elif os.path.isfile(networkfile):
        tmp = torch.load(networkfile)
        if 'module.classifier.bias' in tmp['state']:
            state_dict = {'weight':tmp['state']['module.classifier.weight'], 'bias':tmp['state']['module.classifier.bias']}
        else:
            model = nn.Linear(all_feats[0].size, total_num_classes, bias=False).cuda()
            state_dict = {'weight':tmp['state']['module.classifier.weight']}
        model.load_state_dict(state_dict)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=wd, dampening=0)
        for i in range(niter):
            optimizer.zero_grad()
            idx = np.sort(np.random.choice(base_class_ids, batchsize, replace=False))
            F = all_feats[idx,:]
            F = Variable(torch.Tensor(F)).cuda()
            L = Variable(torch.LongTensor(all_labels[idx])).cuda()
            S = model(F)
            loss_val = loss(S, L)
            loss_val.backward()
            optimizer.step()
            if i % 100 == 0:
                print('Classifier training {:d}: {:f}'.format(i, loss_val.data[0]))
        torch.save(model.state_dict(), cachefile)

    return model





def train_analogy_regressor_main(trainfile, base_classes, cachedir, networkfile, initlr=0.1):

    with h5py.File(trainfile, 'r') as f:
        classification_model = train_classifier(f, base_classes, os.path.join(cachedir, 'classifier.pkl'), networkfile)
        centroids = cluster_feats(f, base_classes, os.path.join(cachedir, 'cluster.pkl'))
    if not os.path.isfile(os.path.join(cachedir, 'analogies.npy')):
        analogies, analogy_scores = mine_analogies(centroids)
        np.save(os.path.join(cachedir, 'analogies.npy'), analogies.astype(int))
    else:
        analogies = np.load(os.path.join(cachedir, 'analogies.npy'))
    generator = train_analogy_regressor(analogies, centroids, base_classes, classification_model, lr=initlr)
    return generator



def do_generate(feats, labels, generator, max_per_label):
    # generate till there are at least max_per_label examples for each label
    unique_labels = np.unique(labels)
    generations_needed = []
    generator['concatenated_centroids'] = generator['concatenated_centroids'].numpy()
    for k, lab in enumerate(unique_labels):
        # for each label
        idx = np.where(labels==lab)[0]
        # generate this many examples:
        num_to_gen = max(max_per_label - idx.size,0)
        if num_to_gen>0:
            # choose a random seed
            seed = np.random.choice(idx, num_to_gen)
            # and a random base class
            base_class = np.random.choice(generator['num_base_classes'], num_to_gen)
            # and two random centroids from this base class
            c_c = np.random.choice(generator['num_clusters_per_class'], num_to_gen)
            c_d = np.random.choice(generator['num_clusters_per_class'], num_to_gen)

            centroid_ids_c = base_class*generator['num_clusters_per_class'] + c_c
            centroid_ids_d = base_class*generator['num_clusters_per_class'] + c_d
            # add to list of things to generate
            generations_needed.append( np.concatenate((seed.reshape((-1,1)), centroid_ids_c.reshape((-1,1)), centroid_ids_d.reshape((-1,1))),axis=1))

    if len(generations_needed)>0:
        generations_needed = np.concatenate(generations_needed, axis=0)
        gen_feats = np.zeros((generations_needed.shape[0],feats.shape[1]))
        gen_labels = np.zeros(generations_needed.shape[0])


        # batch up the generations
        batchsize=1000
        for start in range(0, generations_needed.shape[0], batchsize):
            stop = min(start + batchsize, generations_needed.shape[0])
            g_idx = generations_needed[start:stop,:]
            A = Variable(torch.Tensor(feats[g_idx[:,0],:])).cuda()
            C = Variable(torch.Tensor(generator['concatenated_centroids'][g_idx[:,1],:])).cuda()
            D = Variable(torch.Tensor(generator['concatenated_centroids'][g_idx[:,2],:])).cuda()
            F = generator['model'](A,C,D).cpu().data.numpy().copy()
            gen_feats[start:stop,:] = F
            print(np.linalg.norm(F-feats[g_idx[:,0],:]), np.linalg.norm(F), np.linalg.norm(feats[g_idx[:,0],:]))
            gen_labels[start:stop] = labels[g_idx[:,0]]

        return np.concatenate((feats, gen_feats), axis=0), np.concatenate((labels, gen_labels), axis=0)
    else:
        return feats, labels


def init_generator(generator_file):
    G = torch.load(generator_file)
    featdim = G['concatenated_centroids'].size(1)
    model = AnalogyRegressor(featdim)
    model.load_state_dict(G['model_state'])
    model = model.cuda()
    G['model'] =model
    return G









