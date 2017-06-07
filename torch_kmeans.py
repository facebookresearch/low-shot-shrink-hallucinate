# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# This is a kmeans reimplementation that follows the torch unsup package
# See https://github.com/koraykv/unsup
import numpy as np


def kmeans(x, k, niter=1, batchsize=1000):
    batchsize = min(batchsize, x.shape[0])

    nsamples = x.shape[0]
    ndims = x.shape[1]

    x2 = np.sum(x**2, axis=1)
    centroids = np.random.randn(k, ndims)
    centroidnorm = np.sqrt(np.sum(centroids**2, axis=1, keepdims=True))
    centroids = centroids / centroidnorm
    totalcounts = np.zeros(k)

    for i in range(niter):
        c2 = np.sum(centroids**2, axis=1,keepdims=True)*0.5
        summation = np.zeros((k, ndims))
        counts = np.zeros(k)
        loss = 0

        for j in range(0, nsamples, batchsize):
            lastj = min(j+batchsize, nsamples)
            batch = x[j:lastj]
            m = batch.shape[0]

            tmp = np.dot(centroids, batch.T)
            tmp = tmp - c2
            val = np.max(tmp,0)
            labels = np.argmax(tmp,0)
            loss = loss + np.sum(np.sum(x2[j:lastj])*0.5 - val)

            S = np.zeros((k, m))
            S[labels, np.arange(m)] = 1
            summation = summation + np.dot(S, batch)
            counts = counts + np.sum(S, axis=1)

        for j in range(k):
            if counts[j]>0:
                centroids[j] = summation[j] / counts[j]

        totalcounts = totalcounts + counts
        for j in range(k):
            if totalcounts[j] == 0:
                idx = np.random.choice(nsamples)
                centroids[j] = x[idx]



    return centroids


