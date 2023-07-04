# !/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from random import randrange

from sklearn.preprocessing import normalize

from featuresAlgorithm.base import DISFSBase


class Relief(DISFSBase):
    def __init__(self, F, C, m=5, **kwargs):
        super(Relief, self).__init__(F, C, m, **kwargs)

    def fit(self, features, labels, iter_ratio):
        # initialization
        (n_samples, n_features) = np.shape(features)
        distance = np.zeros((n_samples, n_samples))
        weight = np.zeros(n_features)

        if iter_ratio >= 0.5:
            # compute distance
            for index_i in range(n_samples):
                for index_j in range(index_i + 1, n_samples):
                    distance[index_i, index_j] = self.distanceNorm(features[index_i], features[index_j], 'euclidean')
            distance += distance.T
        else:
            pass

        # start iteration
        for iter_num in range(int(iter_ratio * n_samples)):
            # initialization
            nearHit = list()
            nearMiss = list()
            distance_sort = list()

            # random extract a sample
            index_i = randrange(0, n_samples, 1)
            self_features = features[index_i]

            # search for nearHit and nearMiss
            if iter_ratio >= 0.5:
                distance[index_i, index_i] = np.max(distance[index_i])  # filter self-distance
                for index in range(n_samples):
                    distance_sort.append([distance[index_i, index], index, labels[index]])
            else:
                # compute distance respectively
                distance = np.zeros(n_samples)
                for index_j in range(n_samples):
                    distance[index_j] = self.distanceNorm(features[index_i], features[index_j], 'euclidean')
                distance[index_i] = np.max(distance)  # filter self-distance
                for index in range(n_samples):
                    distance_sort.append([distance[index], index, labels[index]])
            distance_sort.sort(key=lambda x: x[0])
            for index in range(n_samples):
                if nearHit == [] and distance_sort[index][2] == labels[index_i]:
                    # nearHit = distance_sort[index][1]
                    nearHit = features[distance_sort[index][1]]
                elif nearMiss == [] and distance_sort[index][2] != labels[index_i]:
                    # nearMiss = distance_sort[index][1]
                    nearMiss = features[distance_sort[index][1]]
                elif nearHit != [] and nearMiss != []:
                    break
                else:
                    continue

            # update weight
            weight = weight - np.power(self_features - nearHit, 2) + np.power(self_features - nearMiss, 2)
        return weight / (iter_ratio * n_samples)

    def feature_selection(self):
        (features, labels) = (self.F, self.C)

        features = normalize(X=features, norm='l2', axis=0)

        weight = self.fit(features, labels, 1)
        ma = np.argsort(weight, )
        self.selected_feature = list(ma[:self.m])
        return self.selected_feature
