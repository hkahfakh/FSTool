#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
There seems to be a problem with the original paper
Gu X, Guo J, Li C, et al. A feature selection algorithm based on redundancy analysis and interaction weight[J]. Applied Intelligence, 2021, 51(4): 2672-2686.
"""
import numpy as np

from util.metrics.mutualInfor import entropy, multiple_mi
from util.metrics.mutualInfor import calc_su, calc_if

from featuresAlgorithm.base import MIFSBase


class RAIW(MIFSBase):

    def __init__(self, F, C, m=3):
        super(RAIW, self).__init__(F, C, m)

    def sub_feature_score(self, selected_feature_idx, feature_idx):
        redu_list = []
        hc = entropy(self.C)
        for i in feature_idx:
            redu_sum = 0
            hi = entropy(self.F[:, i])
            for j in selected_feature_idx:
                su = 1 - calc_su(self.F[:, i], self.F[:, j])
                hj = entropy(self.F[:, j])
                Ixyz = multiple_mi([self.F[:, j], self.F[:, i], self.C])
                Ixyz = 2 * Ixyz / (hc + hi + hj)
                redu_sum += Ixyz
            redu_list.append(redu_sum)
        return redu_list

    def feature_selection(self):
        """
        选出最好的几个特征  直接返回  而不是下标
        :param F:
        :param C:
        :param delta: 选择特征的个数
        :return: 选出来的特征下标  S选出来的特征
        """
        omage = np.ones(self.F.shape[1])
        su_list = self.get_fc_score_list("su")
        max_idx = np.argmax(su_list)
        self.selected_feature.append(max_idx)

        alpha = 1
        J_list = np.zeros(self.F.shape[1])
        for _ in range(self.m):
            surplus = np.delete(np.arange(self.F.shape[1]), self.selected_feature)

            for i in surplus:
                for j in self.selected_feature:
                    fi = self.F[i]
                    fj = self.F[j]
                    if_value = calc_if(fi, fj, self.C)

                omage = omage * (1 + if_value)
                J = su_list[surplus] * [1 - alpha * su_list[surplus]] * omage

            i = int(np.argmax(J_list))
            self.selected_feature.append(surplus[i])
            self.print_sf()

        return self.selected_feature
