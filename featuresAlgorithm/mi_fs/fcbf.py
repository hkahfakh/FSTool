#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Yu L, Liu H. Efficient feature selection via analysis of relevance and redundancy[J]. The Journal of Machine Learning Research, 2004, 5: 1205-1224.
"""
import numpy as np
from featuresAlgorithm.base import MIFSBase
from util.metrics.mutualInfor import calc_relation_score


def isRedundancy_simple(fi, fj, c, ruler='su'):
    """
    fi是fj的马尔科夫毯
    fj是冗余特征
    :param ruler:
    :param fi:
    :param fj:
    :param c:
    :return:
    """
    Ijc = calc_relation_score(fj, c, ruler)
    Iji = calc_relation_score(fi, fj, ruler)
    if Ijc <= Iji:
        return True


class FCBF(MIFSBase):
    """
    具有m个特征的特征子集S
    这些只需要处理训练集  测试集当然需要处理了   你筛选出特征来测试集那么多特征怎么用   F是训练集  原始特征

    """

    def __init__(self, F, C, m=3):
        super(FCBF, self).__init__(F, C, m)
        self.delta = 0

    def feature_selection(self):
        su_list = self.get_fc_score_list("su")
        t1 = np.vstack((np.arange(self.F.shape[1]), su_list)).T
        t1 = t1[np.argsort(-t1[:, 1])]
        s_list = t1[t1[:, 1] > self.delta, :]

        while len(s_list) != 0:
            # record the index of the feature with the largest su
            j = int(s_list[0, 0])
            fj = self.F[:, j]
            self.selected_feature.append(j)
            if len(self.selected_feature) == self.m:
                break
            np.delete(s_list, 0, 0)
            for i, val in s_list:
                i = int(i)
                fi = self.F[:, i]
                if calc_relation_score(fj, fi, 'su') >= val:
                    idx = s_list[:, 0] != i
                    s_list = s_list[idx]
        return self.selected_feature[:self.m]