#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Arias-Michel R, García-Torres M, Schaerer C E, et al. Feature selection via approximated Markov blankets using the CFS method[C]//2015 International Workshop on Data Mining with Industrial Applications (DMIA). IEEE, 2015: 38-43.
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


class FCBFCFS(MIFSBase):
    """
    具有m个特征的特征子集S
    这些只需要处理训练集  测试集当然需要处理了   你筛选出特征来测试集那么多特征怎么用   F是训练集  原始特征

    """

    def __init__(self, F, C, m=3):
        super(FCBFCFS, self).__init__(F, C, m)
        self.delta = 0

    def ss_mean(self, S):
        res = 0
        for i in S:
            for j in S:
                if i != j:
                    res += calc_relation_score(self.F[:, i], self.F[:, j], 'su')
        return 2 * res / (len(S) * (len(S) - 1))

    def feature_selection(self):
        su_list = self.get_fc_score_list("su")
        t1 = np.vstack((np.arange(self.F.shape[1]), su_list)).T
        order = t1[t1[:, 1].argsort()[::-1]]  # 按照第3列对行排序
        # Symmetrical uncertainty of selected features
        SU = []

        self.selected_feature.append(int(order[0, 0]))
        SU.append(order[0, 1])
        point = order[0, 1]

        for idx in range(1, len(order)):
            i = int(order[idx][0])
            sy_mean = (np.mean(SU) + order[idx, 1]) / 2
            ss_mean = self.ss_mean(self.selected_feature + [i])
            m = len(self.selected_feature) + 1
            J = m * sy_mean / np.sqrt(m + m * (m - 1) * ss_mean)
            if J > point:
                self.selected_feature.append(i)
                if len(self.selected_feature) == self.m:
                    return self.selected_feature
                point = J

        return self.selected_feature[:self.m]
