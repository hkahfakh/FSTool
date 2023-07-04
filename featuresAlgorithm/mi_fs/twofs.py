#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Herman G, Zhang B, Wang Y, et al. Mutual information-based method for selecting informative feature sets[J]. Pattern Recognition, 2013, 46(12): 3315-3327.
"""
import numpy as np

from featuresAlgorithm.base import MIFSBase
from util.metrics.mutualInfor import mutual_info, conditional_mi, entropy


class TwoFS(MIFSBase):
    """
    具有m个特征的特征子集S
    这些只需要处理训练集  测试集当然需要处理了   你筛选出特征来测试集那么多特征怎么用   F是训练集  原始特征

    """

    def __init__(self, F, C, m=3, c=0.4, eta=0.2):
        """

        :param F:
        :param C:
        :param m:
        :param c: lamd最小取值
        :param eta: 冗余系数
        """
        super(TwoFS, self).__init__(F, C, m)
        self.lamd = None
        self.c = c
        self.eta = eta
        self.fc_mi = self.get_fc_score_list()  # 存的每个特征和标签的互信息

    def calc_lamd(self):
        selected_mi_sum = np.sum(self.fc_mi[self.selected_feature])
        hy = entropy(self.C)
        temp_lamd = 1 - selected_mi_sum / hy
        self.lamd = max(self.c, temp_lamd)
        return self.lamd

    def feature_selection(self):
        # 互信息最大的作为第一个特征
        self.selected_feature.append(self.fc_mi.argmax())  # S里存的是原集合的下标
        self.m = self.m - 1  # 表示已经找到一个特征了

        for _ in range(self.m):

            difference_list = list()  # 这个相当于一轮的MI  因为S会变  待选和S的互信息就得重新求了    这个里面最大的要添加进S
            surplus = np.delete(np.arange(self.F.shape[1]), self.selected_feature)

            for i in surplus:
                redu_sum = 0
                for j in self.selected_feature:
                    Ixyz = conditional_mi([self.F[:, i], self.C], self.F[:, j])
                    Iij = mutual_info(self.F[:, j], self.F[:, i])
                    redu = self.eta * Ixyz + (1 - self.eta) * Iij
                    redu_sum += redu
                self.calc_lamd()
                difference = self.lamd * self.fc_mi[i] - redu_sum / len(self.selected_feature)
                difference_list.append(difference)

            difference_list = np.array(difference_list)
            self.selected_feature.append(surplus[difference_list.argmax()])  # 把特征的原始索引加到已选特征里
            self.print_sf()
        return self.selected_feature
