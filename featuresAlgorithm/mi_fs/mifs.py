#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Battiti R. Using mutual information for selecting features in supervised neural net learning[J]. IEEE Transactions on neural networks, 1994, 5(4): 537-550.
"""
import numpy as np

from featuresAlgorithm.base import MIFSBase
from util.metrics.mutualInfor import calc_relation_score


class MIFS(MIFSBase):
    """
    具有m个特征的特征子集S
    这些只需要处理训练集  测试集当然需要处理了   你筛选出特征来测试集那么多特征怎么用   F是训练集  原始特征

    """

    def __init__(self, F, C, m=3, beta=0.5, **kwargs):
        super(MIFS, self).__init__(F, C, m, **kwargs)
        self.beta = beta

    def feature_selection(self):
        mi_list = self.get_fc_score_list()  # 存的每个特征和标签的互信息
        redu_list = np.zeros(self.F.shape[1])
        difference_list = mi_list

        for _ in range(self.m):
            surplus = np.delete(np.arange(self.F.shape[1]), self.selected_feature)
            j = surplus[np.argmax(difference_list[surplus])]
            self.selected_feature.append(j)  # 把特征的原始索引加到已选特征里
            self.print_sf()
            for i in surplus:
                fi = self.F[:, i]
                redu_list[i] += calc_relation_score(fi, self.F[:, j], "mi")
            difference_list = mi_list - redu_list * self.beta

        return self.selected_feature
