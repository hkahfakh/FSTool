#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gao W, Hu L, Zhang P. Class-specific mutual information variation for feature selection[J]. Pattern Recognition, 2018, 79: 328-339.
"""
import numpy as np
from featuresAlgorithm.base import MIFSBase

from util.metrics.mutualInfor import conditional_mi, mutual_info


class DCSF(MIFSBase):
    def __init__(self, F, C, m=3):
        super(DCSF, self).__init__(F, C, m)

    def feature_selection(self):
        mi_list = self.get_fc_score_list("mi")
        maxs = np.zeros(len(mi_list))
        if len(self.selected_feature) == 0:
            self.selected_feature.append(maxs.argmax())
            self.m = self.m - 1  # 表示已经找到一个特征了
        for _ in range(self.m):
            surplus = np.delete(np.arange(self.F.shape[1]), self.selected_feature)
            for i in surplus:
                fi = self.F[:, i]
                fj = self.F[:, self.selected_feature[-1]]
                left = conditional_mi([fi, self.C], fj)
                center = conditional_mi([fj, self.C], fi)
                right = mutual_info(fi, fj)
                three_sum = left + center - right
                maxs[i] += three_sum
        return self.selected_feature


if __name__ == '__main__':
    pass
