#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reshef D N, Reshef Y A, Finucane H K, et al. Detecting novel associations in large data sets[J]. science, 2011, 334(6062): 1518-1524.
"""
import numpy as np
from featuresAlgorithm.base import MIFSBase

class Mic(MIFSBase):
    def __init__(self, F, C, m=5):
        super(Mic, self).__init__(F, C, m, )

    def cal_mid_score(self):
        fc_mi = self.get_fc_score_list()
        if len(self.selected_feature) == 0:
            self.selected_feature.append(fc_mi.argmax())
            self.m = self.m - 1  # 表示已经找到一个特征了

        difference_list = []
        surplus = np.delete(np.arange(self.F.shape[1]), self.selected_feature)
        for t1 in surplus:
            fi = self.F[:, t1]
            mi_sum = self.cal_mi_sum(fi)
            difference = fc_mi[t1] - mi_sum / len(self.selected_feature)
            difference_list.append(difference)
        difference_list = np.array(difference_list)

        return surplus, difference_list

    def feature_selection(self):
        fc_mi = self.get_fc_score_list()
        if len(self.selected_feature) == 0:
            self.selected_feature.append(fc_mi.argmax())
            self.m = self.m - 1  # 表示已经找到一个特征了

        for _ in range(self.m):
            surplus, difference_list = self.cal_mid_score()
            self.selected_feature.append(surplus[difference_list.argmax()])  # 找到这些和里面最大的那个,把特征的原始索引加到已选特征里
        return self.selected_feature


if __name__ == '__main__':
    pass
