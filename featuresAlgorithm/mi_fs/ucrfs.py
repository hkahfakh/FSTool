#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Zhang P, Gao W. Feature selection considering Uncertainty Change Ratio of the class label[J]. Applied Soft Computing, 2020, 95: 106537.
"""
import numpy as np
from featuresAlgorithm.base import MIFSBase
from util.metrics.mutualInfor import conditional_mi, conditional_entropy, mutual_info


class UcrFs(MIFSBase):
    def __init__(self, F, C, m=3):
        super(UcrFs, self).__init__(F, C, m)

    def calc_ucr(self, k):
        ucr = 0
        fk = self.F[:, k]
        for j in self.selected_feature:
            fj = self.F[:, j]
            top = conditional_mi([fj, self.C], fk)
            down = conditional_entropy(self.C, fk) + conditional_entropy(self.C, [fk, fj])
            ucr += top / down
        return ucr

    def feature_selection(self):
        mi_list = self.get_fc_score_list('mi')
        redu_list = np.zeros(len(mi_list))  # 冗余列表
        criterion_list = mi_list.copy()
        ucr_list = np.zeros(len(mi_list))

        for _ in range(self.m):
            surplus = np.delete(np.arange(self.F.shape[1]), self.selected_feature)
            t = criterion_list[surplus]
            j = surplus[np.argmax(t)]
            self.selected_feature.append(j)
            self.print_sf()
            for i in surplus:
                ucr_list[i] = self.calc_ucr(i)
                redu_list[i] += mutual_info(self.F[:, i], self.F[:, j])
                criterion_list[i] = mi_list[i] + ucr_list[i] - redu_list[i] / len(self.selected_feature)
        return self.selected_feature


if __name__ == '__main__':
    pass
