#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gao W, Hu L, Zhang P. Feature redundancy term variation for mutual information-based feature selection[J]. Applied Intelligence, 2020, 50: 1272-1288.
"""
import numpy as np
from featuresAlgorithm.base import MIFSBase
from util.metrics.mutualInfor import mutual_info, conditional_mi


class MRMD(MIFSBase):

    def __init__(self, F, C, m=3):
        super(MRMD, self).__init__(F, C, m)

    def feature_selection(self):
        mi_list = self.get_fc_score_list()
        first_redu = np.zeros(self.F.shape[1])
        two_redu = np.zeros(self.F.shape[1])
        difference_list = mi_list

        for _ in range(self.m):
            surplus = np.delete(np.arange(self.F.shape[1]), self.selected_feature)
            j = surplus[np.argmax(difference_list[surplus])]
            self.selected_feature.append(j)  # 把特征的原始索引加到已选特征里
            self.print_sf()
            fj = self.F[:, j]
            for i in surplus:
                fi = self.F[:, i]
                first_redu[i] += mutual_info(fi, fj)
                two_redu[i] += conditional_mi([fi, self.C], fj)
            redu_list = (first_redu - two_redu) / len(self.selected_feature)
            difference_list = mi_list - redu_list


if __name__ == '__main__':
    pass
