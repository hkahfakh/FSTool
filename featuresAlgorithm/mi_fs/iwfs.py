#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Zeng Z, Zhang H, Zhang R, et al. A novel feature selection method considering feature interaction[J]. Pattern Recognition, 2015, 48(8): 2656-2666.
"""
import numpy as np

from featuresAlgorithm.base import MIFSBase
from util.metrics.mutualInfor.mutualInfor_util import calc_iw


class IWFS(MIFSBase):
    def __init__(self, F, C, m=3):
        super(IWFS, self).__init__(F, C, m)

    def feature_selection(self):
        su_list = self.get_fc_score_list("su")
        omega = np.ones(len(su_list))

        for _ in range(self.m):
            surplus = np.delete(np.arange(self.F.shape[1]), self.selected_feature)
            adj_rela_list = []
            for f_i in surplus:
                adj_rela = omega[f_i] * (1 + su_list[f_i])
                adj_rela_list.append(adj_rela)

            f_j = surplus[np.argmax(adj_rela_list)]
            self.selected_feature.append(f_j)
            self.print_sf()

            surplus = np.delete(np.arange(self.F.shape[1]), self.selected_feature)
            for f_i in surplus:
                iw = calc_iw(self.F[:, f_i], self.F[:, f_j], self.C)
                omega[f_i] = omega[f_i] * iw
        return self.selected_feature


if __name__ == '__main__':
    pass
