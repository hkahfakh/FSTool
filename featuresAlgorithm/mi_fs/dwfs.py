#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sun X, Liu Y, Xu M, et al. Feature selection using dynamic weights for classification[J]. Knowledge-Based Systems, 2013, 37: 541-549.
"""
import numpy as np

from featuresAlgorithm.base import MIFSBase
from util.metrics.mutualInfor.mutualInfor_util import calc_dw


class DWFS(MIFSBase):

    def __init__(self, F, C, m=3, **kwargs):
        super(DWFS, self).__init__(F, C, m, **kwargs)

    def feature_selection(self):
        su_list = self.get_fc_score_list("su")
        omega = np.ones(len(su_list))

        for _ in range(self.m):
            surplus = np.delete(np.arange(self.F.shape[1]), self.selected_feature)
            adj_rela_list = []
            for f_i in surplus:
                adj_rela = omega[f_i] * (su_list[f_i])
                adj_rela_list.append(adj_rela)

            f_j = surplus[np.argmax(adj_rela_list)]
            self.selected_feature.append(f_j)
            self.print_sf()

            surplus = np.delete(np.arange(self.F.shape[1]), self.selected_feature)
            for f_i in surplus:
                dw = calc_dw(self.F[:, f_i], self.F[:, f_j], self.C)
                omega[f_i] = omega[f_i] * (1 + dw)
        return self.selected_feature


if __name__ == '__main__':
    pass
