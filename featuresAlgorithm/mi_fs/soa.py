#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Guo B, Nixon M S. Gait feature subset selection by mutual information[J]. IEEE Transactions on Systems, MAN, and Cybernetics-part a: Systems and Humans, 2008, 39(1): 36-46.
"""
import numpy as np
from featuresAlgorithm.base import MIFSBase

from util.metrics.mutualInfor import mutual_info, conditional_mi


class SOA(MIFSBase):

    def __init__(self, F, C, m=3):
        super(SOA, self).__init__(F, C, m)

    def feature_selection(self):
        mi_list = self.get_fc_score_list("mi")
        J_list = mi_list
        rela_sum = 0
        redu_sum = np.zeros(self.F.shape[1])
        inter_sum = np.zeros(self.F.shape[1])

        for _ in range(self.m):
            surplus = np.delete(np.arange(self.F.shape[1]), self.selected_feature)
            j = surplus[np.argmax(J_list[surplus])]
            self.selected_feature.append(j)
            self.print_sf()
            fj = self.F[:, j]
            rela_sum += mutual_info(fj, self.C)

            for i in surplus:
                fi = self.F[:, i]
                temp = mutual_info(fi, self.C)
                redu_sum[i] += mutual_info(fi, fj)
                inter_sum[i] += conditional_mi([fi, fj], self.C)
                J = (rela_sum + temp) - redu_sum[i] + inter_sum[i]
                J_list[i] = J
            pass


if __name__ == '__main__':
    pass
