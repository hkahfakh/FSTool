#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sun X, Liu Y, Wei D, et al. Selection of interdependent genes via dynamic relevance analysis for cancer diagnosis[J]. Journal of biomedical informatics, 2013, 46(2): 252-258.
"""
import numpy as np
from util.metrics.mutualInfor import mutual_info, calc_cratio

from featuresAlgorithm.base import MIFSBase


class DRGS(MIFSBase):
    def __init__(self, F, C, m):
        super(DRGS, self).__init__(F, C, m)

    def feature_selection(self):
        fc_mi_list = self.get_fc_score_list("mi")  # 所有互信息的值
        dr_list = fc_mi_list.copy()

        for _ in range(self.m):
            surplus = np.delete(np.arange(self.F.shape[1]), self.selected_feature)
            t = dr_list[surplus]
            x_j = surplus[np.argmax(t)]
            self.selected_feature.append(x_j)
            self.print_sf()
            for x_i in surplus:
                dr_list[x_i] = dr_list[x_i] + \
                               calc_cratio(self.F[:, x_i], self.F[:, x_j], self.C) * \
                               mutual_info(self.F[:, x_j], self.C)

        return self.selected_feature
