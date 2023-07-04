#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bennasar M, Hicks Y, Setchi R. Feature selection using joint mutual information maximisation[J]. Expert Systems with Applications, 2015, 42(22): 8520-8532.
"""
import numpy as np
from featuresAlgorithm.base import MIFSBase

from util.metrics.mutualInfor import joint_mi


class JMIM(MIFSBase):

    def __init__(self, F, C, m=3):
        super(JMIM, self).__init__(F, C, m)

    def feature_selection(self):
        mi_list = self.get_fc_score_list("mi")
        all_jmi_list = [[] for _ in range(len(mi_list))]
        min_jmi_list = mi_list

        for _ in range(self.m):
            surplus = np.delete(np.arange(self.F.shape[1]), self.selected_feature)
            j = surplus[np.argmax(min_jmi_list[surplus])]
            self.selected_feature.append(j)
            self.print_sf()
            for i in surplus:
                all_jmi_list[i].append(joint_mi([self.F[:, i], self.F[:, j]], self.C))
                min_jmi_list[i] = np.min(all_jmi_list[i])

        return self.selected_feature


if __name__ == '__main__':
    pass
