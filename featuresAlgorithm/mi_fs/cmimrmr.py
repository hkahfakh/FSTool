#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gu X, Guo J, Xiao L, et al. Conditional mutual information-based feature selection algorithm for maximal relevance minimal redundancy[J]. Applied Intelligence, 2022, 52(2): 1436-1447.
"""
import numpy as np
from featuresAlgorithm.base import MIFSBase
from util.metrics.mutualInfor import mutual_info, conditional_mi


class CmiMrmr(MIFSBase):
    def __init__(self, F, C, m=3):
        super(CmiMrmr, self).__init__(F, C, m)

    def feature_selection(self):
        mi_list = self.get_fc_score_list()  # 存的每个特征和标签的互信息
        self.selected_feature.append(mi_list.argmax())  # 互信息最大的作为第一个特征
        self.m = self.m - 1  # 表示已经找到一个特征了

        s = self.selected_feature[0]

        surplus = np.delete(np.arange(self.F.shape[1]), self.selected_feature)
        i_is_list = []
        i_sci_list = []
        for i in surplus:
            i_is_list.append(mutual_info(self.F[:, i], self.F[:, s]))
            i_sci_list.append(conditional_mi([self.F[:, s], self.C], self.F[:, i]))
        criterion = mi_list[surplus] - np.array(i_is_list) + np.array(i_sci_list)
        self.selected_feature.append(criterion.argmax())  # 互信息最大的作为第一个特征
        self.m = self.m - 1  # 表示已经找到一个特征了

        for _ in range(self.m):
            i_is_list = []
            i_sci_list = []
            cmi_sum_list = []

            surplus = np.delete(np.arange(self.F.shape[1]), self.selected_feature)
            for i in surplus:
                cmi_sum = 0
                mi_sum = 0
                cmi2_sum = 0
                for s in self.selected_feature:
                    for j in self.selected_feature:
                        if s != j:
                            cmi_sum += conditional_mi([self.F[:, j], self.F[:, s]], self.F[:, i])
                    mi_sum += mutual_info(self.F[:, i], self.F[:, s])
                    cmi2_sum += conditional_mi([self.F[:, s], self.C], self.F[:, i])
                i_is_list.append(mi_sum)
                i_sci_list.append(cmi2_sum)
                cmi_sum_list.append(cmi_sum)
            criterion = mi_list[surplus] - (np.array(i_is_list) + np.array(i_sci_list) - np.array(cmi_sum_list)
                                            / (len(self.selected_feature) - 1)) / len(self.selected_feature)
            self.selected_feature.append(surplus[criterion.argmax()])  # 互信息最大的作为第一个特征
            self.print_sf()
            self.m = self.m - 1  # 表示已经找到一个特征了


if __name__ == '__main__':
    pass
