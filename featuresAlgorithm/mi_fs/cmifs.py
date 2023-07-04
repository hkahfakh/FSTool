#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cheng, Qin Z, Feng C, et al. Conditional mutual information‐based feature selection analyzing for synergy and redundancy[J]. Etri Journal, 2011, 33(2): 210-218.
"""
import numpy as np

from util.metrics.mutualInfor import conditional_mi, mutual_info
from featuresAlgorithm.base import MIFSBase


class Cmifs(MIFSBase):
    """
    MI(X1;X2|Y)=H(X1，Y)+H(X2，Y)-H(Y)-H(X1，X2，Y)
    """

    def __init__(self, F, C, m=3):
        super(Cmifs, self).__init__(F, C, m)
        self.delta = 0.1

    def feature_selection(self):
        fc_mi = self.get_fc_score_list()  # 存的每个特征和标签的互信息
        self.selected_feature.append(fc_mi.argmax())  # 互信息最大的作为第一个特征
        self.m = self.m - 1  # 表示已经找到一个特征了

        s_1 = self.selected_feature[0]
        s_n = self.selected_feature[0]
        if self.m > 0:
            surplus = np.delete(np.arange(self.F.shape[1]), self.selected_feature)
            cmi_score = []
            for f_i in surplus:
                cmi_score.append(conditional_mi([self.F[:, f_i], self.C], self.F[:, s_1]))
            self.selected_feature.append(np.array(cmi_score).argmax())  # 互信息最大的作为第一个特征
            self.m = self.m - 1

        deleted_feature = []
        for _ in range(self.m):
            surplus = np.delete(np.arange(self.F.shape[1]),
                                np.concatenate((self.selected_feature, deleted_feature)).astype(int))
            for f_i in surplus:
                fi_mi = mutual_info(self.F[:, f_i], self.C)
                cmi_cin = conditional_mi([self.F[:, f_i], self.C], self.F[:, s_n])
                if fi_mi < 0 or (cmi_cin / fi_mi <= self.delta):
                    deleted_feature.append(f_i)

            difference_list = list()
            surplus = np.delete(np.arange(self.F.shape[1]),
                                np.concatenate((self.selected_feature, deleted_feature)).astype(int))
            for t1 in surplus:
                fi = self.F[:, t1]
                c_mi_l = conditional_mi([fi, self.C], self.F[:, s_1])
                c_mi_m = conditional_mi([fi, self.F[:, s_n]], self.F[:, s_1])
                c_mi_r = conditional_mi([fi, self.F[:, s_n]], self.C)

                difference = c_mi_l - c_mi_m + c_mi_r
                difference_list.append(difference)

            difference_list = np.array(difference_list)
            f_idx = surplus[difference_list.argmax()]
            s_n = f_idx
            self.selected_feature.append(f_idx)  # 把特征的原始索引加到已选特征里
        return self.selected_feature


if __name__ == '__main__':
    pass
