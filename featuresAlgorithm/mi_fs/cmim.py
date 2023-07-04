#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fleuret F. Fast binary feature selection with conditional mutual information[J]. Journal of Machine learning research, 2004, 5(9).
"""
import numpy as np

from util.metrics.mutualInfor import conditional_mi
from featuresAlgorithm.base import MIFSBase


class Cmim(MIFSBase):
    """
    MI(X1;X2|Y)=H(X1，Y)+H(X2，Y)-H(Y)-H(X1，X2，Y)
    """

    def __init__(self, F, C, m=3):
        super(Cmim, self).__init__(F, C, m)

    def sp(self, F, sf):
        f_list = []
        for s_idx in sf:
            f_list.append(F[:, s_idx])
        return f_list

    def feature_selection(self):
        fc_mi = self.get_fc_score_list()  # 存的每个特征和标签的互信息
        self.selected_feature.append(fc_mi.argmax())  # 互信息最大的作为第一个特征
        self.m = self.m - 1  # 表示已经找到一个特征了

        for _ in range(self.m):
            difference_list = list()
            surplus = np.delete(np.arange(self.F.shape[1]), self.selected_feature)
            for t1 in surplus:
                # 这个传入的F里不应包括fi
                # 但是如果删掉  就无法对应他是第几个了   所以需要两个列表 index，他与标签的互信息
                fi = self.F[:, t1]
                c_mi = conditional_mi([fi, self.C], self.sp(self.F, self.selected_feature))
                difference = fc_mi[t1] - c_mi
                difference_list.append(difference)
            difference_list = np.array(difference_list)
            self.selected_feature.append(surplus[difference_list.argmax()])  # 把特征的原始索引加到已选特征里
        return self.selected_feature


if __name__ == '__main__':
    pass
