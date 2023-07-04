#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Yang H, Moody J. Data visualization and feature selection: New algorithms for nongaussian data[J]. Advances in neural information processing systems, 1999, 12.
"""
import numpy as np
from featuresAlgorithm.base import MIFSBase
from util.metrics.mutualInfor import joint_mi


class JMI(MIFSBase):
    """
    Joint Mutual Information
    """

    def __init__(self, F, C, m=3, selected_feature=None):
        """

        :param F:
        :param C:
        :param m: 还要选几个特征
        :param selected_feature: 人为预先选定的特征   一般就是banzhaf用
        """
        super(JMI, self).__init__(F, C, m)
        if selected_feature is not None:
            self.selected_feature = selected_feature  # 已选特征的下标
        else:
            self.selected_feature = list()

    def cal_jmi_score(self):

        score_list = []
        surplus = np.delete(np.arange(self.F.shape[1]), self.selected_feature)
        for t1 in surplus:
            fi = self.F[:, t1]
            score = 0
            for t2 in self.selected_feature:
                fj = self.F[:, t2]
                score += joint_mi([fi, fj], self.C)
            score_list.append(score)
        score_list = np.array(score_list)

        return surplus, score_list

    def feature_selection(self):
        fc_mi = self.get_fc_score_list()
        if len(self.selected_feature) == 0:
            self.selected_feature.append(fc_mi.argmax())
            self.m = self.m - 1  # 表示已经找到一个特征了

        for _ in range(self.m):
            surplus, score_list = self.cal_jmi_score()
            self.selected_feature.append(surplus[score_list.argmax()])  # 找到这些和里面最大的那个,把特征的原始索引加到已选特征里
            self.print_sf()
        return self.selected_feature
