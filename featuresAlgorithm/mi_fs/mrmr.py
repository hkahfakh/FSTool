#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Peng H, Long F, Ding C. Feature selection based on mutual information criteria of max-dependency, max-relevance, and min-redundancy[J]. IEEE Transactions on pattern analysis and machine intelligence, 2005, 27(8): 1226-1238.
"""
import numpy as np
from featuresAlgorithm.base import MIFSBase
from util.metrics.mutualInfor import calc_relation_score


class mRMR(MIFSBase):
    """
    mrmr分为mid和miq两种   这里是mid
    具有m个特征的特征子集S
    这些只需要处理训练集  测试集当然需要处理了   你筛选出特征来测试集那么多特征怎么用   F是训练集  原始特征
    """

    def __init__(self, F, C, m=3, **kwargs):
        """

        :param F:
        :param C:
        :param m: 还要选几个特征
        :param selected_feature: 人为预先选定的特征
        """
        super(mRMR, self).__init__(F, C, m, **kwargs)
        self.rela_list = np.array([])
        self.redu_list = np.array([])

    def fs_init(self):
        """
        没有预先输入特征  就进行正常特征获取
        如果有  对这些特征进行必要的处理 与数据获取
        :return:
        """
        self.rela_list = self.get_fc_score_list()
        self.redu_list = np.zeros(self.F.shape[1])
        surplus = np.delete(np.arange(self.F.shape[1]), self.selected_feature)
        for i in surplus:
            fi = self.F[:, i]
            for j in self.selected_feature:
                fj = self.F[:, j]
                self.redu_list[i] = calc_relation_score(fi, fj, "mi")
        if len(self.selected_feature) > 0:
            return self.rela_list - self.redu_list / len(self.selected_feature)
        return self.rela_list

    def feature_selection(self):
        difference_list = self.fs_init()

        for _ in range(self.m):
            surplus = np.delete(np.arange(self.F.shape[1]), self.selected_feature)
            if self.just_score_stat:
                return difference_list[surplus]
            self.selected_feature.append(surplus[difference_list[surplus].argmax()])  # 找到这些和里面最大的那个,把特征的原始索引加到已选特征里
            self.print_sf()
            for t1 in surplus:
                fi = self.F[:, t1]
                self.redu_list[t1] += calc_relation_score(fi, self.F[:, self.selected_feature[-1]], "mi")
            difference_list = self.rela_list - self.redu_list / len(self.selected_feature)
            difference_list = np.array(difference_list)

        return self.selected_feature
