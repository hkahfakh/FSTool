#!/usr/bin/env python
# -*- coding: utf-8 -*-

from featuresAlgorithm.base._fs_base import FSBase
from util.metrics.mutualInfor import mutual_info, calc_fc_score_list


class MIFSBase(FSBase):
    """
    互信息特征选择算法基类
    """

    def __init__(self, F, C, m, **kwargs):
        """

        :param F:
        :param C:
        :param m:
        :param n_jobs:
        :param selected_feature: 通过其他方法选中的特征
        """
        super(MIFSBase, self).__init__(F, C, m, **kwargs)

    def get_fc_score_list(self, ruler="mi"):
        return calc_fc_score_list(self.F, self.C, ruler)

    def cal_mi_sum(self, fi):
        """
        计算待选特征与已选中各个特征的互信息之和
        :param fi: 待选择的特征
        :return: 返回这些互信息的和
        """
        mi_sum = 0.0
        for j in self.selected_feature:
            fj = self.F[:, j]
            mi = mutual_info(fi, fj)
            mi_sum = mi_sum + mi
        return mi_sum


if __name__ == '__main__':
    pass
