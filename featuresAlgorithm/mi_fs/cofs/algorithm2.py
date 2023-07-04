#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

from featuresAlgorithm.mi_fs import Mim, mRMR
from .algorithm1 import Banzhaf
from featuresAlgorithm.base import MIFSBase


def get_fi(victory):
    """
    返回胜利准则最大的  那个特征的下标
    :param victory: numpy数组
    :return:
    """
    return int(np.argmax(victory))


def calcValue(F, C):
    """
    准则函数
    :param F: 特征
    :param C: 类
    :return: 返回普通特征选择方法每个特征的价值 numpy数组
    """
    m = Mim(F, C)
    return m.get_fc_score_list()


class Cofs(MIFSBase):

    def __init__(self, F, C, m):
        super(Cofs, self).__init__(F, C, m)

    def calc_mrmr_value(self):
        """
        准则函数

        :return: 返回普通特征选择方法每个特征的价值 numpy数组
        """
        m = mRMR(self.F, self.C, 1, selected_feature=self.selected_feature, log_stat=False, just_score_stat=True)
        return m.feature_selection()

    def feature_selection(self):
        """
        选出最好的几个特征  直接返回  而不是下标
        :param F:
        :param C:
        :param delta: 选择特征的个数
        :return: 选出来的特征下标  S选出来的特征
        """
        fc_mi = self.get_fc_score_list()
        if len(self.selected_feature) == 0:
            self.selected_feature.append(fc_mi.argmax())
            self.m = self.m - 1  # 表示已经找到一个特征了
            self.print_sf()

        for _ in range(self.m):
            surplus = np.delete(np.arange(self.F.shape[1]), self.selected_feature)
            J = self.calc_mrmr_value()
            Pv = Banzhaf(self.F[:, surplus], self.C).banzhaf_power_index()
            victory = J * Pv
            i = surplus[get_fi(victory)]
            self.selected_feature.append(i)
            self.print_sf()

        return self.selected_feature
