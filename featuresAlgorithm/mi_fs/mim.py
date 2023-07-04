#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
from featuresAlgorithm.base import MIFSBase


class Mim(MIFSBase):

    def __init__(self, F, C, m=3):
        super(Mim, self).__init__(F, C, m)

    def selectFeature(self, S_MI):
        S = list()
        for j in range(self.m):
            idx = S_MI.argmax()
            S.append(idx)
            S_MI[idx] = -1
        return S

    def feature_selection(self):
        """
        选出与类标签之间互信息最大的几个特征
        :return: S 已选特征集合
        """
        S_MI = self.get_fc_score_list()
        self.selected_feature = self.selectFeature(S_MI)
        return self.selected_feature


if __name__ == '__main__':
    pass
