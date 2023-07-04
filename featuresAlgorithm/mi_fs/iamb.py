#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tsamardinos I, Aliferis C F, Statnikov A R, et al. Algorithms for large scale Markov blanket discovery[C]//FLAIRS conference. 2003, 2: 376-380.
用交互信息代替amb的第二条判别标准
mrmi就是这么干的
但单独这么干  效果不理想  无法超过原始amb
"""
import numpy as np

from featuresAlgorithm.base import MIFSBase
from util.metrics.mutualInfor import multiple_mi


class IAMB(MIFSBase):
    def __init__(self, F, C, m=3):
        super(IAMB, self).__init__(F, C, m)
        self.delta = -0.3  # 这个值需要重点研究

    def feature_selection(self):
        mi_list = self.get_fc_score_list("mi")
        mi_idx = np.argsort(-mi_list)
        self.selected_feature.append(mi_idx[0])

        for i in mi_idx[1:]:
            for j in self.selected_feature:
                Ijic = multiple_mi([self.F[:, j], self.F[:, i], self.C])
                if Ijic < self.delta:
                    break
            else:
                self.selected_feature.append(i)
            if len(self.selected_feature) == self.m:
                break

        return self.selected_feature
