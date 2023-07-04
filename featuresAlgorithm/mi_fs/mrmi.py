#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
论文中伪代码写的不对
Wang L, Jiang S, Jiang S. A feature selection method via analysis of relevance, redundancy, and interaction[J]. Expert Systems with Applications, 2021, 183: 115365.
"""
import numpy as np

from featuresAlgorithm.base import MIFSBase
from util.metrics.mutualInfor import multiple_mi


class MRMI(MIFSBase):
    def __init__(self, F, C, m=3):
        super(MRMI, self).__init__(F, C, m)

    def feature_selection(self):
        S_prime2 = []

        mi_list = self.get_fc_score_list('mi')
        su_list = self.get_fc_score_list('su')
        self.selected_feature.append(np.argmax(su_list))
        S_prime = np.argsort(-su_list)

        self.selected_feature.append(S_prime[0])
        fi = self.F[:, S_prime[0]]

        for j in S_prime[1:]:
            Ijic = multiple_mi([self.F[:, j], fi, self.C])
            if Ijic > 0: # 冗余的按照 amb直接不要了   无关和依赖的才有资格计算MRMI
                break
            else:
                J = mi_list
                for z in self.selected_feature:
                    Izic = multiple_mi([self.F[:, z], fi, self.C])
                S_prime2.append(j)

            S_prime2 = []  # 清空

        return self.selected_feature


if __name__ == '__main__':
    pass
