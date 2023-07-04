#!/usr/bin/env python
# -*- coding: utf-8 -*-
import util.metrics.calc_dis as cd
from featuresAlgorithm.base._fs_base import FSBase


class DISFSBase(FSBase):
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
        super(DISFSBase, self).__init__(F, C, m, **kwargs)

    def distanceNorm(self, v1, v2, Norm):
        # Norm for distance
        return getattr(cd, Norm)(v1, v2)


if __name__ == '__main__':
    print(DISFSBase(1, 2, 3).distanceNorm([1, 2, 3], [1, 2, 3], 'euclidean'))
