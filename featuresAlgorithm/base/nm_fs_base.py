#!/usr/bin/env python
# -*- coding: utf-8 -*-
from featuresAlgorithm.base._fs_base import FSBase


class NMFSBase(FSBase):
    """
    距离特征选择算法基类
    """

    def __init__(self, F, C, m, **kwargs):
        """

        :param F:
        :param C:
        :param m:
        :param n_jobs:
        :param selected_feature: 通过其他方法选中的特征
        """
        super(NMFSBase, self).__init__(F, C, m, **kwargs)



