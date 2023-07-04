#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
进行回归时  需要进行标准化
lasso 回归和岭回归（ridge regression）其实就是在标准线性回归的基础上分别加入 L1 和 L2 正则化（regularization）
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

from featuresAlgorithm.base import NMFSBase


class LASSO(NMFSBase):
    def __init__(self, F, C, m=3):
        super(LASSO, self).__init__(F, C, m)
        self.alpha = 0.1

    def feature_selection(self):
        scaler = StandardScaler()
        X = scaler.fit_transform(self.F)
        lasso = Lasso(alpha=self.alpha)
        lasso.fit(X, self.C)
        surplus = np.nonzero(lasso.coef_)[0]

        self.selected_feature = list(surplus[np.argsort(np.abs(lasso.coef_[surplus]))[::-1]])  # 它这个没有顺序
        return self.selected_feature
