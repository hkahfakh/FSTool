#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sklearn_relief as relief

from featuresAlgorithm.base import DISFSBase


class ReliefF(DISFSBase):
    def __init__(self, F, C, m=5, **kwargs):
        super(ReliefF, self).__init__(F, C, m, **kwargs)

    def feature_selection(self):
        r = relief.ReliefF(
            n_features=self.m  # Choose the best 3 features

        )  # Will run by default on all processors concurrently

        my_transformed_matrix = r.fit_transform(
            self.F,
            self.C
        )
        return my_transformed_matrix


if __name__ == '__main__':
    pass
