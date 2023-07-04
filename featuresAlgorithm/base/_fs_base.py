#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import warnings
import time
from abc import abstractmethod


class FSBase:
    """
    特征选择算法基类
    """

    def __init__(self, F, C, m, **kwargs):
        """

        :param F:
        :param C:
        :param m:
        :param n_jobs:
        :param selected_feature: 通过其他方法选中的特征
        """
        # initiation
        self.F = F
        self.C = C  # 标签
        self.m = m  # 要选取m个特征  最小为1

        self.selected_feature = list()
        self.n_jobs = 1
        self._spend_time = 0
        self.sf_output_stat = False  # 每次选择特征是否输出
        self.just_score_stat = False
        self.log_stat = True  # 默认输出特征选择日志

        if kwargs.get("selected_feature") is not None:
            self.selected_feature = kwargs['selected_feature']
        if kwargs.get("n_jobs") is not None:
            self.n_jobs = kwargs['n_jobs']
        if kwargs.get("log_stat") is not None:
            self.log_stat = kwargs['log_stat']
        if kwargs.get("sf_output_stat") is not None:
            self.sf_output_stat = kwargs['sf_output_stat']
        if kwargs.get("just_score_stat") is not None:
            self.just_score_stat = kwargs['just_score_stat']

        # check
        self.length_check()
        if self.m < 1:
            return

    def __del__(self):
        if self.log_stat is False:
            return
        logger = logging.getLogger('FSTool')
        logger.debug([self.get_name(), self._spend_time, len(self.selected_feature), self.selected_feature])

    def get_name(self):
        return self.__class__.__name__

    @abstractmethod
    def feature_selection(self):
        pass

    def fs2classifer(self):
        return self.F[:, self.selected_feature], self.C

    def fs_thread(self):
        t0 = time.time()
        self.feature_selection()
        t1 = time.time()
        self._spend_time = t1 - t0
        return self.selected_feature

    def length_check(self):
        if self.m > self.F.shape[1]:  # shape[1]数组列数
            self.m = self.F.shape[1]
            warnings.warn("The feature_num has to be set less or equal to {}".format(self.F.shape[1]), UserWarning)

    def print_sf(self):
        if self.sf_output_stat:
            print(self.selected_feature)


if __name__ == '__main__':
    pass
