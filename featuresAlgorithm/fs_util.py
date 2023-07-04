#!/usr/bin/env python
# -*- coding: utf-8 -*-
def n_jobs_change(feature_index, n_jobs):
    """
    特征个数是否少于进程个数，若少则修改为特征个数
    :param feature_index:
    :param n_jobs:
    :return:
    """
    if feature_index < n_jobs:
        n_jobs = feature_index
    return n_jobs
