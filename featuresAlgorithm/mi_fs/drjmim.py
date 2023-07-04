#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hu L, Gao W, Zhao K, et al. Feature selection considering two types of feature relevancy and feature interdependency[J]. Expert Systems with Applications, 2018, 93: 423-434.
"""
import numpy as np
from collections import deque
from multiprocessing import Pool
from featuresAlgorithm.fs_util import n_jobs_change

from util.metrics.mutualInfor import mutual_info, calc_cratio, joint_mi

from featuresAlgorithm.base import MIFSBase


class DRJMIM(MIFSBase):
    def __init__(self, F, C, m, n_jobs=1):
        super(DRJMIM, self).__init__(F, C, m, n_jobs=n_jobs)
        self.all_jmi_list = [[] for _ in range(F.shape[1])]

    def feature_selection(self):
        fc_mi_list = self.get_fc_score_list("mi")  # 所有互信息的值
        dr_list = fc_mi_list.copy()
        dr_jmi_list = dr_list.copy()

        for _ in range(self.m):
            surplus = np.delete(np.arange(self.F.shape[1]), self.selected_feature)
            t = dr_jmi_list[surplus]
            j = surplus[np.argmax(t)]
            self.selected_feature.append(j)
            self.print_sf()
            if self.n_jobs > 1:
                self.n_jobs = n_jobs_change(len(surplus), self.n_jobs)
                results = deque()
                i_iterations = np.array_split(surplus, self.n_jobs)
                with Pool(processes=self.n_jobs) as pool:
                    for i in i_iterations:
                        results.append(pool.apply_async(self.sub_feature_score, (i, j, dr_list)))
                    pool.close()
                    pool.join()
                for res in results:
                    temp = res.get()
                    dr_list[temp[0]] = temp[1][temp[0]]
                    dr_jmi_list[temp[0]] = temp[2][temp[0]]
            else:
                temp = self.sub_feature_score(surplus, j, dr_list)
                dr_list[temp[0]] = temp[1][temp[0]]
                dr_jmi_list[temp[0]] = temp[2][temp[0]]
        return self.selected_feature

    def sub_feature_score(self, feature_idx, j, dr_list):
        min_JMI = np.zeros(self.F.shape[1])
        for i in feature_idx:
            dr_list[i] = dr_list[i] + \
                         calc_cratio(self.F[:, i], self.F[:, j], self.C) * \
                         mutual_info(self.F[:, j], self.C)
            self.all_jmi_list[i].append(joint_mi([self.F[:, i], self.F[:, j]], self.C))
            min_JMI[i] = np.min(self.all_jmi_list[i])

        dr_jmi_list = min_JMI * dr_list

        return feature_idx, dr_list, dr_jmi_list
