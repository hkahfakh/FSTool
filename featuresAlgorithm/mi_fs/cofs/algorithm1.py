#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import comb
from itertools import combinations
from collections import deque
from multiprocessing import Pool

from util.metrics.mutualInfor import conditional_mi, mutual_info


class Banzhaf:
    def __init__(self, F, C, Omega=3, Tau=0.5, n_jobs=1):
        """

        :param F: 特征
        :param C: 类标签
        :param Omega:
        :param Tau:
        """
        self.F = F
        self.C = C
        self.Tau = Tau
        self.Omega = Omega
        self.n_jobs = n_jobs

    def _create_coalitions(self, i):
        """
        just use index as F not real data
        Input: original feature sub feature fi F, limit value Omega
        :param i:
        :return: coalition array
        """
        iter = combinations(np.delete(np.arange(self.F.shape[1]), i), 1)
        return list(iter)

    def _relation_detection(self, i, j):
        """
        Input: 传入特征下标
        Output: coalition array
        所有的都传下标吧
        :param j:
        :param i:
        :return:
        """
        fi = self.F[:, i]
        fj = self.F[:, j]
        MI1 = conditional_mi([fj, self.C], fi)
        MI2 = mutual_info(fj, self.C)

        if MI1 > MI2:
            return True
        else:
            return False

    def _get_feature_number(self, coalition, i):
        zi, mi = 0, 0
        for j in coalition:
            if self._relation_detection(j, i):
                zi = zi + 1  # 相互依存
            else:
                mi = mi + 1  # 冗余或独立
        return zi, mi

    # Input: coalition , threshold Tau
    def _is_win(self, coalition, i):
        """
        Zi（K）是与特征fi相互依赖的特征个数
        mi（K）是与特征fi冗余或独立的特征个数。
        :param coalition: 联盟
        :param i: 待选特征下标
        :return: 这是边际贡献  也是并上特征fi后联盟的胜败
        """
        zi, mi = self._get_feature_number(coalition, i)

        if zi == 0:
            return True if mi != 0 else False

        p = mi / zi
        return True if p >= self.Tau else False

    # Input:
    # Output: normalized Pv vector
    def _normalizedPv(self, data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

    def calc_bi(self, i_list):
        Pv = []
        for i in i_list:
            coalitions = self._create_coalitions(i)
            n = len(coalitions)

            w_1 = 0
            for coalition in coalitions:
                if self._is_win(coalition, i):
                    w_1 += 1

            if w_1 == 0:
                w_2 = 0
                w_3 = 0
            elif w_1 == 1:
                w_2 = n - 1
                w_3 = comb(n - 1, 2)
            elif w_1 == 2:
                w_2 = comb(w_1, 2) + comb(w_1, 1) * comb(n - w_1, 1)
                w_3 = comb(n - 1, 2) + comb(n - 2, 2)
            else:
                w_2 = comb(w_1, 2) + comb(w_1, 1) * comb(n - w_1, 1)
                w_3 = comb(w_1, 3) + comb(w_1, 1) * comb(n - w_1, 2) + comb(w_1, 2) * comb(n - w_1, 1)

            win_times = w_1 + w_2 + w_3
            n = comb(n, 1) + comb(n, 2) + comb(n, 3)
            bi = win_times / n
            Pv.append(bi)
        return Pv

    def banzhaf_power_index(self):
        """
        Input: A training sample O with feature space F and the target C.
        Output: Pv: Banzhaf power index vector of F.
        :return:
        """
        Pv = []

        if self.n_jobs > 1:
            results = deque()
            i_iterations = np.arange(self.F.shape[1])
            i_iterations = np.array_split(i_iterations, self.n_jobs)

            with Pool(processes=self.n_jobs) as pool:
                for i in i_iterations:
                    results.append(pool.apply_async(self.calc_bi, ([i])))
                pool.close()
                pool.join()

            for res in results:
                Pv = Pv + res.get()
        else:
            Pv = self.calc_bi(range(self.F.shape[1]))
        return np.array(Pv)
        # return self.normalizedPv(Pv)

    def updataF(self, F, C):
        self.F = F
        self.C = C


if __name__ == '__main__':
    pass
