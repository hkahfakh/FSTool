#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hua Z, Zhou J, Hua Y, et al. Strong approximate Markov blanket and its application on filter-based feature selection[J]. Applied Soft Computing, 2020, 87: 105957.
"""
import numpy as np

from util.metrics.mutualInfor import calc_relation_score
from featuresAlgorithm.base import MIFSBase


class FFSG(MIFSBase):

    def __init__(self, F, C, m=3):
        super(FFSG, self).__init__(F, C, m)
        self.su_list = []
        self.su_matrix = []

    def calc_delta_prime(self):
        m = self.F.shape[1]
        numerator = 0
        denominator = m * (m - 1)
        for k in range(m):
            for l in range(m):
                if k != l:
                    numerator += calc_relation_score(self.F[:, k], self.F[:, l], "su")
        print(numerator)
        return numerator / denominator

    def sort_L(self):
        L = []
        m = self.F.shape[1]
        for i in range(m):
            li = 0
            for j in range(m):
                if i != j:
                    li += calc_relation_score(self.F[:, i], self.F[:, j], "su")
            L.append(li)
        sort_idx = np.argsort(-np.array(L))
        return np.array(L) / (m - 1), sort_idx

    def ensumble(self):
        m = self.F.shape[1]
        L = []
        for i in range(m):
            li = 0
            for j in range(m):
                if i != j:
                    self.su_matrix[i][j] = calc_relation_score(self.F[:, i], self.F[:, j], "su")
                    li += self.su_matrix[i][j]
            L.append(li)
        L = np.array(L)
        numerator = np.sum(L)
        denominator = m * (m - 1)
        delta_prime = numerator / denominator

        sort_idx = np.argsort(-L)
        return delta_prime, np.array(L) / (m - 1), sort_idx

    def algorithm1(self):
        G = []
        self.su_list = self.get_fc_score_list("su")
        self.su_matrix = np.zeros((self.F.shape[1], self.F.shape[1]))
        delta_prime, L, sort_idx = self.ensumble()
        G_t_list = []

        while len(G) != self.F.shape[1]:
            for i in sort_idx:
                if i not in G:
                    G_t = [i]
                    for j in sort_idx:
                        if j not in G:
                            a = self.su_matrix[i][j] > delta_prime
                            b = self.su_list[i] >= self.su_list[j]
                            c = self.su_matrix[i][j] >= self.su_list[j]
                            if a and b and c:
                                G_t.append(j)
                    G_t_list.append(G_t)  # G_t_list和G是同一数据的不同表现形式
                    G = np.union1d(G, G_t)  # 两个不相交的集合求并集

        G_prime = [temp[0] for temp in G_t_list]
        return G_prime

    def calc_J(self, G_prime):
        m_prime = len(G_prime)
        numerator = m_prime * np.sum(self.su_list[G_prime])
        su_p = 0
        for i in G_prime:
            for j in G_prime:
                if i != j:
                    su_p += self.su_matrix[i][j]

        denominator = np.sqrt(m_prime + su_p)
        J = numerator / denominator
        return J

    def feature_selection(self):
        G_prime = self.algorithm1()
        E_prime = [G_prime[0]]
        E_prime_prime = [G_prime[0]]
        while True:
            # a = self.calc_J(E_prime)
            # b = self.calc_J(E_prime_prime)
            if not (len(E_prime) != len(G_prime)):
                break

            surplus = np.setdiff1d(G_prime, E_prime)
            J_list = []
            for j in surplus:
                J = self.calc_J(np.union1d(E_prime, [j]))
                J_list.append(J)
            j = surplus[np.argmax(J_list)]

            E_prime_prime = np.append(E_prime, j)

            a = self.calc_J(E_prime)
            b = self.calc_J(E_prime_prime)
            if b >= a:
                E_prime = E_prime_prime
            else:
                break
            if len(E_prime) > self.m:
                break
        self.selected_feature = list(E_prime)
        return self.selected_feature
