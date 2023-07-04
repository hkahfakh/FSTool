#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

"""
用Numpy实现常见距离度量
https://zhuanlan.zhihu.com/p/132682864
"""

def euclidean(x, y):
    """
    欧氏距离(Euclidean distance)
    :param x:
    :param y:
    :return:
    """
    return np.sqrt(np.sum((x - y) ** 2))


def manhattan(x, y):
    """
    曼哈顿距离(Manhattan distance)
    :param x:
    :param y:
    :return:
    """
    return np.sum(np.abs(x - y))


def chebyshev(x, y):
    """
    切比雪夫距离(Chebyshev distance)
    :param x:
    :param y:
    :return:
    """
    return np.max(np.abs(x - y))


def minkowski(x, y, p):
    """
    闵可夫斯基距离(Minkowski distance)
    :param x:
    :param y:
    :param p:
    :return:
    """
    return np.sum(np.abs(x - y) ** p) ** (1 / p)


def hamming(x, y):
    """
    汉明距离(Hamming distance)
    :param x:
    :param y:
    :return:
    """
    return np.sum(x != y) / len(x)
