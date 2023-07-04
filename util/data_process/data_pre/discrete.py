#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据的离散化标准化
"""
from caimcaim import CAIMD
import numpy as np
from sklearn import preprocessing
import pandas as pd


def discretization_caimcaim(X, y, file_name="default", save_flag=1):
    X = X.astype("float64")
    y = y.astype("float64")
    caim = CAIMD()
    X = caim.fit_transform(X, y)
    if save_flag == 1:
        data1 = np.c_[X, y.T]
        np.save("./dataSet/" + file_name + "_dis.npy", data1)
    return X, y


def standardization(data):
    """
    直接标准化不知道怎么处理分母为0  所以直接调函数了
    :param data:
    :return:
    """
    x_scale = preprocessing.scale(data)
    return x_scale


def mat_quat(x, bins_num=300):
    """
    矩阵离散化
    等宽分箱
    :param bins_num:
    :param x: 矩阵
    :return:
    """
    for i in range(x.shape[1]):
        if len(np.unique(x[:, i])) > bins_num:
            x[:, i] = pd.cut(x[:, i], bins_num, labels=range(bins_num))
    return x


def Discretization_EqualFrequency(datas):
    """
    矩阵离散化
    等频分箱
    :param datas:
    :return:
    """

    def rank_qcut(vector, k):
        quantile = np.array([float(i) / k for i in range(k + 1)])  # Quantile: k+1 values
        funBounder = lambda x: (quantile >= x).argmax()
        return vector.rank(pct=True).apply(funBounder)

    FeatureNumber = datas.shape[1]
    DisDatas = np.zeros_like(datas)  # 函数主要是想实现构造一个矩阵W_update，其维度与矩阵W一致，并为其初始化为全0；
    # w = [float(i) / k for i in range(k + 1)]
    for i in range(FeatureNumber):
        k = len(np.unique(datas[:, i]))
        if k >= 10:
            k = 10
        DisOneFeature = rank_qcut(pd.Series(datas[:, i]), k)
        DisDatas[:, i] = DisOneFeature
    return DisDatas
