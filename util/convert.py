#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.io as sio


def transform2blanket(x):
    new_x = []
    for p in range(x.shape[0]):
        new_x.append([x[p]])

    return np.array(new_x)


def mat2other(mat_data, file_type):
    """

    :param mat_data: sio.loadmat(file_path)
    :return:
    """
    X = mat_data['X']  # data
    X = X.astype(float)
    y = mat_data['Y']  # label
    y = np.array([y[:, 0]]).T
    data = np.hstack((X, y))

    return data


def csv2other(csv_data, file_type=None):
    """

    :param csv_data: pd.read_csv(file_path, low_memory=False)
    :param file_type:
    :return:
    """
    data = csv_data.values
    a = np.array(data)
    return a


def excel2list(excel_text):
    """
    excel 复制str的转成list
    :param excel_text:
    :return:
    """
    finally_list = []
    l1 = excel_text.split('\n')
    for l2 in l1:
        if len(l2) != 0:
            finally_list.append(l2.split('\t'))
    finally_list = np.array(finally_list).astype("float64").T
    return finally_list


def excellist2list(excellist):
    """
    excel 复制str list的转成list
    :param excellist:
    :return:
    """
    finally_list = []
    for e in excellist:
        finally_list.append(excel2list(e))
    return finally_list


def list2excel(data_list):
    """
    列表转excel  竖直形式
    :param data_list:
    :return:
    """
    str1 = ""
    for temp in data_list:
        str1 = str1 + "\n" + str(temp)
    return str1
