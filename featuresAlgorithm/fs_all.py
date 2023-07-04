#!/usr/bin/env python
# -*- coding: utf-8 -*-
import inspect

from util.res_process import SaveTime


def get_algorithm(name):
    """
    通过字符串，返回对应的类对象
    :param name:
    :return:
    """
    from . import dis_fs, mi_fs, norm_fs
    c = inspect.getmembers(dis_fs, inspect.isclass)
    c += inspect.getmembers(mi_fs, inspect.isclass)
    c += inspect.getmembers(norm_fs, inspect.isclass)
    temp_list = [i[0].lower() for i in c]
    if name.lower() in temp_list:
        p = temp_list.index(name.lower())
        return c[p][1]
    return False


def all_select(train_X, train_y, feature_output_num, algo_name, di=None, temp_f=None):
    """

    :param algo_name: 算法名
    :param train_X: 特征
    :param train_y: 类标签
    :param feature_output_num: 特征输出数目
    :param di: DataItem
    :param temp_f:  已选特征
    :return:
    """

    algo = get_algorithm(algo_name)
    st = SaveTime()
    if temp_f is not None:
        select_algo = algo(train_X[:, temp_f], train_y, feature_output_num)
        selected_f = select_algo.fs_thread()
        selected_f = temp_f.take(selected_f)
    else:
        select_algo = algo(train_X, train_y, feature_output_num, )
        selected_f = select_algo.fs_thread()
    expense = st.get_spend_time()
    if di is not None:
        di.update_item_fs(select_algo.get_name(), selected_f, expense)

    return selected_f
