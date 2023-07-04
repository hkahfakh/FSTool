#!/usr/bin/env python
# -*- coding: utf-8 -*-
from util.data_process.data_gen import getData as gd
from util.res_process import SaveData


def gen_k_fold_data(X, y, file_name, di=None, k=10):
    """
    返回k折交叉验证数据的索引，有di就更新di数据
    :param X:
    :param y:
    :param file_name:
    :param di:
    :param k:
    :return:
    """
    train_index_list, test_index_list = gd.k_flod_data(X, y, k)
    if di is not None:
        di.update_item_origin(file_name, X.shape[1], X.shape[0], k)
    return train_index_list, test_index_list


def write_classifier_data(name, acc_list, rec_list, expense_list, di):
    """
    为di添加属性，然后保存到临时文件
    :param name:
    :param acc_list:
    :param rec_list:
    :param expense_list:
    :param di:
    :return:
    """
    sd = SaveData(cv_stat=True)
    for acc, rec, expense in zip(acc_list, rec_list, expense_list, ):
        di.update_item_classifier(name, acc, rec, expense, ["positive_num", "negative_num"])
        sd.set_temp_file_name(name + ".csv")
        sd.run(di)  # 写入数据


def write_di(name, di_list):
    """
    写入di列表到临时csv
    :param name:
    :param di_list:
    :return:
    """
    sd = SaveData(cv_stat=True)
    for di in di_list:
        sd.set_temp_file_name(name + ".csv")
        sd.run(di)  # 写入数据
