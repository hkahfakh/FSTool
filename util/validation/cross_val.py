#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import logging
import numpy as np
from collections import deque
from multiprocessing import Pool
from prettytable import PrettyTable

from util.convert import list2excel
from util.res_process import SaveTime
from util.classifier import classifier_method
from featuresAlgorithm.fs_all import all_select
from .data_process import gen_k_fold_data, write_di

from util.base_info import get_classifier_list


def calc_acc_ave(di_list):
    acc_sum = 0
    for di in di_list:
        acc_sum += di.get_acc()
    return acc_sum / len(di_list)


def init_list(classifier_name):
    my_variable = []
    for x in range(len(classifier_name)):
        my_variable.append([])
    return my_variable


def select_feature(X, y, feature_output_num, algo_name, di):
    if algo_name.find("_") != -1:
        algo_name = algo_name.split('_')[0]
    sf = all_select(X, y, feature_output_num, algo_name, di)
    return sf


def multi_classifier_fs(X, y, file_name, k, di, sf, classifier_name):
    """
    返回多个分类器的分类结果
    X,y是索引
    :param X:
    :param y:
    :param file_name:
    :param k:
    :param di:
    :param sf:
    :return:
    """
    di_list = init_list(classifier_name)

    train_index_list, test_index_list = gen_k_fold_data(X, y, file_name, di, k)
    for train_index, test_index in zip(train_index_list, test_index_list):
        x1, x2 = X[train_index], X[test_index]
        y1, y2 = y[train_index], y[test_index]

        for idx, value in enumerate(classifier_name):
            di1 = classifier_method(x1, y1, x2, y2, sf, value, copy.deepcopy(di))  # 生成di
            di_list[idx].append(di1)

    return di_list


def multi_classifier(X, y, file_name, k, di, output_num, classifier_name, algorithm_name):
    """
    返回多个分类器的分类结果
    X,y是索引
    :param X:
    :param y:
    :param file_name:
    :param k:
    :param di:
    :param output_num:
    :return:
    """
    di_list = init_list(classifier_name)

    train_index_list, test_index_list = gen_k_fold_data(X, y, file_name, di, k)
    for train_index, test_index in zip(train_index_list, test_index_list):
        x1, x2 = X[train_index], X[test_index]
        y1, y2 = y[train_index], y[test_index]
        st = SaveTime()
        sf = select_feature(x1, y1, output_num, algorithm_name, di)
        expense = st.get_spend_time()

        di.update_item_fs(algorithm_name, sf, expense)
        for idx, value in enumerate(classifier_name):
            di1 = classifier_method(x1, y1, x2, y2, sf, value, copy.deepcopy(di))  # 生成di
            di_list[idx].append(di1)
    return di_list


def get_fold_path(path):
    """

    :param path:
    :return:
    """
    folder_path = path
    if folder_path.find(".") != -1:
        folder_path = folder_path.split('.')[0]
    if folder_path.find("_") != -1:
        folder_path = folder_path.split('_')[0]
    return folder_path


def k_fold_cross_validation(X, y, output_num, file_name, algorithm_name, di):
    """
    对已选特征序列进行   分类率测试
    交叉验证加载了分类器上
    :param time_flag: 是否在文件名上添加时间戳
    :param X: 特征
    :param y: 类标签
    :param feature_list: 特征序列
    :param file_name: 数据集
    :param algorithm_name: 要保存的文件名
    :param di:
    :return:
    """
    k = 10  # k-fold cross validation
    loop_times = 5 # Number of processes
    classifier_name = get_classifier_list()

    acc_ave_list = init_list(classifier_name)

    logger = logging.getLogger('FSTool')
    folder_path = get_fold_path(file_name)
    time_stamp = ''
    if algorithm_name.find("test") != -1:
        time_stamp = time.strftime("%Y_%m_%d_%H_%M_%S-", time.localtime())

    di.update_item_origin(file_name, X.shape[1], X.shape[0], k)

    feature_list = select_feature(X, y, output_num, algorithm_name, di)

    expense_time = di.get_feature_select_expense()
    for f_idx in range(1, len(feature_list) + 1):
        sf = feature_list[:f_idx]
        di.update_item_fs(algorithm_name, sf, expense_time)

        classifier_temp = init_list(classifier_name)

        if loop_times > 1:
            results = deque()
            with Pool(processes=loop_times) as pool:
                for _ in range(loop_times):
                    results.append(pool.apply_async(multi_classifier_fs, (X, y, file_name, k, di, sf, classifier_name)))
                pool.close()
                pool.join()
            for res in results:
                temp = res.get()
                for idx, [c_name, di_list] in enumerate(zip(classifier_name, temp)):
                    write_di(folder_path + '/' + time_stamp + algorithm_name + "_" + c_name, di_list)
                    classifier_temp[idx] = classifier_temp[idx] + di_list
            for idx, temp in enumerate(classifier_temp):
                acc_ave_list[idx].append(calc_acc_ave(temp))
        else:
            for _ in range(loop_times):
                classifier_di_list = multi_classifier_fs(X, y, file_name, k, di, sf, classifier_name)
                for idx, [c_name, di_list] in enumerate(zip(classifier_name, classifier_di_list)):
                    write_di(folder_path + '/' + time_stamp + algorithm_name + "_" + c_name, di_list)
                    classifier_temp[idx] = classifier_temp[idx] + di_list
            for idx, temp in enumerate(classifier_temp):
                acc_ave_list[idx].append(calc_acc_ave(temp))
        x = PrettyTable(border=False)
        x.field_names = classifier_name
        temp = np.around(np.array(acc_ave_list), decimals=4)
        x.add_row(temp[:, -1])
        x.add_column("avg", [np.around(np.mean(np.array(acc_ave_list)[:, -1], axis=0), decimals=4)])
        print(len(sf), sf)
        print("{}".format(x))

    for n, acc_ave in zip(classifier_name, acc_ave_list):
        logger.debug([n, folder_path + '/' + algorithm_name + "_" + n, acc_ave])
    logger.debug(["avg", folder_path + '/' + algorithm_name, list(np.mean(acc_ave_list, axis=0))])
    print(list2excel(list(np.mean(acc_ave_list, axis=0))))
    return acc_ave_list


def k_fold_cross_validation_fs(X, y, output_num, file_name, algorithm_name, di=None):
    """
    交叉验证加在了  特征选择和分类器上
    :param X: 
    :param y: 
    :param feature_output_num: 
    :param file_name: 
    :param di:
    :return: 
    """
    k = 10  # k折
    loop_times = 5
    classifier_name = ['svm', 'knn', 'gnb']
    acc_ave_list = init_list(classifier_name)

    logger = logging.getLogger('FSTool')
    folder_path = get_fold_path(file_name)
    time_stamp = ''
    if algorithm_name.find("test") != -1:
        time_stamp = time.strftime("%Y_%m_%d_%H_%M_%S-", time.localtime())
    di.update_item_origin(file_name, X.shape[1], X.shape[0], k)

    for f_idx in range(1, output_num + 1):
        classifier_temp = init_list(classifier_name)
        if loop_times > 1:
            results = deque()
            with Pool(processes=loop_times) as pool:
                for _ in range(loop_times):
                    results.append(pool.apply_async(multi_classifier,
                                                    (X, y, file_name, k, di, f_idx, classifier_name, algorithm_name)))
                pool.close()
                pool.join()
            for res in results:
                temp = res.get()
                for idx, [c_name, di_list] in enumerate(zip(classifier_name, temp)):
                    write_di(folder_path + '/' + time_stamp + algorithm_name + "_" + c_name, di_list)
                    classifier_temp[idx] = classifier_temp[idx] + di_list
            for idx, temp in enumerate(classifier_temp):
                acc_ave_list[idx].append(calc_acc_ave(temp))
        else:
            for _ in range(loop_times):
                classifier_di_list = multi_classifier(X, y, file_name, k, di, f_idx, classifier_name, algorithm_name)
                for idx, [c_name, di_list] in enumerate(zip(classifier_name, classifier_di_list)):
                    write_di(folder_path + '/' + time_stamp + algorithm_name + "_" + c_name, di_list)
                    classifier_temp[idx] = classifier_temp[idx] + di_list
            for idx, temp in enumerate(classifier_temp):
                acc_ave_list[idx].append(calc_acc_ave(temp))

    for n, acc_ave in zip(classifier_name, acc_ave_list):
        logger.debug([n, folder_path + '/' + algorithm_name + "_" + n, acc_ave])
    logger.debug(["avg", folder_path + '/' + algorithm_name, list(np.mean(acc_ave_list, axis=0))])
    print(list2excel(list(np.mean(acc_ave_list, axis=0))))
    return acc_ave_list
