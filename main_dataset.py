#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
同一个数据集使用所有算法进行特征选择
"""
import logging

from util.log_process import log_config
from util.data_process import get_data
from util.res_process import write_data, DataItem
from util.validation import k_fold_cross_validation
from util.base_info import get_one_dataset_name, inspection, get_algorithm_name_list


def dataset_init():
    inspection()
    log_config()

    logger = logging.getLogger('FSTool')
    logger.debug("main_dataset start")

    algorithm_name_list = get_algorithm_name_list()
    logger.debug("Excute algorithm:{} {}".format(len(algorithm_name_list),algorithm_name_list))

    file_name = get_one_dataset_name()
    logger.debug("Dataset: {}".format(file_name))

    return logger, algorithm_name_list, file_name


if __name__ == '__main__':
    logger, algorithm_name_list, file_name = dataset_init()
    for algorithm_name in algorithm_name_list:
        di = DataItem()

        acc_list = []

        data = get_data(file_name)
        X, y = data[:, :-1], data[:, -1]

        output_num = 50

        acc_ave_list = k_fold_cross_validation(X, y, output_num, file_name, algorithm_name, di)  # 处理序列
        write_data(file_name, algorithm_name, acc_ave_list)
