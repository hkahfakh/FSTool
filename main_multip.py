#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
十折交叉验证加在了   分类器上
同一个算法对所有实验数据集进行选择并分类

"""
import logging

from util.log_process import log_config
from util.data_process import get_data
from util.res_process import write_data, DataItem
from util.validation import k_fold_cross_validation
from util.base_info import inspection, get_formal_dataset_list, get_temp_algorithm_name_list


def multip_init():
    inspection()
    log_config()
    logger = logging.getLogger('FSTool')
    logger.debug("main_multip start")
    algorithm_name_list = get_temp_algorithm_name_list()
    logger.debug("Excute algorithm: {} {}".format(len(algorithm_name_list), algorithm_name_list))
    dataset_list = get_formal_dataset_list()
    logger.debug("Excute datasets: {} {}".format(len(dataset_list), dataset_list))
    return logger, algorithm_name_list, dataset_list


if __name__ == '__main__':
    logger, algorithm_name_list, dataset_list = multip_init()
    for algorithm_name in algorithm_name_list:
        for idx, file_name in enumerate(dataset_list):
            di = DataItem()
            logger.debug(file_name)

            data = get_data(file_name)
            X, y = data[:, :-1], data[:, -1]

            output_num = 30

            acc_ave_list = k_fold_cross_validation(X, y, output_num, file_name, algorithm_name, di)  # 处理序列
            write_data(file_name, algorithm_name, acc_ave_list)
