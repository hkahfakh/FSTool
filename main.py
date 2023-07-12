#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging

from util.log_process import log_config
from util.data_process import get_data
from util.res_process import DataItem
from util.validation import k_fold_cross_validation
from util.base_info import get_one_dataset_name, inspection

if __name__ == '__main__':
    inspection()
    di = DataItem()
    log_config()

    file_name = get_one_dataset_name()
    logger = logging.getLogger('FSTool')
    logger.debug(file_name)
    data = get_data(file_name)

    X, y = data[:, :-1], data[:, -1]

    output_num = 3

    algorithm_name = "mrmr"
    acc_ave_list = k_fold_cross_validation(X, y, output_num, file_name, algorithm_name, di)  # 处理序列
    # write_data(file_name, algorithm_name, acc_ave_list)
