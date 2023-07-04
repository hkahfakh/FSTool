#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging

from util.log_process import log_config
from util.data_process import get_data
from util.res_process import SaveData, write_data
from util.validation import select_feature, k_fold_cross_validation
from util.base_info import get_one_dataset_name

if __name__ == '__main__':
    feature_output_num = 1
    di = SaveData(cv_stat=True)
    log_config()

    while True:
        file_name = get_one_dataset_name()
        logger = logging.getLogger('FSTool')
        logger.debug(file_name)

        data = get_data(file_name)
        X, y = data[:, :-1], data[:, -1]

        output_num = 30
        algorithm_name = "drtwofs_test"
        fs_list = select_feature(X, y, output_num, algorithm_name, di)
        acc_ave_list = k_fold_cross_validation(X, y, fs_list, file_name, algorithm_name, di)  # 处理序列
        write_data(file_name, algorithm_name, acc_ave_list)

        if feature_output_num == data.shape[1]:
            break
        feature_output_num = feature_output_num + 1
