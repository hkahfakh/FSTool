#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import csv
from datetime import datetime
from util.res_process.data_item import DataItem


class SaveTime:
    def __init__(self):
        self.start_time = datetime.now()
        self.end_time = datetime.now()

    def get_spend_time(self):
        self.end_time = datetime.now()
        return (self.end_time - self.start_time).total_seconds()


class SaveData:
    def __init__(self, reduce_stat=False, cv_stat=False):
        self.path = "./ExperimentalData/"
        self.file_name = "ExperimentalData.csv"
        self.temp_file_name = None
        self.debug_mode = False
        self.reduce_stat = reduce_stat
        self.cv_stat = cv_stat  # 交叉验证

        self.di = DataItem(reduce_stat)

        self._check_file()

    def _delete_data(self):
        try:
            if self.debug_mode:
                os.remove(self.path + self.file_name)  # 删除文件
        except FileNotFoundError:
            pass

    def _check_file(self):
        """
        判断路径下是否有目标文件
        :return:
        """
        file_name = self.get_file_name()
        self._delete_data()
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        if not os.path.exists(self.path + file_name):
            self._create_file()

    def _create_file(self):
        dataset_head = ["时间", "数据集", "原始特征个数", "原始样本数"]
        dr_cv_head = ["划分比例", ]
        if self.cv_stat:
            dr_cv_head = ["交叉验证K", ]
        reduce_head = ["降维方法", "降维输出特征", "降维花费", ]
        fs_head = ["特征选择算法", "特征选择输出特征", "特征选择花费", ]
        classifier_head = ["分类器名称", "准确率", "召回率", "分类花费", "入侵和良性样本个数"]
        if self.reduce_stat is not True:
            reduce_head = []

        sheet_head = dataset_head + dr_cv_head + reduce_head + fs_head + classifier_head
        file_name = self.get_file_name()
        with open(self.path + file_name, 'wt') as f:
            cw = csv.writer(f, lineterminator='\n')
            # 采用writerow()方法
            cw.writerow(sheet_head)  # 将列表的每个元素写到csv文件的一行

    def set_temp_file_name(self, file_name):
        self.file_name = file_name

    def get_file_name(self):
        if self.temp_file_name is None:
            file_name = self.file_name
        else:
            file_name = self.temp_file_name
        return file_name

    def write_item(self, di):
        di.update_time()
        item = di.get_value()
        # print(item)  # 显示每条保存的数据
        file_name = self.get_file_name()
        self._check_file()  # 判断临时文件是否存在
        with open(self.path + file_name, 'a') as f:
            cw = csv.writer(f, lineterminator='\n')
            cw.writerow(item)  # 将列表的每个元素写到csv文件的一行

    def run(self, di):
        self.write_item(di)
        self.set_temp_file_name(None)
