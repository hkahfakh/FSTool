#!/usr/bin/env python
# -*- coding: utf-8 -*-
import xlwt
import xlrd
from xlutils.copy import copy

from .xlwt_base import XlwtBase
from util.base_info import get_algorithm_name_list, get_dis_dataset_list, get_classifier_list


class MyXlwt(XlwtBase):
    def __init__(self, file_name, sheet_index):
        """
        自定义类说明：
        :param sheet_name:默认sheet表对象名称，默认值为 'sheet_1'
        :param re_write: 单元格重写写功能默认开启
        """
        super(MyXlwt, self).__init__(file_name, sheet_index, False)

    def create_file(self):
        """
        如果没有文件就创建
        :return:
        """

        dataset_list = get_dis_dataset_list()
        algo_list = get_algorithm_name_list()
        classifier_list = get_classifier_list()

        workbook = xlwt.Workbook(encoding='utf-8')  # 新建工作簿
        for dataset_name in dataset_list:
            if dataset_name.find("_") != -1:
                dataset_name = dataset_name.split("_")[0]
            workbook.add_sheet(dataset_name)  # 新建sheet
        workbook.save(self.file_name)  # 保存

        rexcel = xlrd.open_workbook(self.file_name)  # 用wlrd提供的方法读取一个excel文件
        self.work_book = copy(rexcel)  # 用xlutils提供的copy方法将xlrd的对象转化为xlwt的对象
        for dataset_name in dataset_list:
            self.sheet = self.work_book.get_sheet(dataset_list.index(dataset_name))
            for idx, classifier_name in enumerate(classifier_list):
                self.write(0, 1 + len(algo_list) * idx, classifier_name)
                self.write_row(1, 1 + len(algo_list) * idx, algo_list)
            self.write_col(2, 0, list(range(1, 51)))
        self.save(self.file_name)
