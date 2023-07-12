#!/usr/bin/env python
# -*- coding: utf-8 -*-
import xlwt

from util.xls_process import MyXlwt
from util.base_info import get_algorithm_name_list, get_dis_dataset_list, get_normal_dataset_list


def get_sheet(dataset_name):
    if dataset_name.find("dis") != -1:
        dis_list = get_dis_dataset_list()
        sheet = dis_list.index(dataset_name)
    else:
        normal_list = get_normal_dataset_list()
        sheet = normal_list.index(dataset_name)
    return sheet


def write_data(dataset_name, algorithm_name, data):
    """
    将算法得到的分类数据写入文件中
    :param dataset_name:
    :param algorithm_name:
    :param data:
    :return:
    """
    if algorithm_name.find("_") != -1:
        if algorithm_name.find("test") != -1:
            return
        algorithm_name = algorithm_name.split("_")[0]

    sheet = MyXlwt('tt.xls', get_sheet(dataset_name))
    start_row = 2
    algorithm_name_list = get_algorithm_name_list()
    if algorithm_name in algorithm_name_list:
        idx = algorithm_name_list.index(algorithm_name)
        sheet.write_cols(start_row, idx + 1, data, len(algorithm_name_list))
        sheet.save('tt.xls')
        print("数据总结完毕")
    else:
        raise Exception(algorithm_name, "不属于要存入tt.xls的算法 请检查excel_process.py")


def gen_tt(dataset_list, algo_list, classifier_list):
    """
    生成空白的tt文件   里面不存在数据  只有格式
    :param dataset_list:
    :param algo_list:
    :param classifier_list:
    :return:
    """
    workbook = xlwt.Workbook(encoding='utf-8')  # 新建工作簿
    for dataset_name in dataset_list:
        if dataset_name.find("_") != -1:
            dataset_name = dataset_name.split("_")[0]
        workbook.add_sheet(dataset_name)  # 新建sheet
    workbook.save(r'.\gen_tt.xls')  # 保存

    for dataset_name in dataset_list:
        sheet = MyXlwt(r'.\gen_tt.xls', get_sheet(dataset_name))
        for idx, classifier_name in enumerate(classifier_list):
            sheet.write(0, 1 + len(algo_list) * idx, classifier_name)
            sheet.write_row(1, 1 + len(algo_list) * idx, algo_list)
        sheet.write_col(2, 0, list(range(1, 51)))
        sheet.save(r'.\gen_tt.xls')

# if __name__ == '__main__':
# # 实例化自写类
# test = MyXlwt('my_test.xls', 1)
#
# l4 = ['行%s' % i for i in range(5)]
# l5 = ['列%s' % i for i in range(3)]
# ls3 = [[1, 2, 3], ['a', 'b', 'c', 'd'], ['A', 'B', 'C', 'D', 'E']]
# # 使用自写的创建单元格格式方法来创建以下两种格式
# # Times New Roman 字体，20号大小，加粗，水平居中，垂直居中
# h_s = test.diy_style('Times New Roman', 20)
# # Times New Roman 字体，10号大小，不加粗，水平左对齐，垂直居中
# s2 = test.diy_style('Times New Roman', 10, False, 1)
# # 写入数据
# # 0行，1列 按行写入数据 格式 h_s
# test.write_row(0, 1, l4, h_s)
# # 1行，0列 按列写入数据 格式 h_s
# test.write_col(1, 0, l5, h_s)
# # 1行，1列 按行写入多组数据 格式 s2
# test.write_rows(1, 1, ls3, s2)
# # 保存文件
# test.save('my_test.xls')


# gen_tt(get_dis_dataset_list(), get_algorithm_name_list(), get_classifier_list())
