#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pandas as pd


def combin_csv():
    """
    拼接路径下的所有csv文件
    :return:
    """
    folder_path = r'./dataSet/1'  # 要拼接的文件夹及其完整路径，注意不要包含中文
    save_file_path = r'./dataSet/1'  # 拼接后要保存的文件路径
    save_file_name = r'1-3.csv'  # 合并后要保存的文件名

    # # 修改当前工作目录
    # os.chdir(Folder_Path)
    # # 将该文件夹下的所有文件名存入一个列表
    file_list = os.listdir(folder_path)

    # 读取第一个CSV文件并包含表头
    df = pd.read_csv(folder_path + '\\' + file_list[0], low_memory=False)  # 编码默认UTF-8，若乱码自行更改

    # 将读取的第一个CSV文件写入合并后的文件保存
    df.to_csv(save_file_path + '\\' + save_file_name, encoding="utf_8_sig", index=False)

    # 循环遍历列表中各个CSV文件名，并追加到合并后的文件
    for i in range(1, len(file_list)):
        df = pd.read_csv(folder_path + '\\' + file_list[i], low_memory=False)
        df.to_csv(save_file_path + '\\' + save_file_name, encoding="utf_8_sig", index=False, header=False, mode='a+')
