#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- coding:gb2312 -*-
import json


def read_txt_high(filename):
    with open(filename, 'r') as file_to_read:
        list0 = []  # 文件中的第一列数据
        list1 = []  # 文件中的第二列数据
        while True:
            lines = file_to_read.readline()  # 整行读取数据
            if not lines:
                break
            item = [i for i in lines.split()]
            data0 = json.loads(item[0])  # 每行第一个值
            data1 = json.loads(item[1])  # 每行第二个值
            list0.append(data0)
            list1.append(data1)
    return list0, list1


if __name__ == '__main__':
    aa, bb = read_txt_high('data.log')
    print(aa)
    print(bb)
