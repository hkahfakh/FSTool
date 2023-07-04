#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
xlwt文件 基础操作
"""
import os
import warnings
import xlwt
import xlrd
from xlwt import Style
from xlutils.copy import copy


class XlwtBase(object):
    def __init__(self, file_name, sheet_index, debug_mode):
        """

        :param file_name: 保存数据的excel表文件名
        :param sheet_index: 第几个表 进行数据读写
        """
        self.debug_mode = debug_mode
        self.col_data = {}
        self.file_name = file_name
        self.sheet_index = sheet_index
        self.delete_data()
        self.check_file()  # 文件不存在就去创建
        rexcel = xlrd.open_workbook(self.file_name)  # 用wlrd提供的方法读取一个excel文件
        self.check_sheet(rexcel)  # 工作表不够  弹出异常
        self.work_book = copy(rexcel)  # 用xlutils提供的copy方法将xlrd的对象转化为xlwt的对象   所有的表都在这里面
        self.sheet = self.work_book.get_sheet(sheet_index)  # 用xlwt对象的方法获得要操作的sheet  这个是可以随时变动

    def delete_data(self):
        # self.debug_mode = False
        try:
            if self.debug_mode:
                os.remove(self.file_name)  # 删除文件
        except FileNotFoundError:
            pass

    def check_file(self):
        """
        判断路径下是否有目标文件
        :return:
        """
        self.delete_data()
        if not os.path.exists(self.file_name):
            self.create_file()

    def check_sheet(self, rexcel):
        sn = rexcel.sheet_names()
        if self.sheet_index > len(sn):
            warnings.warn("sheet not existence")

    def create_file(self):
        """
        如果没有文件就创建
        :return:
        """
        workbook = xlwt.Workbook(encoding='utf-8')  # 新建工作簿
        workbook.add_sheet("Sheet")  # 新建sheet
        workbook.save(self.file_name)  # 保存

    def save(self, file_name):
        self.work_book.save(file_name)

    def write(self, row, col, label, style=Style.default_style):
        """
        在默认sheet表对象一个单元格内写入数据
        :param style:
        :param row: 写入行
        :param col: 写入列
        :param label: 写入数据
        """

        self.sheet.write(row, col, label, style)

        # 将列数据加入到col_data字典中
        if col not in self.col_data.keys():
            self.col_data[col] = []
            self.col_data[col].append(label)
        else:
            self.col_data[col].append(label)

    def write_row(self, start_row, start_col, date_list, style=Style.default_style, bold_list=None):
        """
        按行写入一行数据
        :param style:
        :param start_row:写入行序号
        :param start_col: 写入列序号
        :param date_list: 写入数据：列表
        :return: 返回行对象
        """
        tnr_bold = self.diy_style('Times New Roman', 10)
        tnr = self.diy_style('Times New Roman', 10, False)
        if bold_list is not None:
            for col, [label, b] in enumerate(zip(date_list, bold_list)):
                if b:
                    self.write(start_row, start_col + col, label, tnr_bold)
                else:
                    self.write(start_row, start_col + col, label, tnr)
        else:
            for col, label in enumerate(date_list):
                self.write(start_row, start_col + col, label, style)
        return self.sheet.row(start_row)

    def write_rows(self, start_row, start_col, data_lists, style=Style.default_style):
        """
        按行写入多组数据
        :param style:
        :param start_row: 开始写入行序号
        :param start_col: 写入列序号
        :param data_lists: 列表嵌套列表数据
        :return: 返回写入行对象列表
        """
        row_obj = []
        for row_, data in enumerate(data_lists):
            if isinstance(data, list):
                self.write_row(start_row + row_, start_col, data, style)
                row_obj.append(self.sheet.row(start_row + row_))
            else:
                msg = '数据列表不是嵌套列表数据，而是%s' % type(data)
                raise Exception(msg)
        return row_obj

    def write_col(self, start_row, start_col, date_list, style=Style.default_style):
        """
        按列写入一列数据
        :param style:
        :param start_row:写入行序号
        :param start_col: 写入列序号
        :param date_list: 写入数据：列表
        :return: 返回写入的列对象
        """
        for row, label in enumerate(date_list):
            self.write(row + start_row, start_col, label, style)

        return self.sheet.col(start_col)

    def write_cols(self, start_row, start_col, data_lists, step=1, style=Style.default_style):
        """
        按列写入多列数据
        :param style:
        :param start_row:开始写入行序号
        :param start_col: 开始写入列序号
        :param data_lists: 列表嵌套列表数据
        :return: 返回列对象列表
        """
        col_obj = []
        for col_, data in enumerate(data_lists):
            if isinstance(data, list):
                self.write_col(start_row, start_col + col_ * step, data, style)
                col_obj.append(self.sheet.col(start_col + col_))
            else:
                msg = '数据列表不是嵌套列表数据，而是%s' % type(data)
                raise Exception(msg)

        return col_obj

    def change_style(self, row, col):
        """
        按行写入一行数据
        :param style:
        :param start_row:写入行序号
        :param start_col: 写入列序号
        :param date_list: 写入数据：列表
        :return: 返回行对象
        """
        # self.sheet.row(row).set_style(style)
        style = xlwt.easyxf("""
                        font:
                            bold on;
                         """)
        sheet_row = self.sheet.row(row)
        sheet_row.set_style(style)

    def diy_style(self, font_name, font_height, bold=True, horz=2):
        """
        创建单元格格式：（默认垂直居中）
        :param font_name: 字体名称
        :param font_height: 字体高度
        :param bold: 默认加粗
        :param horz: 水平对齐方式，默认水平居中：2，左对齐：1，右对齐：3
        :return: 返回设置好的格式
        """
        style = xlwt.XFStyle()
        # 字体设置
        font = style.font
        font.name = font_name
        font.height = font_height * 20
        font.bold = bold
        # 对齐方式
        alignment = style.alignment
        # 水平居中
        alignment.horz = horz
        # 垂直居中
        alignment.vert = 1

        return style

    def set_col_width(self, col_ro_cols, width):
        """
        设置单元格宽度
        :param col_ro_cols: 一个列序号，或列序号列表
        :param width: 列宽度
        :return: None
        """
        if isinstance(col_ro_cols, int):
            self.sheet.col(col_ro_cols).width = 256 * width
        else:
            for col_ in col_ro_cols:
                self.sheet.col(col_).width = 256 * width

    def set_row_height(self, row_ro_rows, height):
        """
        设置单元格高度
        :param row_ro_rows:行序号、或行序号列表
        :param height: 行高度
        :return: None
        """
        if isinstance(row_ro_rows, int):
            self.sheet.row(row_ro_rows).height_mismatch = True
            self.sheet.row(row_ro_rows).height = 20 * height
        else:
            for row_ in row_ro_rows:
                # 需先将单元格高度不自动匹配单元格内容高度打开，才能设置高度
                self.sheet.row(row_).height_mismatch = True
                self.sheet.row(row_).height = 20 * height

    def adjust_col_width(self, font_height):
        """
        设置自适应列宽
        :param font_height: 文字字体高度
        :return: None
        """

        # 获取字符串长度
        def string_len(string):
            length = 0
            for s in string:
                if s.isupper() or s.islower():
                    length += 2
                elif s.isspace():
                    length += 1
                else:
                    length += 3
            return length

        col_width = {}
        mul = font_height * 20
        for col, str_len in self.col_data.items():
            max_len = max([string_len(str(i)) for i in str_len])
            col_width[col] = max_len

        for col_, width_ in col_width.items():
            if width_ * mul <= 2350:
                self.sheet.col(col_).width = 2350
            else:
                self.sheet.col(col_).width = mul * width_

    def set_unite_height(self, height):
        """
        设置统一行高
        :param height:行高
        :return:None
        """
        rows = self.sheet.get_rows().keys()
        for row in rows:
            self.set_row_height(row, height)
