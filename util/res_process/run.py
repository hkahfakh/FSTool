#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用于运行实验数据分析
运行 res_anls.formatread
"""
from util.res_process.res_anls import FormatRead

if __name__ == '__main__':
    fr = FormatRead('tt.xls', "test")
    # fr.open_sheet('warpAr10P')
    fr.run_analysis()
