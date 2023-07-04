#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from functools import wraps
from datetime import datetime


def logged(level):
    """
    Add logging to a function. level is the logging
    level, name is the logger name, and message is the
    log message. If name and message aren't specified,
    they default to the function's module and name.
    """

    def decorate(func):
        log = logging.getLogger('FSTool')

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            r = func(*args, **kwargs)
            end_time = datetime.now()

            log.log(level, [func.__name__, r, (end_time - start_time).total_seconds()])
            return r

        return wrapper

    return decorate


def save_di(level):
    """
    用在fs_all中的特征选择中
    装饰器 保存花费时间
    """

    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            selected_f = func(*args, **kwargs)
            end_time = datetime.now()
            expense = (end_time - start_time).total_seconds()
            if kwargs.get("di") is not None:
                kwargs['di'].update_item_fs(kwargs['algo_name'], selected_f, expense)
            print("work")
            return selected_f

        return wrapper

    return decorate


def working_dir():
    """
    临时转换工作路径到根目录
    :return:
    """
    import os

    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cwd = os.getcwd()  # 获取工作路径
            cur_path = os.path.abspath(os.path.dirname(__file__))  # 获取当前文件的目录  这个是getData.py的路径
            root_path = cur_path[:cur_path.find("FSTool\\") + len("FSTool\\")]  # 获取根目录
            if os.path.samefile(root_path, cwd) is not True:  # 如果当前路径不是根目录
                os.chdir(root_path)
                r = func(*args, **kwargs)
                os.chdir(cwd)
            else:
                r = func(*args, **kwargs)
            return r

        return wrapper

    return decorate
