#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import logging.config
from os import path, chdir, getcwd


def log_config():
    # log_file_path = path.join(, 'logging.conf')
    origin = getcwd()
    chdir(path.dirname(path.abspath(__file__)))
    logging.config.fileConfig('logging.conf')
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    chdir(origin)


if __name__ == '__main__':
    log_config()
    # 创建logger
    logger = logging.getLogger('FSTool')
    # 通过logger记录日志
    logger.debug('debug message')
    logger.info('info message')
    logger.warning('warn message')
    logger.error('error message')
    logger.critical('critical message')
