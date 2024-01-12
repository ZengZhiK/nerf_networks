# -*- coding: utf-8 -*-
"""
author:     tianhe
time  :     2023-01-18 11:42
"""
import os
import logging
from logging.handlers import RotatingFileHandler


def init(log_dir):
    # 日志文件大小最大为1MB，最多备份1个
    rfh = RotatingFileHandler(filename=os.path.join(log_dir, 'logs.log'),
                              maxBytes=100 * 1024 * 1024, backupCount=1, encoding='utf-8')

    # 创建一个控制屏幕输出的屏幕操作符
    sh = logging.StreamHandler()

    log_fmt = '%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s - %(message)s'
    date_fmt = '%Y-%m-%d %H:%M:%S'

    # # logger对象来绑定：文件操作符，屏幕操作符
    logging.basicConfig(handlers=[rfh, sh], level=logging.INFO, format=log_fmt, datefmt=date_fmt)
