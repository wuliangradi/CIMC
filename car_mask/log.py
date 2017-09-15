#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    the path of AI
    Contact:wuliangwuwu@126.com
"""
import logging
import logging.handlers
import os


def init_log(logfile=None, log_name="", level=logging.INFO):
    """initialize log module
    :param logfile:
    :param log_name:
    :param level:
    :return:
    """
    logger = logging.getLogger(log_name)
    if not logger.handlers:
        logger = logging.getLogger(log_name)
        logger.propagate = False
        logger.setLevel(level)
        format_ = "%(levelname)s: %(asctime)s: %(filename)s:%(lineno)d **  %(message)s"
        date_fmt = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(format_, datefmt=date_fmt)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        if logfile is not None:
            dir_logfile = os.path.dirname(logfile)
            if not os.path.isdir(dir_logfile):
                os.makedirs(dir_logfile)
            handler = logging.FileHandler(logfile)
            logger.addHandler(handler)
            handler.setFormatter(formatter)
    return logger
