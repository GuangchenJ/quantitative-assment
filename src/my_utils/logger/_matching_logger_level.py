#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023/10/17 15:40 CST
# @Author  : Guangchen Jiang
# @Email   : guangchen98.jiang@gmail.com
# @File    : src/my_utils/logger/matching_logger_level.py
# @Software: PyCharm

import logging

_info_name_list = ["CRITICAL", "FATAL", "ERROR", "WARN", "WARNING", "INFO", "DEBUG", "NOTSET"]


def matching_logger_level(level: str) -> int:
    assert level.upper() in _info_name_list, ("Logger's level should be set to one of the following "
                                              "values: 'CRITICAL', 'FATAL', 'ERROR', 'WARN', 'WARNING', "
                                              "'INFO', 'DEBUG', 'NOTSET'.")
    matching_dict = lambda x: {
        "NOTSET" == level.upper(): logging.NOTSET,
        "DEBUG" == level.upper(): logging.DEBUG,
        "INFO" == level.upper(): logging.INFO,
        "WARN" == level.upper(): logging.WARN,
        "WARNING" == level.upper(): logging.WARNING,
        "ERROR" == level.upper(): logging.ERROR,
        "FATAL" == level.upper(): logging.FATAL,
        "CRITICAL" == level.upper(): logging.CRITICAL,
    }
    return matching_dict(level)[True]


if __name__ == '__main__':
    print(matching_logger_level("notset"))
