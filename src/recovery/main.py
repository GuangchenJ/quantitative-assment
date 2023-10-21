#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023/10/21 10:35 CST
# @Author  : Guangchen Jiang
# @Email   : guangchen98.jiang@gmail.com
# @File    : src/recovery/main.py
# @Software: PyCharm

"""
Simulating the recovery process from a single disagreement in a homogeneous population of leading eight players using
quantitative assessment with $R=3$. Players start with reputation scores `0`, except for P1, who thinks P2 is bad and
assigns a score of `-1`. This script works for L1, L2, L3, L5 (norms with only one absorbing state). Output is a file
containing steps to recovery and number of defections until recovery.
"""
import argparse
import json
import random
import time
import tqdm
import sys
import yaml

import numpy as np

import scipy.linalg

from distutils.util import strtobool
from functools import reduce

sys.path.append(r".")
sys.path.append(r"../..")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--log_level", type=str, default="INFO",
                        help="the logging level, include `CRITICAL`, `FATAL`, `ERROR`, `WARN`, `WARNING`, `INFO`, "
                             "`DEBUG`, `NOTSET`, default is `INFO`.")
    parser.add_argument("--multiprocess", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, multiprocess will be enabled by default")
    parser.add_argument("--process_num", type=int, default=0,
                        help="the number of processes used by the program, which defaults to `0`, is the same as "
                             "`$nproc`, and it must in [1, $nproc].")
    parser.add_argument("--config_path", type=str, default="../../config.yaml",
                        help="the path of the a yaml file for algorithm specific arguments, "
                             "default is '../../config.yaml'.")
    parser.add_argument("--save_data_path", type=str, default="./data.json",
                        help="the path of the a json file for save data, default is './data.json'.")

    # Algorithm specific arguments
    parser.add_argument("--leading8idx", type=int, default=0,
                        help="(the index of leading eight norm)-1, e.g. use 0 for L1.")
    args = parser.parse_args()

    return args


def seconds2str(t: float) -> str:
    return "%d:%02d:%02d.%03d" % \
        reduce(lambda ll, b: divmod(ll[0], b) + ll[1:], [(t * 1000,), 1000, 60, 60])
