#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023/10/21 10:35 CST
# @Author  : Guangchen Jiang
# @Email   : guangchen98.jiang@gmail.com
# @File    : src/recovery/recovery.py
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
import os
import time
import tqdm
import sys
import yaml

import numpy as np

from distutils.util import strtobool
from functools import reduce

from dynamic_step import dynamic_simu

sys.path.append(r".")
sys.path.append(r"../..")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--log_level", type=str, default="INFO",
                        help="the logging level, include `CRITICAL`, `FATAL`, `ERROR`, `WARN`, `WARNING`, `INFO`, "
                             "`DEBUG`, `NOTSET`, default is `INFO`.")
    parser.add_argument("--multiprocess", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, multiprocess will be enabled by default")
    parser.add_argument("--process_num", type=int, default=0,
                        help="the number of processes used by the program, which defaults to `0`, is the same as "
                             "`$nproc`, and it must in [1, $nproc].")
    parser.add_argument("--config_path", type=str, default="./configs/recovery_config.yaml",
                        help="the path of the a yaml file for algorithm specific arguments, "
                             "default is './configs/recovery_config.yaml'.")
    parser.add_argument("--save_data_path", type=str, default="./res",
                        help="the path directory of the a json file for save data, default is './res'.")

    # Algorithm specific arguments
    parser.add_argument("--leading8idx", type=int, default=0,
                        help="(the index of leading eight norm)-1, e.g. use 0 for L1.")
    args = parser.parse_args()

    return args


def seconds2str(t: float) -> str:
    return "%d:%02d:%02d.%03d" % \
        reduce(lambda ll, b: divmod(ll[0], b) + ll[1:], [(t * 1000,), 1000, 60, 60])


if __name__ == '__main__':
    MAX_LEVEL = 1
    MIN_LEVEL = -MAX_LEVEL

    sys_args = parse_args()

    run_name = f"recovery__{sys_args.leading8idx}__{sys_args.seed}__{int(time.time())}"

    # TRY NOT TO MODIFY: seeding
    random.seed(sys_args.seed)
    np.random.seed(sys_args.seed)

    from src.my_utils import logger

    logger = logger.get_logger(name="simulation.recovery", level=logger.matching_logger_level(sys_args.log_level))
    logger.debug(f"run_name: {run_name}")

    with open(os.path.abspath(os.path.join(__file__, "../../..", sys_args.config_path)), 'r') as f:
        yaml_args = yaml.safe_load(f)
    logger.debug(f"sys_args: {sys_args}")
    logger.debug(f"yaml_args: {yaml_args}")

    simu_start_time = time.time()
    logger.info("Simulation start!")

    pbar = tqdm.tqdm(total=int(yaml_args["total round"]))
    pbar.set_description('Processing')

    avg_step, avg_defect = dynamic_simu(MIN_LEVEL, MAX_LEVEL, yaml_args, sys_args, logger, pbar)

    pbar.close()

    _save_data_path = os.path.abspath(os.path.join(__file__, "../../..", sys_args.save_data_path))

    if not os.path.exists(_save_data_path):
        os.makedirs(_save_data_path)
    with open(os.path.join(_save_data_path, f'{run_name}.json'), 'w') as f:
        json.dump({"avg_step": avg_step, "avg_defect": avg_defect}, f)
    logger.info(f"Result data saved to: {os.path.join(_save_data_path, f'{run_name}.json')}")

    logger.info(f"Simulation end! Time cost: {seconds2str(time.time() - simu_start_time)}")
