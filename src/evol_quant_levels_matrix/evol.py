#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023/10/17 15:04 CST
# @Author  : Guangchen Jiang
# @Email   : guangchen98.jiang@gmail.com
# @File    : src/evol_quant_levels_matrix/evol.py
# @Software: PyCharm

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

from com_p_matrix import all_com_p_matrix, cooperates_homogeneous
from stationary_dist_and_matrix import get_stationary_dist_and_matrix

sys.path.append(r".")
sys.path.append(r"../..")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0,
                        help="seed of the experiment")
    parser.add_argument("--log_level", type=str, default="INFO",
                        help="the logging level, include `CRITICAL`, `FATAL`, `ERROR`, `WARN`, `WARNING`, `INFO`, "
                             "`DEBUG`, `NOTSET`, default is `INFO`.")
    parser.add_argument("--multiprocess", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, multiprocess will be enabled by default")
    parser.add_argument("--process_num", type=int, default=0,
                        help="the number of processes used by the program, which defaults to `0`, is the same as "
                             "`$nproc`, and it must in [1, $nproc].")
    parser.add_argument("--config_path", type=str, default="./config.yaml",
                        help="the path of the a yaml file for algorithm specific arguments, "
                             "default is './config.yaml'.")
    parser.add_argument("--save_data_path", type=str, default="../../res",
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
    sys_args = parse_args()

    run_name = f"evol__{sys_args.leading8idx}__{sys_args.seed}__{int(time.time())}"

    # TRY NOT TO MODIFY: seeding
    random.seed(sys_args.seed)
    np.random.seed(sys_args.seed)

    from src.my_utils import logger

    logger = logger.get_logger(name="simulation.evol_quant", level=logger.matching_logger_level(sys_args.log_level))
    logger.debug(f"run_name: {run_name}")

    with open(sys_args.config_path, 'r') as f:
        yaml_args = yaml.safe_load(f)
    logger.debug(f"sys_args: {sys_args}")
    logger.debug(f"yaml_args: {yaml_args}")

    simu_start_time = time.time()
    logger.info("Simulation start!")

    pbar = tqdm.tqdm(total=2 * yaml_args["judgment threshold"] * yaml_args["population size"])
    pbar.set_description('Processing')

    res_list = list()

    for max_level in range(1, yaml_args["judgment threshold"] + 1):
        allc_com_p = all_com_p_matrix(1, -max_level, max_level, yaml_args, sys_args, logger, pbar)
        alld_com_p = all_com_p_matrix(0, -max_level, max_level, yaml_args, sys_args, logger, pbar)

        stationary_distribution_result, fixed_matrix = get_stationary_dist_and_matrix(
            allc_com_p, alld_com_p, yaml_args, sys_args, logger
        )
        self_cooperate = cooperates_homogeneous(-max_level, max_level, yaml_args, sys_args)
        cooperate_total = stationary_distribution_result[0] * self_cooperate + stationary_distribution_result[1]
        res_list.append({
            "min_level": -max_level,
            "max_level": max_level,
            "stationary_distribution_result": stationary_distribution_result.tolist(),
            "self_cooperate": self_cooperate,
            "cooperate_total": cooperate_total,
            "fixed_matrix": fixed_matrix.tolist()
        })
        pbar.update(1)

    pbar.close()

    if not os.path.exists(sys_args.save_data_path):
        os.makedirs(sys_args.save_data_path)
    with open(os.path.join(sys_args.save_data_path, f"{run_name}.json"), 'w') as f:
        json.dump(res_list, f)
    logger.info(f"Result data saved to: {os.path.join(sys_args.save_data_path, f"{run_name}.json")}")

    logger.info(f"Simulation end! Time cost: {seconds2str(time.time() - simu_start_time)}")
