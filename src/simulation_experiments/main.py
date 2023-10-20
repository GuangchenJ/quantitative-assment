#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023/10/17 15:04 CST
# @Author  : Guangchen Jiang
# @Email   : guangchen98.jiang@gmail.com
# @File    : src/simulation_experiments/main.py
# @Software: PyCharm

import argparse
import random
import time
import tqdm
import sys
import yaml

import numpy as np

import scipy.linalg

from distutils.util import strtobool
from functools import reduce

from com_p_matrix import all_com_p_matrix
from stationary_dist_and_matrix import get_stationary_dist_and_matrix

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

    # Algorithm specific arguments
    # parser.add_argument("--expt_id", type=str, default="RPDG",
    #                     help="the id of the simulation experiments,including repeated prisoner's dilemma game (RPDG) "
    #                          "experiment, repeated public goods game (RPGG) experiment, stochastic prisoner's dilemma "
    #                          "game (SPDG) experiment.")
    # parser.add_argument("--total_rounds", type=int, default=10e6,
    #                     help="total timesteps of the experiments")
    parser.add_argument("--leading8idx", type=int, default=0,
                        help="(the index of leading eight norm)-1, e.g. use 0 for L1.")
    args = parser.parse_args()

    return args


def seconds2str(t: float) -> str:
    return "%d:%02d:%02d.%03d" % \
        reduce(lambda ll, b: divmod(ll[0], b) + ll[1:], [(t * 1000,), 1000, 60, 60])


if __name__ == '__main__':
    sys_args = parse_args()

    run_name = f"{sys_args.seed}__{int(time.time())}"

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

    pbar = tqdm.tqdm(total=2 * yaml_args["judgment threshold"] * (yaml_args["population size"] - 1))
    pbar.set_description('Processing')

    for max_level in range(1, yaml_args["judgment threshold"] + 1):
        allc_com_p = all_com_p_matrix(1, -max_level, max_level, yaml_args, sys_args, logger, pbar)
        alld_com_p = all_com_p_matrix(0, -max_level, max_level, yaml_args, sys_args, logger, pbar)
        stationary_distribution_result, fixed_matrix = get_stationary_dist_and_matrix(
            allc_com_p, alld_com_p, yaml_args, sys_args, logger
        )
        self_cooprate

    pbar.close()
    simu_end_time = time.time()
    logger.info(f"Simulation end! Time cost: {seconds2str(simu_start_time - simu_end_time)}")
