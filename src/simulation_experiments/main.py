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
import sys
import yaml

import numpy as np

from distutils.util import strtobool

sys.path.append(r"../../..")

g0 = np.int8(0)
g1 = np.int8(1)
a0 = np.int8(0)
a1 = np.int8(1)
# ref: Ohtsuki, Hisashi, and Yoh Iwasa. "How should we define goodness?—reputation dynamics in indirect reciprocity."
# Journal of theoretical biology 231.1 (2004): 107-120.
# dc11, dc10, dc01, dc00, dd11, dd10, dd01, dd00
L8ASSESSMENTS = np.array([
    [g1, g1, g1, g1, g0, g1, g0, g0],
    [g1, g0, g1, g1, g0, g1, g0, g0],
    [g1, g1, g1, g1, g0, g1, g0, g1],
    [g1, g1, g1, g0, g0, g1, g0, g1],
    [g1, g0, g1, g1, g0, g1, g0, g1],
    [g1, g0, g1, g0, g0, g1, g0, g1],
    [g1, g1, g1, g0, g0, g1, g0, g0],
    [g1, g0, g1, g0, g0, g1, g0, g0]
], dtype=np.int8)
# p11, p10, p01, p00
L8ACTIONS = np.array([
    [a1, a0, a1, a1],
    [a1, a0, a1, a1],
    [a1, a0, a1, a0],
    [a1, a0, a1, a0],
    [a1, a0, a1, a0],
    [a1, a0, a1, a0],
    [a1, a0, a1, a0],
    [a1, a0, a1, a0]
], dtype=np.int8)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--log_level", type=str, default="debug",
                        help="the logging level, include 'CRITICAL', 'FATAL', 'ERROR', "
                             "'WARN', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'.")
    parser.add_argument("--multiprocess", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, multiprocess will be enabled by default")
    parser.add_argument("--process_num", type=int, default=-1,
                        help="the number of processes used by the program, which defaults to `-1`, is the same as "
                             "$nproc, and it must < $nproc.")

    # Algorithm specific arguments
    parser.add_argument("--expt_id", type=str, default="RPDG",
                        help="the id of the simulation experiments,including repeated prisoner's dilemma game (RPDG) "
                             "experiment, repeated public goods game (RPGG) experiment, stochastic prisoner's dilemma "
                             "game (SPDG) experiment.")
    # parser.add_argument("--total_rounds", type=int, default=10e6,
    #                     help="total timesteps of the experiments")
    parser.add_argument("--leading8idx", type=int, default=0,
                        help="(the index of leading eight norm)-1, e.g. use 0 for L1.")
    args = parser.parse_args()

    return args


def score2rep(_score: int, _threshold: int) -> np.int8:
    return np.int8(1) if _score >= _threshold else np.int8(0)


def score_add(_number: int, _sign: int, _min_level: int, _max_level: int) -> np.int8:
    """
    outputting new reputation score after an assessment
    """
    return np.clip(_number + _sign, _min_level, _max_level)


def cooperates(_k: int, _idx: int, _min_level: int, _max_level: int, _args: dict, _sys_args: argparse.Namespace):
    """
    Calculating cooperation rates of two strategies in a population: $_k players use a leading eight norm
    and $(args["population size"] -_k) players using ALLC ($_idx=1) or ALLD ($_idx=0). We simulate the
    reputation dynamics for $args["total round"] timesteps and calculate the average cooperation frequency
    for each of the 4 strategy combinations.
    """
    strategies_num = np.int8(2)
    assessments = np.array([L8ASSESSMENTS[_sys_args.leading8idx], np.ones(8) if _idx else np.zeros(8)]).astype(np.int8)
    actions = np.array([L8ACTIONS[_sys_args.leading8idx], np.ones(4) if _idx else np.zeros(4)]).astype(np.int8)
    strategies_array = np.concatenate((np.zeros(_k), np.ones(_args["population size"] - _k))).astype(np.int8)
    image_matrix = np.zeros((_args["population size"], _args["population size"]), dtype=np.int8)
    cooperate_matrix = np.zeros((_args["population size"], _args["population size"]), dtype=np.int64)
    interact_matrix = np.zeros((_args["population size"], _args["population size"]), dtype=np.int64)
    avg_image_matrix = image_matrix.astype(np.float64)
    for _i in range(_args["total round"]):
        # donor acts according to his reputation repository (his row of image matrix)
        donor, receiver = np.array(random.sample(range(0, _args["population size"] - 1), 2), np.int8)
        donor_act = actions[strategies_array[donor]]
        reputation_donor_score = image_matrix[donor][donor]
        reputation_receiver_score = image_matrix[donor][receiver]
        reputation_donor = score2rep(reputation_donor_score, _args["threshold"])
        reputation_receiver = score2rep(reputation_receiver_score, _args["threshold"])
        action = donor_act[(np.int8(1) - reputation_donor) * np.int8(2) + (np.int8(1) - reputation_receiver)]
        # updating observers opinions
        # todo: multiprocess
        for _j in np.arange(_args["population size"], dtype=np.int8):
            if (_j == donor) or (_j == receiver) or (random.random() < _args["observation probability"]):
                obs_assessment = assessments[strategies_array[_j]]
                rep_donor_obs = score2rep(image_matrix[_j][donor], _args["threshold"])
                rep_receiver_obs = score2rep(image_matrix[_j][receiver], _args["threshold"])
                obs_asmt = obs_assessment[
                    (1 - action) * 4 + (1 - rep_donor_obs) * 2 + (1 - rep_receiver_obs)
                    if random.random() < (1 - _args["error rate"]) else
                    action * 4 + (1 - rep_donor_obs) * 2 + (1 - rep_receiver_obs)
                ]
                image_matrix[_j][donor] = np.clip(
                    image_matrix[_j][donor] + 1 if obs_asmt else -1,
                    _min_level, _max_level
                )
        # after burn in time, start recording cooperative actions and total number of interactions
        if _i > _args["burn in time"]:
            cooperate_matrix[donor][receiver] += action
            interact_matrix[donor][receiver] += 1
            if _i == _args["burn in time"] + 1:
                avg_image_matrix = image_matrix.astype(np.float64)
            else:
                avg_image_matrix = np.float64(1. * (_i - _args["burn in time"] - 1.)) / np.float64(
                    _i - _args["burn in time"]) * avg_image_matrix + np.float64(1.) / np.float64(
                    _i - _args["burn in time"]) * image_matrix.astype(np.float64)

    cooperate_matrix_means = np.zeros((strategies_num, strategies_num), dtype=np.float64)
    strategies_matrix_means = np.zeros((strategies_num, strategies_num), dtype=np.float64)

    # todo: multiprocess
    for _i in np.arange(strategies_num, dtype=np.int8):
        for _j in np.arange(strategies_num, dtype=np.int8):
            cooperate_matrix_means[_i, _j] = np.sum(np.sum(
                cooperate_matrix[_i == strategies_array],
                where=[_j == strategies_array], axis=1
            )) / np.sum(np.sum(
                interact_matrix[_i == strategies_array],
                where=[_j == strategies_array],
                axis=1
            ))
            if _i != _j:
                strategies_matrix_means[_i, _j] = np.mean(np.mean(
                    avg_image_matrix[_i == strategies_array],
                    where=[_j == strategies_array], axis=1
                ))
            elif (strategy_players_i := np.nonzero(strategies_array == _i)[0]).shape[0] > 1:
                strategy_players_j = np.where(strategies_array == _j)[0]
                strategies_matrix_means[_i, _j] = np.mean([
                    avg_image_matrix[x, y] for x in strategy_players_i for y in strategy_players_j if x != y
                ])

    # return average pairwise cooperation rate of the four possible strategy combinations
    # for this population composition
    return [
        cooperate_matrix_means[0, 0], cooperate_matrix_means[0, 1],
        cooperate_matrix_means[1, 0], cooperate_matrix_means[1, 1]
    ]


def all_com_p_matrix(_idx: int, _min_level: int, _max_level: int, _args: dict, _sys_args: argparse.Namespace):
    coop_com_p = np.zeros((_args["population size"] - 1, 4))
    for _k in range(1, _args["population size"]):
        coop_com_p[_k - 1] = cooperates(_k, _idx, _max_level, _min_level, _args, _sys_args)
    return coop_com_p


if __name__ == '__main__':
    sys_args = parse_args()

    run_name = f"{sys_args.expt_id}__{sys_args.seed}__{int(time.time())}"

    # TRY NOT TO MODIFY: seeding
    random.seed(sys_args.seed)
    np.random.seed(sys_args.seed)

    from src.my_utils import logger

    logger = logger.get_logger(name="simulation.evol_quant", level=logger.matching_logger_level(sys_args.log_level))
    logger.debug(f"run_name: {run_name}")

    with open('../../config.yaml', 'r') as f:
        yaml_args = yaml.safe_load(f)
    logger.debug(f"yaml_args: {yaml_args}")

    for max_level in range(1):
        min_level = -max_level
        allc_com_p = all_com_p_matrix(1, min_level, max_level, yaml_args, sys_args)
        alld_com_p = all_com_p_matrix(1, min_level, max_level, yaml_args, sys_args)

    logger.debug(type(yaml_args))
    logger.debug(type(sys_args))
