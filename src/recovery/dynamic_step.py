#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023/10/21 16:16 CST
# @Author  : Guangchen Jiang
# @Email   : guangchen98.jiang@gmail.com
# @File    : src/recovery/dynamic_step.py
# @Software: PyCharm

import argparse
import logging
import random
import tqdm

import numpy as np

from multiprocessing import cpu_count, shared_memory
from multiprocessing import Process

g0 = 0
g1 = 1
a0 = 0
a1 = 1
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
]).astype(int)
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
]).astype(int)


def score2rep(_score: int, _threshold: int) -> int:
    return 1 if _score >= _threshold else 0


def is_all_good(_matrix: np.ndarray) -> bool:
    """recovery state: no player is assigned a negative reputation score"""
    _mat_flat = _matrix.flatten()
    for _m in _mat_flat:
        if _m < 0:
            return False
    return True


def is_all_bad(_matrix: np.ndarray):
    # Is Player1 thinks everyone except him is bad and all others think P1 is bad
    return np.all(_matrix[1:, :1] == -1) and np.all(_matrix[:1, 1:] == -1)


def dynamic_step(_min_level: int,
                 _max_level: int,
                 _actions: np.ndarray,
                 _strategies_array: np.ndarray,
                 _assessments: np.ndarray,
                 _args: dict,
                 _sys_args: argparse.Namespace) -> tuple[float, float]:
    # image matrix
    image_matrix = np.zeros((_args["population size"], _args["population size"])).astype(int)
    image_matrix[0][1] = -1
    steps = 0.
    defections = 0.

    while not is_all_good(image_matrix) and not is_all_bad(image_matrix) and steps < 1000:
        donor, receiver = random.sample(range(_args["population size"]), 2)
        donor_act = _actions[_strategies_array[donor]]
        reputation_donor_score = image_matrix[donor][donor]
        reputation_receiver_score = image_matrix[donor][receiver]
        reputation_donor = score2rep(reputation_donor_score, _args["score threshold"])
        reputation_receiver = score2rep(reputation_receiver_score, _args["score threshold"])
        action = donor_act[(1 - reputation_donor) * 2 + (1 - reputation_receiver)]
        for _j in range(_args["population size"]):
            if (_j == donor) or (_j == receiver) or (random.random() < _args["observation probability"]):
                obs_assessment = _assessments[_strategies_array[_j]]
                rep_donor_obs = score2rep(image_matrix[_j][donor], _args["score threshold"])
                rep_receiver_obs = score2rep(image_matrix[_j][receiver], _args["score threshold"])
                obs_asmt = obs_assessment[
                    (1 - action) * 4 + (1 - rep_donor_obs) * 2 + (1 - rep_receiver_obs)
                    if random.random() < (1 - _args["error rate"]) else
                    action * 4 + (1 - rep_donor_obs) * 2 + (1 - rep_receiver_obs)
                ]
                image_matrix[_j][donor] = np.clip(
                    image_matrix[_j][donor] + 1 if obs_asmt else -1,
                    _min_level, _max_level
                )
        steps += 1.
        defections += 1. - float(action)

    if is_all_bad(image_matrix):
        return -1, -1
    else:
        return steps, defections


def mp_dynamic_step(_it: int,
                    _min_level: int,
                    _max_level: int,
                    _actions: np.ndarray,
                    _strategies_array: np.ndarray,
                    _assessments: np.ndarray,
                    _args: dict,
                    _sys_args: argparse.Namespace,
                    _shm_sc_name: str,
                    _shm_sc_shape: tuple,
                    _sc_dtype: np.dtype,
                    _shm_dc_name: str,
                    _shm_dc_shape: tuple,
                    _dc_dtype: np.dtype) -> None:
    # deal with shared memory
    existing_shm_sc = shared_memory.SharedMemory(name=_shm_sc_name)
    step_ctr = np.ndarray(_shm_sc_shape, dtype=_sc_dtype, buffer=existing_shm_sc.buf)
    existing_shm_dc = shared_memory.SharedMemory(name=_shm_dc_name)
    defection_ctr = np.ndarray(_shm_dc_shape, dtype=_dc_dtype, buffer=existing_shm_dc.buf)

    step_ctr[_it], defection_ctr[_it] = dynamic_step(
        _min_level, _max_level, _actions,
        _strategies_array, _assessments, _args, _sys_args
    )
    existing_shm_sc.close()
    existing_shm_dc.close()
    return None


def dynamic_simu(_min_level: int,
                 _max_level: int,
                 _args: dict,
                 _sys_args: argparse.Namespace,
                 _logger: logging.Logger,
                 pbar: tqdm.tqdm = None):
    assessments = np.array([np.zeros(8), np.ones(8), L8ASSESSMENTS[_sys_args.leading8idx]]).astype(int)
    actions = np.array([np.zeros(4), np.ones(4), L8ACTIONS[_sys_args.leading8idx]]).astype(int)
    strategies_array = np.concatenate(
        (np.zeros(0), np.ones(0), np.ones(_args["population size"]) * 2)
    ).astype(int)
    step_ctr = np.zeros(_args["total round"]).astype(float)
    defection_ctr = np.ones(_args["total round"]).astype(float)
    if _sys_args.multiprocess:
        shm_s = shared_memory.SharedMemory(create=True, size=step_ctr.nbytes)
        shm_d = shared_memory.SharedMemory(create=True, size=defection_ctr.nbytes)
        # map shared memory `multiprocessing.shared_memory.SharedMemory` to `numpy.ndarray`.
        shm_step_ctr = np.ndarray(step_ctr.shape, dtype=step_ctr.dtype, buffer=shm_s.buf)
        shm_defection_ctr = np.ndarray(defection_ctr.shape, dtype=defection_ctr.dtype, buffer=shm_d.buf)
        if _sys_args.process_num < 0 or _sys_args.process_num > cpu_count():
            p_step = cpu_count()
            _logger.warning("Please check the `process_num` in hyperparameters, the `process_num` must be set to "
                            "[1, $nproc], or use the default parameter `0`, which is now set to `$nproc`.")
        else:
            p_step = cpu_count() if 0 == _sys_args.process_num else _sys_args.process_num

        for _it in range(0, _args["total round"], p_step):
            p_list = list()
            chunk = range(_args["total round"])[_it:_it + p_step]
            for _precess_i in chunk:
                p_list.append(Process(target=mp_dynamic_step, args=(
                    _precess_i, _min_level, _max_level,
                    actions, strategies_array, assessments,
                    _args, _sys_args,
                    shm_s.name, step_ctr.shape, step_ctr.dtype,
                    shm_d.name, defection_ctr.shape, defection_ctr.dtype
                )))
            for _p in p_list:
                _p.start()
            for _p in p_list:
                _p.join()
            pbar.update(len(p_list))

        # update shared memory
        step_ctr[:] = np.copy(shm_step_ctr)
        defection_ctr[:] = np.copy(shm_defection_ctr)
        del shm_step_ctr
        del shm_defection_ctr

        shm_d.close()
        shm_s.close()
        shm_d.unlink()
        shm_s.close()
    else:
        for _it in range(_args["total round"]):
            step_ctr[_it], defection_ctr[_it] = dynamic_step(
                _min_level, _max_level, actions,
                strategies_array, assessments, _args, _sys_args
            )
            pbar.update(1)

    # conditioned on recovery being reached
    step_ctr_new = np.delete(step_ctr, np.where(-1 == step_ctr))
    defect_ctr_new = np.delete(defection_ctr, np.where(-1 == defection_ctr))
    return np.average(step_ctr_new), np.average(defect_ctr_new)
