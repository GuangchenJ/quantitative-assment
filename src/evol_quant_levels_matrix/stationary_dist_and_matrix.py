#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023/10/20 20:14 CST
# @Author  : Guangchen Jiang
# @Email   : guangchen98.jiang@gmail.com
# @File    : src/evol_quant_levels_matrix/stationary_dist_and_matrix.py
# @Software: PyCharm

import argparse
import logging

import numpy as np

from deeptime.markov.tools.analysis import stationary_distribution


def payoff_leading8_vs_all(_s0_num: int,
                           _pop_size: int,
                           _b: float,
                           _c: float,
                           _all_com_p: np.ndarray) -> list[float, float]:
    """
    payoffs: leading eight norm vs ALLC
    """
    leading8_payoff = (float(_b) * (_all_com_p[_s0_num - 1][0] * (float(_s0_num) - 1.) +
                                    _all_com_p[_s0_num - 1][2] * (float(_pop_size) - float(_s0_num))
                                    ) / (float(_pop_size) - 1.)
                       - float(_c) * (
                               _all_com_p[_s0_num - 1][0] * (float(_s0_num) - 1.) +
                               _all_com_p[_s0_num - 1][1] * (float(_pop_size) - float(_s0_num))
                       ) / (float(_pop_size) - 1.))
    allx_payoff = (float(_b) * (_all_com_p[_s0_num - 1][3] * (float(_pop_size) - float(_s0_num) - 1.) +
                                _all_com_p[_s0_num - 1][1] * float(_s0_num)
                                ) / (float(_pop_size) - 1.)
                   - float(_c) * (
                           _all_com_p[_s0_num - 1][3] * (float(_pop_size) - float(_s0_num) - 1.) +
                           _all_com_p[_s0_num - 1][2] * float(_s0_num)
                   ) / (float(_pop_size) - 1.))
    return [leading8_payoff, allx_payoff]


def payoff_allc_vs_alld(_s0_num: int,
                        _pop_size: int,
                        _b: float,
                        _c: float) -> list[float, float]:
    """
    payoffs: ALLC vs ALLD
    """
    allc_payoff = float(_b) * (float(_s0_num) - 1.) / (float(_pop_size) - 1.) - float(_c)
    alld_payoff = float(_b) * float(_s0_num) / (float(_pop_size) - 1.)
    return [allc_payoff, alld_payoff]


def get_payoff(_idx: int,
               _s0_num: int,
               _pop_size: int,
               _b: float,
               _c: float,
               _all_com_p: np.ndarray = None) -> list[float, float]:
    """
    calling correct payoff function
    """
    if 1 == _idx or 2 == _idx:
        return payoff_leading8_vs_all(_s0_num, _pop_size, _b, _c, _all_com_p)
    elif 3 == _idx:
        return payoff_allc_vs_alld(_s0_num, _pop_size, _b, _c)
    else:
        raise "Warning, wrong index!"


def get_fixed_prob(_idx: int,
                   _args: dict,
                   _sys_args: argparse.Namespace,
                   _all_com_p: np.ndarray = None) -> tuple[float, float]:
    """
    calculating fixation probability of strategy M into strategy R from payoffs
    Ref: Eqn 2. in paper
    """
    sum_q = 0.
    sum_q_invade = 0.
    remain = 1.
    invade = 1.
    for _k in range(1, _args["population size"]):
        pi_m, pi_r = get_payoff(
            _idx, _k, _args["population size"],
            _args["reward matrix"]["benefit of cooperation"],
            _args["reward matrix"]["cost of cooperation"],
            _all_com_p
        )
        remain *= np.exp(-float(_args["selection strength"]) * (pi_m - pi_r))
        sum_q += remain
        pi_m_invade, pi_r_invade = get_payoff(
            _idx, _args["population size"] - _k, _args["population size"],
            _args["reward matrix"]["benefit of cooperation"],
            _args["reward matrix"]["cost of cooperation"],
            _all_com_p
        )
        invade *= np.exp(-float(_args["selection strength"]) * (pi_r_invade - pi_m_invade))
        sum_q_invade += invade
    xf = 1. / (1. + sum_q)
    xf_invade = 1. / (1. + sum_q_invade)
    return xf, xf_invade


def get_stationary_dist_and_matrix(_allc_com_p: np.ndarray,
                                   _alld_com_p: np.ndarray,
                                   _args: dict,
                                   _sys_args: argparse.Namespace,
                                   _logger: logging.Logger):
    """
    calculating selection-mutation equilibrium
    """
    fixed_prob_matrix = np.zeros((3, 3)).astype(float)
    lc = get_fixed_prob(1, _args, _sys_args, _allc_com_p)
    ld = get_fixed_prob(2, _args, _sys_args, _alld_com_p)
    cd = get_fixed_prob(3, _args, _sys_args)
    _logger.debug(f"leading8 & allc: {lc}")
    _logger.debug(f"leading8 & alld: {ld}")
    _logger.debug(f"allc & alld: {ld}")
    # fixation of Leading8 into AllC
    fixed_prob_matrix[0][1] = lc[1]
    # fixation of Leading8 into AllD
    fixed_prob_matrix[0][2] = ld[1]
    # fixation of AllC into AllD
    fixed_prob_matrix[1][2] = cd[1]
    fixed_prob_matrix[1][0] = lc[0]
    fixed_prob_matrix[2][0] = ld[0]
    fixed_prob_matrix[2][1] = cd[0]
    fixed_prob_matrix /= 2.
    for _i in range(3):
        row_sum = 0.
        for _j in range(3):
            if _i != _j:
                row_sum += fixed_prob_matrix[_i][_j]
        fixed_prob_matrix[_i][_i] = 1. - row_sum
    mu = stationary_distribution(fixed_prob_matrix)
    return mu, fixed_prob_matrix * 2.
