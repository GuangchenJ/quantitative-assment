#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023/10/19 15:01 CST
# @Author  : Guangchen Jiang
# @Email   : guangchen98.jiang@gmail.com
# @File    : src/simulation_experiments/test3.py
# @Software: PyCharm

# from scipy.sparse import issparse as _issparse
# from scipy.sparse import csr_matrix as _csr_matrix

def is_transition_matrix(T, tol=1e-12):
    r"""Check if the given matrix is a transition matrix.

    Parameters
    ----------
    T : (M, M) ndarray or scipy.sparse matrix
        Matrix to check
    tol : float (optional)
        Floating point tolerance to check with

    Returns
    -------
    is_transition_matrix : bool
        True, if T is a valid transition matrix, False otherwise

    Notes
    -----
    A valid transition matrix :math:`P=(p_{ij})` has non-negative
    elements, :math:`p_{ij} \geq 0`, and elements of each row sum up
    to one, :math:`\sum_j p_{ij} = 1`. Matrices wit this property are
    also called stochastic matrices.

    Examples
    --------
    >>> import numpy as np

    >>> A = np.array([[0.4, 0.5, 0.3], [0.2, 0.4, 0.4], [-1, 1, 1]])
    >>> is_transition_matrix(A)
    False

    >>> T = np.array([[0.9, 0.1, 0.0], [0.5, 0.0, 0.5], [0.0, 0.1, 0.9]])
    >>> is_transition_matrix(T)
    True

    """
    T = _types.ensure_ndarray_or_sparse(T, ndim=2, uniform=True, kind='numeric')
    if _issparse(T):
        return sparse.assessment.is_transition_matrix(T, tol)
    else:
        return dense.assessment.is_transition_matrix(T, tol)


def stationary_distribution(T):
    r"""Compute stationary distribution of stochastic matrix T.

    Parameters
    ----------
    T : (M, M) ndarray or scipy.sparse matrix
        Transition matrix

    Returns
    -------
    mu : (M,) ndarray
        Vector of stationary probabilities.

    Notes
    -----
    The stationary distribution :math:`\mu` is the left eigenvector
    corresponding to the non-degenerate eigenvalue :math:`\lambda=1`,

    .. math:: \mu^T T =\mu^T.

    Examples
    --------

    >>> import numpy as np\

    >>> T = np.array([[0.9, 0.1, 0.0], [0.4, 0.2, 0.4], [0.0, 0.1, 0.9]])
    >>> mu = stationary_distribution(T)
    >>> mu
    array([ 0.44444444, 0.11111111, 0.44444444])

    """
    # is this a transition matrix?
    if not is_transition_matrix(T):
        raise ValueError("Input matrix is not a transition matrix."
                         "Cannot compute stationary distribution")
    # is the stationary distribution unique?
    if not is_connected(T, directed=False):
        raise ValueError("Input matrix is not weakly connected. "
                         "Therefore it has no unique stationary "
                         "distribution. Separate disconnected components "
                         "and handle them separately")
    # we're good to go...
    if _issparse(T):
        mu = sparse.stationary_vector.stationary_distribution(T)
    else:
        mu = dense.stationary_vector.stationary_distribution(T)
    return mu
