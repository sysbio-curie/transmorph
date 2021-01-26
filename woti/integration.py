#!/usr/bin/env python3

import numpy as np
import ot

from typing import Callable

from .density import normal_kernel_weights


def ot_transform(
    xs: np.ndarray,
    yt: np.ndarray,
    ws: np.ndarray,
    vs: np.ndarray,
    max_iter: int = 1e7,
    cost_function: Callable = ot.dist,
):
    """"""
    n, m = len(xs), len(yt)
    assert n >= 0, "Source matrix cannot be empty."
    assert m >= 0, "Reference matrix cannot be empty."
    assert n == len(ws), "Weights size does not coincidate with source points."
    assert m == len(vs), "Weights size does not coincidate with reference points."

    # Computing weights
    M = cost_function(xs, yt)
    assert M.shape == (n, m), "Cost function should return a pairwise (n,m) matrix."
    M /= M.max()

    # Normalization of weights
    assert abs(np.sum(ws) - 1) < 1e-9 and all(
        ws >= 0
    ), "Source weights must be in the probability simplex."
    assert abs(np.sum(vs) - 1) < 1e-9 and all(
        vs >= 0
    ), "Reference weights must be in the probability simplex."
    transport_plan = ot.emd(ws, vs, M, numItermax=max_iter)

    return np.diag(1 / ws) @ transport_plan @ yt


def bot_transform(
    xs: np.ndarray,
    yt: np.ndarray,
    max_iter: int = 1e7,
    cost_function: Callable = ot.dist,
):
    """"""
    # Computing equal weights
    n, m = len(xs), len(yt)
    assert n >= 0, "Source matrix cannot be empty."
    assert m >= 0, "Reference matrix cannot be empty."
    ws, vs = np.array([1 / n] * n), np.array([1 / m] * m)

    # Adjusting for approximation
    ws /= np.sum(ws)
    vs /= np.sum(vs)

    return ot_transform(xs, yt, ws, vs, max_iter=max_iter, cost_function=cost_function)


def wot_transform(
    xs: np.ndarray,
    yt: np.ndarray,
    max_iter: int = 1e7,
    cost_function: Callable = ot.dist,
    scale: float = 0.1,
    scale_ref: float = -1,
):

    if scale_ref == -1:
        scale_ref = scale

    n, m = len(xs), len(yt)
    assert n >= 0, "Source matrix cannot be empty."
    assert m >= 0, "Reference matrix cannot be empty."
    ws, vs = normal_kernel_weights(xs, scale=scale), normal_kernel_weights(
        yt, scale=scale_ref
    )

    return ot_transform(xs, yt, ws, vs, max_iter=max_iter, cost_function=cost_function)
