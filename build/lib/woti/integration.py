#!/usr/bin/env python3

import numpy as np
import ot


def OT_transform(Xs, Yt):  # Raw
    a_s, a_t = np.array([1 / len(Xs)] * len(Xs)), np.array([1 / len(Yt)] * len(Yt))
    M = ot.dist(Xs, Yt)
    M /= M.max()

    # Gs = ot.sinkhorn(a_s, a_t, M, 2e-3)
    Gs = ot.emd(a_s, a_t, M)
    return np.dot(np.diag(1 / a_s), np.dot(Gs, Yt))


def OT_transform_ds(Xs, Yt, a_s, a_t):  # Density corrected
    a_s /= a_s.sum()
    a_t /= a_t.sum()
    M = ot.dist(Xs, Yt)
    M /= M.max()

    # Gs = ot.sinkhorn(a_s, a_t, M, 2e-3)
    Gs = ot.emd(a_s, a_t, M)
    return np.dot(np.diag(1 / a_s), np.dot(Gs, Yt))
