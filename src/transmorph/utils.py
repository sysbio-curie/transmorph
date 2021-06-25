#!/usr/bin/env python3

import numpy as np

def col_normalize(x: np.ndarray):
    x_std = x.std(axis=0)
    x_std[x_std == 0] = 1
    return x / x_std
