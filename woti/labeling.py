#!/usr/bin/env python3

import numpy as np

def _transfer(
        y: np.ndarray,
        P: np.ndarray
) -> np.ndarray:
    """
    Performs OT-based label transfer
    """
    # P OT plan, (n,m) matrix
    # y reference labels (m,1)
    return y[np.argsort(P, axis=0)[:,-1]]
