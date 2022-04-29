#!/usr/bin/env python3

import numpy as np

from transmorph.engine.checking import LISI
from transmorph.stats.lisi import compute_lisi

NBATCHES, NSAMPLES, NDIMS = 4, 200, 10


def test_checking_lisi():
    # Checks if Checking.LISI = stat.compute_lisi
    Xs = [np.random.random(size=(NSAMPLES, NDIMS)) for _ in range(NBATCHES)]
    y = np.array(sum([[i] * X.shape[0] for i, X in enumerate(Xs)], []))

    # Checking
    checking_lisi = LISI(perplexity=30)
    checking_lisi.check(Xs)
    score_checking = checking_lisi.score

    # Stats
    score_stat = compute_lisi(np.concatenate(Xs, axis=0), y, perplexity=30).mean()
    assert (score_checking - score_stat) <= 1e-6


if __name__ == "__main__":
    test_checking_lisi()
