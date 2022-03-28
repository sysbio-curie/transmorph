#!/usr/bin/env python3

import numpy as np

from transmorph.datasets import load_test_datasets_small
from transmorph.matching import MatchingMNN


def test_matching_mnn_accuracy():
    datasets = load_test_datasets_small()
    src, ref = datasets["src"], datasets["ref"]
    thrs = [0.85, 0.90, 0.80, 0.75, 0.70]
    for nnb, thr in enumerate(thrs, 1):
        mt = MatchingMNN(n_neighbors=nnb)
        mt.fit([src, ref])
        T = mt.get_matching(src, ref).toarray()
        err_matchs = np.array(src.obs["class"])[:, None] != np.array(ref.obs["class"])
        errors = (T * err_matchs).sum()
        accuracy = 1 - errors / T.sum()
        assert accuracy > thr


if __name__ == "__main__":
    test_matching_mnn_accuracy()
