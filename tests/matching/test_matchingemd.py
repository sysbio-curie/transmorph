#!/usr/bin/env python3

from transmorph.datasets import load_test_datasets_small
from transmorph.matching import MatchingEMD


def test_matching_emd_accuracy():
    # Tests matching quality of MNN on small controlled dataset
    datasets = load_test_datasets_small()
    src, ref = datasets["src"], datasets["ref"]
    err_matchs = datasets["error"]
    mt = MatchingEMD()
    mt.fit([src, ref])
    T = mt.get_matching(src, ref).toarray()
    errors = (T * err_matchs).sum()
    accuracy = 1 - errors / T.sum()
    assert accuracy > 0.8


if __name__ == "__main__":
    test_matching_emd_accuracy()
