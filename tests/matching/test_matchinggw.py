#!/usr/bin/env python3

from transmorph.datasets import load_test_datasets_small
from transmorph.matching import MatchingGW

from transmorph.utils.plotting import plot_result


def test_matching_gw_accuracy():
    # Tests matching quality of GW on small controlled dataset
    datasets = load_test_datasets_small()
    src, ref = datasets["src"], datasets["ref"]
    err_matchs = datasets["error"]
    mt = MatchingGW()
    mt.fit([src, ref])
    T = mt.get_matching(src, ref)
    errors = (T.toarray() * err_matchs).sum()
    accuracy = 1 - errors / T.toarray().sum()
    assert accuracy >= 0.04

    plot_result(
        datasets=[src, ref],
        matching_mtx=T,
        color_by="class",
        title=f"Gromov-Wasserstein matching [acc={'{:.2f}'.format(accuracy)}]",
        xlabel="Feature 1",
        ylabel="Feature 2",
        show=False,
        save=True,
        caller_path=f"{__file__}",
        suffix="",
    )


if __name__ == "__main__":
    test_matching_gw_accuracy()
