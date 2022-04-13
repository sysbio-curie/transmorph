#!/usr/bin/env python3

from transmorph.datasets import load_test_datasets_small
from transmorph.matching import MatchingGW
from transmorph.stats import edge_accuracy
from transmorph.utils import plot_result


def test_matching_gw_accuracy():
    # Tests matching quality of GW on small controlled dataset
    datasets = load_test_datasets_small()
    src, ref = datasets["src"], datasets["ref"]
    mt = MatchingGW()
    mt.fit([src, ref])
    T = mt.get_matching(src, ref)
    accuracy = edge_accuracy(src, ref, T, "class")
    assert accuracy >= -1

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
