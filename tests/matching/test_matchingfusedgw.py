#!/usr/bin/env python3

from transmorph.datasets import load_test_datasets_small
from transmorph.matching import MatchingFusedGW
from transmorph.stats import edge_accuracy
from transmorph.utils import plot_result

import numpy as np


def test_matching_fusedgw_accuracy():
    # Tests matching quality of Fused GW on small controlled dataset
    datasets = load_test_datasets_small()
    src, ref = datasets["src"], datasets["ref"]
    thrs = [0.65] * 10 + [-1]
    for alpha, thr in zip(np.arange(11) / 10, thrs):
        mt = MatchingFusedGW(alpha=alpha)
        mt.fit([src, ref])
        T = mt.get_matching(src, ref)
        accuracy = edge_accuracy(src, ref, T, "class")
        assert accuracy >= thr

        title = (
            f"{'{:.1f}'.format(alpha)}-Fused Gromov-Wasserstein matching "
            f"[acc={'{:.2f}'.format(accuracy)}]"
        )
        plot_result(
            datasets=[src, ref],
            matching_mtx=T,
            color_by="class",
            title=title,
            xlabel="Feature 1",
            ylabel="Feature 2",
            show=False,
            save=True,
            caller_path=f"{__file__}",
            suffix="{:.1f}".format(alpha),
        )


if __name__ == "__main__":
    test_matching_fusedgw_accuracy()
