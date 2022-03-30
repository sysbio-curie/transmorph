#!/usr/bin/env python3

from transmorph.datasets import load_test_datasets_small
from transmorph.matching import MatchingFusedGW

import numpy as np

from transmorph.utils.plotting import plot_result


def test_matching_fusedgw_accuracy():
    # Tests matching quality of Fused GW on small controlled dataset
    datasets = load_test_datasets_small()
    src, ref = datasets["src"], datasets["ref"]
    err_matchs = datasets["error"]
    thrs = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.0]
    for alpha, thr in zip(np.arange(11) / 10, thrs):
        mt = MatchingFusedGW(alpha=alpha)
        mt.fit([src, ref])
        T = mt.get_matching(src, ref)
        errors = (T.toarray() * err_matchs).sum()
        accuracy = 1 - errors / T.toarray().sum()
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
