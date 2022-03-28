#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os

from transmorph.datasets import load_test_datasets_small
from transmorph.matching import MatchingFusedGW


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

        plt.figure()
        plt.scatter(*src.X.T, label="src", s=60, ec="k")
        plt.scatter(*ref.X.T, label="ref", s=60, ec="k")

        Tcoo = T.tocoo()
        for i, j, v in zip(Tcoo.row, Tcoo.col, Tcoo.data):
            plt.plot(
                [src.X[i][0], ref.X[j][0]],
                [src.X[i][1], ref.X[j][1]],
                alpha=v * ref.n_obs,
                c="k",
            )
        plt.legend()
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title(
            f"{'{:.1f}'.format(alpha)}-Fused Gromov-Wasserstein matching "
            f"[acc={'{:.2f}'.format(accuracy)}]"
        )
        plt.savefig(
            f"{os.getcwd()}/transmorph/tests/matching/figures/small_"
            f"fusedgw_{'{:.1f}'.format(alpha)}.png"
        )
        plt.close()


if __name__ == "__main__":
    test_matching_fusedgw_accuracy()
