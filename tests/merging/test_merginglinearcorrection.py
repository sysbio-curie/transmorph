#!/usr/bin/env python3

import matplotlib.pyplot as plt
import os

from scipy.sparse import csr_matrix

from transmorph.datasets import load_test_datasets_small
from transmorph.merging import MergingLinearCorrection
from transmorph.utils import matching_divergence


def test_merging_linearcorrection():
    # Tests matching quality of partial OT on small controlled dataset
    datasets = load_test_datasets_small()
    src, ref = datasets["src"], datasets["ref"]
    match = csr_matrix(1.0 - datasets["error"])
    mg = MergingLinearCorrection(n_neighbors=3)
    X_src, X_ref = mg.fit([src, ref], matching_mtx=match, reference_idx=1)
    score = matching_divergence(X_src, X_ref, match)
    assert score < 2.3

    plt.figure()
    plt.scatter(*src.X.T, label="src (before)", s=60, c="silver")
    plt.scatter(*ref.X.T, label="ref (before)", s=60, c="silver", marker="s")
    plt.scatter(*X_src.T, label="src (after)", s=60, ec="k")
    plt.scatter(*X_ref.T, label="ref", s=60, ec="k", marker="s")

    plt.legend()
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(f"Merging linear correction (MD={'{:.2f}'.format(score)})")
    plt.savefig(
        f"{os.getcwd()}/transmorph/tests/merging/figures/small_linearcorrection.png"
    )
    plt.close()


if __name__ == "__main__":
    test_merging_linearcorrection()
