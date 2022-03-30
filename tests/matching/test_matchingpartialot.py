#!/usr/bin/env python3

import numpy as np

from transmorph.datasets import load_test_datasets_small
from transmorph.matching import MatchingPartialOT

from transmorph.utils.plotting import plot_result


def test_matching_partialot_accuracy():
    # Tests matching quality of partial OT on small controlled dataset
    datasets = load_test_datasets_small()
    src, ref = datasets["src"], datasets["ref"]
    err_matchs = datasets["error"]
    thrs = [1.0, 0.5, 0.95, 0.95, 1.0, 0.9, 0.85, 0.75, 0.8, 0.8]
    for ratio, thr in zip(np.arange(1, 11) / 10, thrs):
        mt = MatchingPartialOT(transport_mass=ratio)
        mt.fit([src, ref])
        T = mt.get_matching(src, ref)
        errors = (T.toarray() * err_matchs).sum()
        accuracy = 1 - errors / T.toarray().sum()
        assert accuracy >= thr

        title = (
            f"Partial optimal transport (eps={'{:.1f}'.format(ratio)}) "
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
            suffix="{:.1f}".format(ratio),
        )


if __name__ == "__main__":
    test_matching_partialot_accuracy()
