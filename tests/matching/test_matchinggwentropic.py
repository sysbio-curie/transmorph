#!/usr/bin/env python3

from transmorph.datasets import load_test_datasets_small
from transmorph.matching import MatchingGWEntropic
from transmorph.stats import edge_accuracy
from transmorph.utils import plot_result


def test_matching_gwentropic_accuracy():
    # Tests matching quality of entropic GW on small controlled dataset
    datasets = load_test_datasets_small()
    src, ref = datasets["src"], datasets["ref"]
    thrs = [-1] * 4
    for epsilon, thr in zip([1e-3, 5e-3, 1e-2, 5e-2], thrs):
        mt = MatchingGWEntropic(epsilon=epsilon)
        mt.fit([src, ref])
        T = mt.get_matching(src, ref)
        accuracy = edge_accuracy(src, ref, T, "class")
        assert accuracy >= thr

        title = (
            f"Entropic Gromov-Wasserstein (eps={'{:.3f}'.format(epsilon)}) "
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
            suffix="{:.3f}".format(epsilon),
        )


if __name__ == "__main__":
    test_matching_gwentropic_accuracy()
