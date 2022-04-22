#!/usr/bin/env python3

from transmorph.datasets import load_test_datasets_small
from transmorph.engine.matching import MatchingOT
from transmorph.stats import edge_accuracy
from transmorph.utils import plot_result


def test_matching_emd_accuracy():
    # Tests matching quality of EMD on small controlled dataset
    datasets = load_test_datasets_small()
    src, ref = datasets["src"], datasets["ref"]
    for optimizer in ("emd", "sinkhorn", "partial"):
        mt = MatchingOT(optimizer=optimizer)
        results = mt.fit([src.X, ref.X])
        T = results[0, 1]
        accuracy = edge_accuracy(src, ref, T, "class")
        # assert accuracy >= 0.65
        plot_result(
            datasets=[src, ref],
            matching_mtx=T,
            color_by="class",
            title=f"Optimal transport matching [acc={'{:.2f}'.format(accuracy)}]",
            xlabel="Feature 1",
            ylabel="Feature 2",
            show=False,
            save=True,
            caller_path=f"{__file__}",
            suffix=optimizer,
        )


if __name__ == "__main__":
    test_matching_emd_accuracy()
