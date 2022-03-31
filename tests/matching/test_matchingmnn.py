#!/usr/bin/env python3

from transmorph.datasets import load_test_datasets_small
from transmorph.matching import MatchingMNN
from transmorph.utils import plot_result


def test_matching_mnn_accuracy():
    # Tests matching quality of MNN on small controlled dataset
    datasets = load_test_datasets_small()
    src, ref = datasets["src"], datasets["ref"]
    err_matchs = datasets["error"]
    thrs = [1.0, 0.85, 0.90, 0.80, 0.75]
    for nnb, thr in enumerate(thrs, 1):
        mt = MatchingMNN(n_neighbors=nnb)
        mt.fit([src, ref])
        T = mt.get_matching(src, ref)
        errors = (T.toarray() * err_matchs).sum()
        accuracy = 1 - errors / T.toarray().sum()
        assert accuracy >= thr

        title = f"{nnb}-MNN matching [acc={'{:.2f}'.format(accuracy)}]"
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
            suffix=f"{nnb}",
        )


if __name__ == "__main__":
    test_matching_mnn_accuracy()
