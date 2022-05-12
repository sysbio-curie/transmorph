#!/usr/bin/env python3

from transmorph.datasets import load_test_datasets_small, load_travaglini_10x
from transmorph.engine.matching import MNN
from transmorph.stats import edge_accuracy
from transmorph.utils.plotting import scatter_plot


def test_matching_mnn_accuracy():
    # Tests matching quality of OT matching on small controlled dataset
    datasets = load_test_datasets_small()
    src, ref = datasets["src"], datasets["ref"]
    mt = MNN(n_neighbors=3)
    results = mt.fit([src.X, ref.X])
    T = results[0, 1]
    time = mt.get_time_spent() * 1000
    accuracy = edge_accuracy(src, ref, T, "class")
    assert accuracy >= 0.5
    scatter_plot(
        datasets=[src, ref],
        matching_mtx=T,
        color_by="class",
        title="MNN matching\n"
        f"[acc={'{:.2f}'.format(accuracy)}, "
        f"time={'{:.2f}'.format(time)}ms]",
        xlabel="Feature 1",
        ylabel="Feature 2",
        show=False,
        save=True,
        caller_path=__file__,
        suffix="",
    )


def test_matching_mnn_commonfeatures():
    # Tests the UsesCommonFeatures trait
    datasets = list(load_travaglini_10x().values())
    mt = MNN(common_features_mode="pairwise")
    mt.retrieve_common_features(datasets, True)
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        Xi, Xj = mt.slice_features(
            X1=datasets[i].X,
            idx_1=i,
            X2=datasets[j].X,
            idx_2=j,
        )
        assert Xi.shape[1] == Xj.shape[1]
    mt = MNN(common_features_mode="total")
    mt.retrieve_common_features(datasets, True)
    sliced = [
        mt.slice_features(
            X1=datasets[i].X,
            idx_1=i,
        )
        for i in range(len(datasets))
    ]
    slice_size = sliced[0].shape[1]
    assert all(X.shape[1] == slice_size for X in sliced)
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        Xi, Xj = mt.slice_features(
            X1=datasets[i].X,
            idx_1=i,
            X2=datasets[j].X,
            idx_2=j,
        )
        assert Xi.shape[1] == Xj.shape[1] == slice_size


if __name__ == "__main__":
    test_matching_mnn_accuracy()
    # test_matching_mnn_commonfeatures()
