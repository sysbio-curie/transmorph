#!/usr/bin/env python3

from transmorph.datasets import load_test_datasets_small, load_travaglini_10x
from transmorph.engine.matching import MatchingOT
from transmorph.stats import edge_accuracy
from transmorph.utils import plot_result


def test_matching_ot_accuracy():
    # Tests matching quality of OT matching on small controlled dataset
    datasets = load_test_datasets_small()
    src, ref = datasets["src"], datasets["ref"]
    scores = [0.8, 0.8, 0.5]
    for i, optimizer in enumerate(["emd", "sinkhorn", "partial"]):
        kwargs = {}
        if optimizer == "partial":
            kwargs["partial_transport_mass"] = 0.5
        mt = MatchingOT(optimizer=optimizer, **kwargs)
        results = mt.fit([src.X, ref.X])
        T = results[0, 1]
        time = mt.get_time_spent() * 1000
        accuracy = edge_accuracy(src, ref, T, "class")
        assert accuracy >= scores[i]
        plot_result(
            datasets=[src, ref],
            matching_mtx=T,
            color_by="class",
            title="Optimal transport matching\n"
            f"[acc={'{:.2f}'.format(accuracy)}, "
            f"time={'{:.2f}'.format(time)}ms]",
            xlabel="Feature 1",
            ylabel="Feature 2",
            show=False,
            save=True,
            caller_path=__file__,
            suffix=optimizer,
        )


def test_matching_ot_commongenes():
    # Tests the UsesCommonFeatures trait
    datasets = list(load_travaglini_10x().values())
    mt = MatchingOT(common_features_mode="pairwise")
    mt.retrieve_common_features(datasets, True)
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        Xi, Xj = mt.slice_features(
            X1=datasets[i].X,
            idx_1=i,
            X2=datasets[j].X,
            idx_2=j,
        )
        assert Xi.shape[1] == Xj.shape[1]
    mt = MatchingOT(common_features_mode="total")
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
    test_matching_ot_accuracy()
    test_matching_ot_commongenes()
