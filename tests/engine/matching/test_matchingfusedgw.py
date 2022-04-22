#!/usr/bin/env python3

from transmorph.datasets import load_test_datasets_small
from transmorph.datasets import load_travaglini_10x
from transmorph.engine.matching import MatchingFusedGW
from transmorph.engine.traits import UsesMetric
from transmorph.stats import edge_accuracy
from transmorph.utils import plot_result


def test_matching_fusedgw_accuracy():
    # Tests matching quality of GW on small controlled dataset
    datasets = load_test_datasets_small()
    src, ref = datasets["src"], datasets["ref"]
    UsesMetric.set_metric(src, "cosine")
    UsesMetric.set_metric(ref, "minkowski", {"p": 3})
    # Without custom metric
    mt = MatchingFusedGW()
    assert mt.get_metadata(0, "metric") == "sqeuclidean"
    assert mt.get_metadata(1, "metric") == "sqeuclidean"
    assert mt.get_metadata(0, "metric_kwargs") == {}
    assert mt.get_metadata(1, "metric_kwargs") == {}
    results = mt.fit([src.X, ref.X])
    T = results[0, 1]
    accuracy = edge_accuracy(src, ref, T, "class")
    time = mt.get_time_spent() * 1000
    assert accuracy >= -1
    plot_result(
        datasets=[src, ref],
        matching_mtx=T,
        color_by="class",
        title="Fused Gromov-Wasserstein matching\n"
        f"[acc={'{:.2f}'.format(accuracy)}, "
        f"time={'{:.2f}'.format(time)}ms]",
        xlabel="Feature 1",
        ylabel="Feature 2",
        show=False,
        save=True,
        caller_path=__file__,
        suffix="nometric",
    )

    # With custom metrics
    mt = MatchingFusedGW()
    mt.retrieve_all_metadata([src, ref])
    assert mt.get_metadata(0, "metric") == "cosine"
    assert mt.get_metadata(1, "metric") == "minkowski"
    assert mt.get_metadata(0, "metric_kwargs") == {}
    mt_kwargs = mt.get_metadata(1, "metric_kwargs")
    assert mt_kwargs is not None
    assert "p" in mt_kwargs
    assert mt_kwargs["p"] == 3
    results = mt.fit([src.X, ref.X])
    T = results[0, 1]
    accuracy = edge_accuracy(src, ref, T, "class")
    time = mt.get_time_spent() * 1000
    assert accuracy >= -1
    plot_result(
        datasets=[src, ref],
        matching_mtx=T,
        color_by="class",
        title="Fused Gromov-Wasserstein matching\n"
        f"[acc={'{:.2f}'.format(accuracy)}, "
        f"time={'{:.2f}'.format(time)}ms]",
        xlabel="Feature 1",
        ylabel="Feature 2",
        show=False,
        save=True,
        caller_path=__file__,
        suffix="metric",
    )


def test_matching_fusedgw_commongenes():
    # Tests the UsesCommonFeatures trait
    datasets = list(load_travaglini_10x().values())
    mt = MatchingFusedGW(common_features_mode="pairwise")
    mt.retrieve_common_features(datasets, True)
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        Xi, Xj = mt.slice_features(
            X1=datasets[i].X,
            idx_1=i,
            X2=datasets[j].X,
            idx_2=j,
        )
        assert Xi.shape[1] == Xj.shape[1]
    mt = MatchingFusedGW(common_features_mode="total")
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
    test_matching_fusedgw_accuracy()
    test_matching_fusedgw_commongenes()
