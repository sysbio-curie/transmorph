#!/usr/bin/env python3

from transmorph.datasets import load_test_datasets_small
from transmorph.datasets import load_test_datasets_random
from transmorph.engine.matching import FusedGW
from transmorph.engine.traits import UsesMetric
from transmorph.stats import edge_accuracy


def test_matching_fusedgw_accuracy():
    # Tests matching quality of GW on small controlled dataset
    datasets = list(load_test_datasets_small().values())
    # Without custom metric
    mt = FusedGW(default_GW_metric="sqeuclidean")
    mt.retrieve_all_metrics(datasets)
    assert mt.get_metric(0) == ("sqeuclidean", {})
    assert mt.get_metric(1) == ("sqeuclidean", {})
    results = mt.fit([adata.X for adata in datasets])
    T = results[0, 1]
    accuracy = edge_accuracy(datasets[0], datasets[1], T, "class")
    assert accuracy >= -1

    # With custom metrics
    UsesMetric.set_adata_metric(datasets[0], "cosine")
    UsesMetric.set_adata_metric(datasets[1], "minkowski", {"p": 3})
    mt = FusedGW(default_GW_metric="sqeuclidean")
    mt.retrieve_all_metrics(datasets)
    assert mt.get_metric(0) == ("cosine", {})
    metric, kwargs = mt.get_metric(1)
    assert metric == "minkowski"
    assert "p" in kwargs
    assert kwargs["p"] == 3
    results = mt.fit([adata.X for adata in datasets])
    T = results[0, 1]
    accuracy = edge_accuracy(datasets[0], datasets[1], T, "class")
    assert accuracy >= -1


def test_matching_fusedgw_commonfeatures():
    # Tests the UsesCommonFeatures trait
    datasets = list(load_test_datasets_random().values())
    mt = FusedGW(common_features_mode="pairwise")
    mt.retrieve_common_features(datasets, True)
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        Xi, Xj = mt.slice_features(
            X1=datasets[i].X,
            idx_1=i,
            X2=datasets[j].X,
            idx_2=j,
        )
        assert Xi.shape[1] == Xj.shape[1]
    mt = FusedGW(common_features_mode="total")
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
    test_matching_fusedgw_commonfeatures()
