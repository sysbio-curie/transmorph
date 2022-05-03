#!/usr/bin/env python3

from transmorph.datasets import load_test_datasets_small
from transmorph.engine.matching import GW
from transmorph.engine.traits import UsesMetric
from transmorph.stats import edge_accuracy
from transmorph.utils import plot_result


def test_matching_gw_accuracy():
    # Tests matching quality of GW on small controlled dataset
    for optimizer in ("gw", "entropic_gw"):
        datasets = list(load_test_datasets_small().values())
        # Without custom metric
        mt = GW(optimizer=optimizer)
        mt.retrieve_all_metrics(datasets)
        assert mt.get_metric(0) == ("sqeuclidean", {})
        assert mt.get_metric(1) == ("sqeuclidean", {})
        results = mt.fit([adata.X for adata in datasets])
        T = results[0, 1]
        accuracy = edge_accuracy(datasets[0], datasets[1], T, "class")
        time = mt.get_time_spent() * 1000
        assert accuracy >= -1
        plot_result(
            datasets=datasets,
            matching_mtx=T,
            color_by="class",
            title="Gromov-Wasserstein matching\n"
            f"[acc={'{:.2f}'.format(accuracy)}, "
            f"time={'{:.2f}'.format(time)}ms]",
            xlabel="Feature 1",
            ylabel="Feature 2",
            show=False,
            save=True,
            caller_path=__file__,
            suffix=f"{optimizer}_nometric",
        )

        # With custom metrics
        UsesMetric.set_adata_metric(datasets[0], "cosine")
        UsesMetric.set_adata_metric(datasets[1], "minkowski", {"p": 3})
        mt = GW(optimizer=optimizer)
        mt.retrieve_all_metrics(datasets)
        assert mt.get_metric(0) == ("cosine", {})
        metric, kwargs = mt.get_metric(1)
        assert metric == "minkowski"
        assert "p" in kwargs
        assert kwargs["p"] == 3
        results = mt.fit([adata.X for adata in datasets])
        T = results[0, 1]
        accuracy = edge_accuracy(datasets[0], datasets[1], T, "class")
        time = mt.get_time_spent() * 1000
        assert accuracy >= -1
        plot_result(
            datasets=datasets,
            matching_mtx=T,
            color_by="class",
            title="Gromov-Wasserstein matching\n"
            f"[acc={'{:.2f}'.format(accuracy)}, "
            f"time={'{:.2f}'.format(time)}ms]",
            xlabel="Feature 1",
            ylabel="Feature 2",
            show=False,
            save=True,
            caller_path=__file__,
            suffix=f"{optimizer}_metric",
        )


if __name__ == "__main__":
    test_matching_gw_accuracy()
