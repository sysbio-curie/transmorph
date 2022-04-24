#!/usr/bin/env python3

from transmorph.datasets import load_test_datasets_small
from transmorph.engine.matching import GW
from transmorph.engine.traits import UsesMetric
from transmorph.stats import edge_accuracy
from transmorph.utils import plot_result


def test_matching_gw_accuracy():
    # Tests matching quality of GW on small controlled dataset
    datasets = load_test_datasets_small()
    src, ref = datasets["src"], datasets["ref"]
    UsesMetric.set_metric(src, "cosine")
    UsesMetric.set_metric(ref, "minkowski", {"p": 3})
    for optimizer in ("gw", "entropic_gw"):
        # Without custom metric
        mt = GW(optimizer=optimizer)
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
        mt = GW(optimizer=optimizer)
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
