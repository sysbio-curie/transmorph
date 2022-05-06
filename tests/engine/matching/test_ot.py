#!/usr/bin/env python3

from transmorph.datasets import load_test_datasets_small
from transmorph.engine.matching import OT
from transmorph.stats import edge_accuracy
from transmorph.utils.plotting import scatter_plot


def test_matching_ot_accuracy():
    # Tests matching quality of OT matching on small controlled dataset
    datasets = load_test_datasets_small()
    src, ref = datasets["src"], datasets["ref"]
    scores = [0.8, 0.8, 0.5, 0.9]
    for i, solver in enumerate(["emd", "sinkhorn", "partial", "unbalanced"]):
        kwargs = {}
        if solver == "partial":
            kwargs["partial_transport_mass"] = 0.5
        mt = OT(solver=solver, **kwargs)
        results = mt.fit([src.X, ref.X])
        T = results[0, 1]
        time = mt.get_time_spent() * 1000
        accuracy = edge_accuracy(src, ref, T, "class")
        scatter_plot(
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
            suffix=solver,
        )
        assert accuracy >= scores[i]


if __name__ == "__main__":
    test_matching_ot_accuracy()
