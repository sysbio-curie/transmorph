#!/usr/bin/env python3

from transmorph.datasets import load_test_datasets_small
from transmorph.engine.matching import Labels
from transmorph.stats import edge_accuracy
from transmorph.utils import plot_result


def test_matching_labels_accuracy():
    # Tests matching quality of OT matching on small controlled dataset
    datasets = load_test_datasets_small()
    src, ref = datasets["src"], datasets["ref"]
    mt = Labels(label_obs="class")
    mt.retrieve_all_labels([src, ref])
    results = mt.fit([src.X, ref.X])
    T = results[0, 1]
    time = mt.get_time_spent() * 1000
    accuracy = edge_accuracy(src, ref, T, "class")
    assert accuracy == 1.0
    plot_result(
        datasets=[src, ref],
        matching_mtx=T,
        color_by="class",
        title="Labels matching\n"
        f"[acc={'{:.2f}'.format(accuracy)}, "
        f"time={'{:.2f}'.format(time)}ms]",
        xlabel="Feature 1",
        ylabel="Feature 2",
        show=False,
        save=True,
        caller_path=__file__,
        suffix="",
    )


if __name__ == "__main__":
    test_matching_labels_accuracy()
