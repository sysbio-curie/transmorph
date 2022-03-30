#!/usr/bin/env python3

from scipy.sparse import csr_matrix

from transmorph.datasets import load_test_datasets_small
from transmorph.merging import MergingMDI
from transmorph.utils import matching_divergence

from transmorph.utils.plotting import plot_result


def test_merging_mdi():
    # Tests matching quality of partial OT on small controlled dataset
    datasets = load_test_datasets_small()
    src, ref = datasets["src"], datasets["ref"]
    match = csr_matrix(1.0 - datasets["error"])
    mg = MergingMDI(n_neighbors=3)
    src.obsm["transmorph"], ref.obsm["transmorph"] = mg.fit(
        [src, ref], matching_mtx=match
    )
    score = matching_divergence(src.obsm["transmorph"], ref.obsm["transmorph"], match)
    assert score < 2.5

    plot_result(
        datasets=[src, ref],
        color_by="class",
        title=f"Merging MDI (MD={'{:.2f}'.format(score)})",
        xlabel="Feature 1",
        ylabel="Feature 2",
        show=False,
        save=True,
        caller_path=f"{__file__}",
        suffix="",
    )


if __name__ == "__main__":
    test_merging_mdi()
