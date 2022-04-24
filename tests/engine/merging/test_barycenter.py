#!/usr/bin/env python3

from transmorph.datasets import load_test_datasets_small
from transmorph.engine.matching import Labels
from transmorph.engine.merging import Barycenter
from transmorph.engine.traits import UsesReference
from transmorph.utils import plot_result, anndata_manager as adm, AnnDataKeyIdentifiers


def test_merging_barycenter():
    # Tests matching quality of partial OT on small controlled dataset
    datasets = list(load_test_datasets_small().values())
    UsesReference.write_is_reference(datasets[1])
    matching = Labels(label_obs="class")
    matching.retrieve_labels(datasets)
    T = matching.fit([adata.X for adata in datasets])
    mg = Barycenter()
    mg.retrieve_reference_index(datasets)
    mg.set_matchings(T)
    Xs_out = mg.transform([adata.X for adata in datasets])
    for adata, X_out in zip(datasets, Xs_out):
        adm.set_value(
            adata=adata,
            key=AnnDataKeyIdentifiers.TransmorphRepresentation,
            field="obsm",
            value=X_out,
            persist="output",
        )
    plot_result(
        datasets=datasets,
        color_by="class",
        title="Merging barycenter",
        xlabel="Feature 1",
        ylabel="Feature 2",
        show=False,
        save=True,
        caller_path=__file__,
        suffix="",
    )


if __name__ == "__main__":
    test_merging_barycenter()