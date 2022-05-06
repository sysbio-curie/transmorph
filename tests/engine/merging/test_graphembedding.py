#!/usr/bin/env python3

from transmorph.datasets import load_test_datasets_small
from transmorph.engine.matching import Labels
from transmorph.engine.merging import GraphEmbedding
from transmorph.engine.traits import UsesNeighbors
from transmorph.utils.plotting import scatter_plot
from transmorph.utils.anndata_manager import (
    anndata_manager as adm,
    AnnDataKeyIdentifiers,
)


def test_merging_umap():
    # Tests matching quality of partial OT on small controlled dataset
    datasets = list(load_test_datasets_small().values())
    for adata in datasets:
        adm.set_value(
            adata=adata,
            key=AnnDataKeyIdentifiers.BaseRepresentation,
            field="obsm",
            value=adata.X,
            persist="pipeline",
        )
    UsesNeighbors.compute_neighbors_graphs(
        datasets=datasets,
        representation_key=AnnDataKeyIdentifiers.BaseRepresentation,
    )
    matching = Labels(label_obs="class")
    matching.retrieve_all_labels(datasets)
    T = matching.fit([adata.X for adata in datasets])
    mg = GraphEmbedding(optimizer="umap", n_neighbors=3)
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
    scatter_plot(
        datasets=datasets,
        color_by="class",
        title="Merging UMAP",
        xlabel="UMAP1",
        ylabel="UMAP2",
        show=False,
        save=True,
        caller_path=__file__,
        suffix="umap",
    )


def test_merging_mde():
    # Tests matching quality of partial OT on small controlled dataset
    datasets = list(load_test_datasets_small().values())
    for adata in datasets:
        adm.set_value(
            adata=adata,
            key=AnnDataKeyIdentifiers.BaseRepresentation,
            field="obsm",
            value=adata.X,
            persist="pipeline",
        )
    UsesNeighbors.compute_neighbors_graphs(
        datasets=datasets,
        representation_key=AnnDataKeyIdentifiers.BaseRepresentation,
    )
    matching = Labels(label_obs="class")
    matching.retrieve_all_labels(datasets)
    T = matching.fit([adata.X for adata in datasets])
    mg = GraphEmbedding(optimizer="mde", n_neighbors=3)
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
    scatter_plot(
        datasets=datasets,
        color_by="class",
        title="Merging MDE",
        xlabel="MDE1",
        ylabel="MDE2",
        show=False,
        save=True,
        caller_path=__file__,
        suffix="mde",
    )


if __name__ == "__main__":
    test_merging_umap()
    test_merging_mde()
