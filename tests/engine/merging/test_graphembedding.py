#!/usr/bin/env python3

from transmorph.datasets import load_test_datasets_small
from transmorph.engine.matching import Labels
from transmorph.engine.merging import GraphEmbedding
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
    matching = Labels(label_obs="class")
    matching.retrieve_all_labels(datasets)
    T = matching.fit([adata.X for adata in datasets])
    mg = GraphEmbedding(optimizer="umap", n_neighbors=3)
    mg.set_matchings(T)
    Xs_out = mg.transform(datasets, [adata.X for adata in datasets])
    for adata, X_out in zip(datasets, Xs_out):
        adm.set_value(
            adata=adata,
            key=AnnDataKeyIdentifiers.TransmorphRepresentation,
            field="obsm",
            value=X_out,
            persist="output",
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
    matching = Labels(label_obs="class")
    matching.retrieve_all_labels(datasets)
    T = matching.fit([adata.X for adata in datasets])
    mg = GraphEmbedding(optimizer="mde", n_neighbors=3)
    mg.set_matchings(T)
    Xs_out = mg.transform(datasets, [adata.X for adata in datasets])
    for adata, X_out in zip(datasets, Xs_out):
        adm.set_value(
            adata=adata,
            key=AnnDataKeyIdentifiers.TransmorphRepresentation,
            field="obsm",
            value=X_out,
            persist="output",
        )


if __name__ == "__main__":
    test_merging_umap()
    test_merging_mde()
