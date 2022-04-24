#!/usr/bin/env python3

import numpy as np

from transmorph import settings
from transmorph.datasets import load_test_datasets_small
from transmorph.engine.layers import (
    LayerInput,
    LayerMatching,
    LayerMerging,
    LayerOutput,
)
from transmorph.engine.matching import Labels
from transmorph.engine.merging import (
    Merging,
    Barycenter,
    LinearCorrection,
    GraphEmbedding,
)
from transmorph.engine.traits import HasMetadata, UsesNeighbors, UsesReference
from transmorph.utils import anndata_manager as adm, AnnDataKeyIdentifiers

ALL_MERGINGS = [
    # constructor, parameters
    (Barycenter, {}),
    (LinearCorrection, {}),
    (GraphEmbedding, {"optimizer": "umap"}),
]


def test_layer_merging():
    # Tests all types of merging in a simple
    # in -> matching -> merging -> out setup.
    settings.n_neighbors = 3
    datasets = list(load_test_datasets_small().values())
    for merging_algo, kwargs in ALL_MERGINGS:
        # Writing metadata
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
        UsesReference.write_is_reference(datasets[1])

        # Building model
        linput = LayerInput()
        matching = Labels(label_obs="class")
        matching.retrieve_labels(datasets)
        lmatching = LayerMatching(matching=matching)
        merging: Merging = merging_algo(**kwargs)
        lmerging = LayerMerging(merging=merging)
        loutput = LayerOutput()
        linput.connect(lmatching)
        lmatching.connect(lmerging)
        lmerging.connect(loutput)

        # Fitting model
        linput.fit(datasets)
        lmatching.fit(datasets)
        lmerging.fit(datasets)
        loutput.fit(datasets)

        # Comparing out results to merging results
        Xs_test = [lmerging.get_representation(adata) for adata in datasets]
        Xs_out = [loutput.get_representation(adata) for adata in datasets]
        for Xm, Xo in zip(Xs_test, Xs_out):
            np.testing.assert_array_equal(Xm, Xo)

        # Testing merged positions against reference
        assert lmatching.matching_matrices is not None
        merging: Merging = merging_algo(**kwargs)
        if isinstance(merging, HasMetadata):
            merging.retrieve_all_metadata(datasets)
        if isinstance(merging, UsesReference):
            merging.retrieve_reference_index(datasets)
        merging.set_matchings(lmatching.matching_matrices)
        Xs_true = merging.transform([adata.X for adata in datasets])

        for X_test, X_true in zip(Xs_test, Xs_true):
            # Cannot control randomness of UMAP GraphEmbedding, despite
            # setting randomstate.
            if isinstance(merging, GraphEmbedding):
                continue
            np.testing.assert_array_equal(X_test, X_true)

        # Removing all non-output content.
        for adata in datasets:
            adm.clean(adata, level="pipeline")


if __name__ == "__main__":
    test_layer_merging()
