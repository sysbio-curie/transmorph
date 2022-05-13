#!/usr/bin/env python3

import numpy as np

from transmorph.datasets import load_test_datasets_small, load_bank
from transmorph.engine.layers import (
    LayerInput,
    LayerMatching,
    LayerMerging,
    LayerOutput,
    LayerTransformation,
)
from transmorph.engine.matching import Labels
from transmorph.engine.merging import (
    Merging,
    Barycenter,
    LinearCorrection,
    GraphEmbedding,
)
from transmorph.engine.traits import (
    HasMetadata,
    UsesNeighbors,
    UsesReference,
    IsRepresentable,
)
from transmorph.engine.transforming import Standardize, PCA
from transmorph.utils.anndata_manager import (
    anndata_manager as adm,
    AnnDataKeyIdentifiers,
)

ALL_MERGINGS = [
    # constructor, parameters
    (Barycenter, {}),
    (LinearCorrection, {}),
    (GraphEmbedding, {"optimizer": "umap"}),
]
N_PCS = 20

travaglini = list(load_bank("travaglini_10x", n_samples=1000).values())


def test_layer_merging():
    # Tests all types of merging in a simple
    # in -> matching -> merging -> out setup.
    datasets = list(load_test_datasets_small().values())
    for merging_algo, kwargs in ALL_MERGINGS:
        # Writing metadata
        UsesNeighbors.compute_neighbors_graphs(
            datasets=datasets,
            representation_key=AnnDataKeyIdentifiers.BaseRepresentation,
        )
        UsesReference.write_is_reference(datasets[1])

        # Building model
        linput = LayerInput()
        matching = Labels(label_obs="class")
        matching.retrieve_all_labels(datasets)
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
        IsRepresentable.assert_representation_equals([lmerging, loutput], datasets)

        # Testing merged positions against reference
        assert lmatching.matching_matrices is not None
        merging: Merging = merging_algo(**kwargs)
        if isinstance(merging, HasMetadata):
            merging.retrieve_all_metadata(datasets)
        if isinstance(merging, UsesReference):
            merging.retrieve_reference_index(datasets)
        merging.set_matchings(lmatching.matching_matrices)
        Xs_true = merging.transform([adata.X for adata in datasets])
        Xs_test = [lmerging.get_representation(adata) for adata in datasets]

        for X_test, X_true in zip(Xs_test, Xs_true):
            # Cannot control randomness of UMAP GraphEmbedding, despite
            # setting randomstate.
            if isinstance(merging, GraphEmbedding):
                continue
            np.testing.assert_array_equal(X_test, X_true)

        # Removing all non-output content.
        UsesNeighbors.reset()
        for adata in datasets:
            adm.clean(adata, level="pipeline")


def test_is_feature_space_propagation_1():
    # Tests if is_feature_space is correctly propagated
    # along the network.
    # Network 1: Not feature space from the beginning
    # but then only space-preserving operations
    UsesReference.write_is_reference(travaglini[1])
    matching = Labels(label_obs="compartment")
    UsesNeighbors.compute_neighbors_graphs(
        travaglini,
        AnnDataKeyIdentifiers.BaseRepresentation,
    )
    merging = LinearCorrection()
    linput = LayerInput()
    lmatching = LayerMatching(matching=matching)
    lmerging = LayerMerging(merging)
    lmerging.add_transformation(Standardize(True, True))
    loutput = LayerOutput()
    linput.connect(lmatching)
    lmatching.connect(lmerging)
    lmerging.connect(loutput)
    linput.fit(travaglini)
    linput.is_feature_space = False
    lmatching.fit(travaglini)
    lmerging.fit(travaglini)
    loutput.fit(travaglini)
    adm.clean(travaglini, level="output")
    assert linput.is_feature_space is False
    assert lmerging.is_feature_space is False
    assert loutput.is_feature_space is False


def test_is_feature_space_propagation_2():
    # Tests if is_feature_space is correctly propagated
    # along the network.
    # Network 2: Feature space from the beginning
    # and then only space-preserving operations
    UsesReference.write_is_reference(travaglini[1])
    matching = Labels(label_obs="compartment")
    UsesNeighbors.compute_neighbors_graphs(
        travaglini,
        AnnDataKeyIdentifiers.BaseRepresentation,
    )
    merging = LinearCorrection()
    linput = LayerInput()
    lmatching = LayerMatching(matching=matching)
    lmerging = LayerMerging(merging)
    lmerging.add_transformation(Standardize(True, True))
    loutput = LayerOutput()
    linput.connect(lmatching)
    lmatching.connect(lmerging)
    lmerging.connect(loutput)
    linput.fit(travaglini)
    lmatching.fit(travaglini)
    lmerging.fit(travaglini)
    loutput.fit(travaglini)
    adm.clean(travaglini, level="output")
    assert linput.is_feature_space is True
    assert lmerging.is_feature_space is True
    assert loutput.is_feature_space is True


def test_is_feature_space_propagation_3():
    # Tests if is_feature_space is correctly propagated
    # along the network.
    # Network 3: Feature space from the beginning
    # but changes in the middle, then continues
    # with space-preserving operations.
    UsesReference.write_is_reference(travaglini[1])
    matching = Labels(label_obs="compartment")
    UsesNeighbors.compute_neighbors_graphs(
        travaglini,
        AnnDataKeyIdentifiers.BaseRepresentation,
    )
    merging = LinearCorrection()
    linput = LayerInput()
    ltran1 = LayerTransformation()
    ltran1.add_transformation(Standardize())
    ltran1.add_transformation(Standardize())
    ltran2 = LayerTransformation()
    ltran2.add_transformation(Standardize())
    ltran2.add_transformation(PCA(n_components=N_PCS))  # <- broken here
    ltran2.add_transformation(Standardize())
    lmatching = LayerMatching(matching=matching)
    lmerging = LayerMerging(merging)
    lmerging.add_transformation(Standardize(True, True))
    loutput = LayerOutput()
    linput.connect(ltran1)
    ltran1.connect(ltran2)
    ltran2.connect(lmatching)
    lmatching.connect(lmerging)
    lmerging.connect(loutput)
    linput.fit(travaglini)
    ltran1.fit(travaglini)
    ltran2.fit(travaglini)
    lmatching.fit(travaglini)
    lmerging.fit(travaglini)
    loutput.fit(travaglini)
    adm.clean(travaglini, level="output")
    assert linput.is_feature_space is True
    assert ltran1.is_feature_space is True
    assert ltran2.is_feature_space is False
    assert lmerging.is_feature_space is False
    assert loutput.is_feature_space is False


def test_is_feature_space_propagation_4():
    # Tests if is_feature_space is correctly propagated
    # along the network.
    # Network 4: Feature space from the beginning
    # but changes in the middle, then continues
    # with space-preserving operations. But the last
    # step uses a feature space embedding.
    UsesReference.write_is_reference(travaglini[1])
    matching = Labels(label_obs="compartment")
    UsesNeighbors.compute_neighbors_graphs(
        travaglini,
        AnnDataKeyIdentifiers.BaseRepresentation,
    )
    merging = LinearCorrection()
    linput = LayerInput()
    ltran1 = LayerTransformation()
    ltran1.add_transformation(Standardize())
    ltran1.add_transformation(Standardize())
    ltran2 = LayerTransformation()
    ltran2.add_transformation(Standardize())
    ltran2.add_transformation(PCA(n_components=N_PCS))  # <- broken here
    ltran2.add_transformation(Standardize())
    lmatching = LayerMatching(matching=matching)
    lmerging = LayerMerging(merging)
    lmerging.add_transformation(Standardize(True, True))
    loutput = LayerOutput()
    lmerging.embedding_reference = ltran1  # <- saved here
    linput.connect(ltran1)
    ltran1.connect(ltran2)
    ltran2.connect(lmatching)
    lmatching.connect(lmerging)
    lmerging.connect(loutput)
    linput.fit(travaglini)
    ltran1.fit(travaglini)
    ltran2.fit(travaglini)
    lmatching.fit(travaglini)
    lmerging.fit(travaglini)
    loutput.fit(travaglini)
    adm.clean(travaglini, level="output")
    assert linput.is_feature_space is True
    assert ltran1.is_feature_space is True
    assert ltran2.is_feature_space is False
    assert lmerging.is_feature_space is True
    assert loutput.is_feature_space is True


def test_is_feature_space_propagation_5():
    # Tests if is_feature_space is correctly propagated
    # along the network.
    # Network 5: Feature space from the beginning,
    # then only space-preserving operations. The
    # last merging is endowed with a space transforming
    # transformation, which should invalidate feature
    # space preservation on last layers.
    UsesReference.write_is_reference(travaglini[1])
    matching = Labels(label_obs="compartment")
    UsesNeighbors.compute_neighbors_graphs(
        travaglini,
        AnnDataKeyIdentifiers.BaseRepresentation,
    )
    merging = LinearCorrection()
    linput = LayerInput()
    lmatching = LayerMatching(matching=matching)
    lmerging = LayerMerging(merging)
    lmerging.add_transformation(Standardize(True, True))
    lmerging.add_transformation(PCA(n_components=N_PCS))
    loutput = LayerOutput()
    linput.connect(lmatching)
    lmatching.connect(lmerging)
    lmerging.connect(loutput)
    linput.fit(travaglini)
    lmatching.fit(travaglini)
    lmerging.fit(travaglini)
    loutput.fit(travaglini)
    adm.clean(travaglini, level="output")
    assert linput.is_feature_space is True
    assert lmerging.is_feature_space is False
    assert loutput.is_feature_space is False


if __name__ == "__main__":
    test_layer_merging()
