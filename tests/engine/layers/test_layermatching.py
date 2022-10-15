#!/usr/bin/env python3

import numpy as np

from transmorph.datasets import load_test_datasets_small, load_travaglini_10x
from transmorph.engine.layers import LayerInput, LayerMatching
from transmorph.engine.matching import Matching, Labels, OT, GW, FusedGW, MNN
from transmorph.engine.subsampling import VertexCover
from transmorph.engine.transforming import CommonFeatures, Standardize, PCA
from transmorph.engine.traits import (
    HasMetadata,
    UsesMetric,
    UsesSampleLabels,
)

ALL_MATCHINGS = [
    # constructor, parameters
    (FusedGW, {}),
    (GW, {}),
    (Labels, {"label_obs": "class"}),
    (MNN, {"n_neighbors": 3}),
    (OT, {}),
]
N_PCS = 20

travaglini = list(load_travaglini_10x().values())


def test_layer_matching():
    # Tests all types of matchings in a layer to ensure
    # all needed traits are initialized.
    # We trust matching tests to ensure correctness of
    # Matchings.
    datasets = list(load_test_datasets_small().values())
    for matching_algo, kwargs in ALL_MATCHINGS:
        linput = LayerInput()
        matching_lay: Matching = matching_algo(**kwargs)
        if isinstance(matching_lay, HasMetadata):
            matching_lay.retrieve_all_metadata(datasets)
        if isinstance(matching_lay, UsesSampleLabels):
            matching_lay.retrieve_all_labels(datasets)
        if isinstance(matching_lay, UsesMetric):
            matching_lay.retrieve_all_metrics(datasets)
        lmatching = LayerMatching(matching=matching_lay)
        linput.connect(lmatching)
        linput.fit(datasets)
        lmatching.fit(datasets)

        matching_ref: Matching = matching_algo(**kwargs)
        if isinstance(matching_ref, HasMetadata):
            matching_ref.retrieve_all_metadata(datasets)
        if isinstance(matching_ref, UsesSampleLabels):
            matching_ref.retrieve_all_labels(datasets)
        if isinstance(matching_ref, UsesMetric):
            matching_ref.retrieve_all_metrics(datasets)
        dict_true = matching_ref.fit([adata.X for adata in datasets])
        dict_test = lmatching.matching_matrices

        assert dict_test is not None
        for key in dict_test:
            assert key in dict_true
        for key in dict_true:
            assert key in dict_test
            Ttrue = dict_true[key].toarray()
            Ttest = dict_test[key].toarray()
            np.testing.assert_array_equal(Ttrue, Ttest)


def test_layer_matching_contains_transformations():
    # Testing layer matching with transformations
    # embedded as preprocessing steps.
    # TODO: find a way to test the internal state of
    # layermatching in a better way. For now we
    # only trust the logs and absence of crash :(
    linput = LayerInput()
    matching = Labels(label_obs="class")
    lmatching = LayerMatching(matching=matching)
    lmatching.add_transformation(CommonFeatures())
    lmatching.add_transformation(Standardize(center=True, scale=True))
    lmatching.add_transformation(PCA(n_components=N_PCS))
    linput.connect(lmatching)
    linput.fit(travaglini)
    lmatching.fit(travaglini)


def test_layer_matching_subsampling():
    # Testing layer matching with a subsampling.
    datasets = list(load_test_datasets_small().values())
    linput = LayerInput()
    matching = Labels(label_obs="class")
    lmatching = LayerMatching(matching=matching, subsampling=VertexCover())
    linput.connect(lmatching)
    linput.fit(datasets)
    lmatching.fit(datasets)


def test_layer_matching_mnn_qtree():
    # Testing MNN <-> UsesNeighbors qtree interface
    linput = LayerInput()
    matching = MNN(n_neighbors=20)
    lmatching = LayerMatching(matching=matching)
    lmatching.add_transformation(CommonFeatures())
    lmatching.add_transformation(Standardize(True, True))
    lmatching.add_transformation(PCA(n_components=30))
    linput.connect(lmatching)
    linput.fit(travaglini)
    lmatching.fit(travaglini)


if __name__ == "__main__":
    test_layer_matching_mnn_qtree()
