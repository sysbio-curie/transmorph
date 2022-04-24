#!/usr/bin/env python3

import numpy as np

from transmorph.datasets import load_travaglini_10x
from transmorph.engine.layers import LayerInput, LayerTransformation
from transmorph.engine.traits import HasMetadata, UsesCommonFeatures
from transmorph.engine.transforming import (
    CommonFeatures,
    PCA,
    Standardize,
    Transformation,
)
from transmorph.utils import anndata_manager as adm, AnnDataKeyIdentifiers

ALL_TRANSFORMATIONS = [
    (CommonFeatures, {}),
    (PCA, {"n_components": 20}),
    (Standardize, {"center": True, "scale": True}),
]


def test_layer_transformation():
    # Test input -> transformation with the
    # various transformations available. We
    # trust transformation unit tests, and
    # focus on information passing here.
    datasets = list(load_travaglini_10x().values())
    for transformation_algo, kwargs in ALL_TRANSFORMATIONS:
        # Loading datasets with base representation
        for adata in datasets:
            adm.set_value(
                adata=adata,
                key=AnnDataKeyIdentifiers.BaseRepresentation,
                field="obsm",
                value=adata.X,
                persist="pipeline",
            )
        # Creating and fitting test network
        transformation = transformation_algo(**kwargs)
        linput = LayerInput()
        ltrans = LayerTransformation()
        ltrans.add_transformation(transformation=transformation)
        linput.connect(ltrans)
        linput.fit(datasets)
        ltrans.fit(datasets)
        Xs_test = [ltrans.get_representation(adata) for adata in datasets]

        # Retrieving true value
        transformation: Transformation = transformation_algo(**kwargs)
        if isinstance(transformation, HasMetadata):
            transformation.retrieve_all_metadata(datasets)
        if isinstance(transformation, UsesCommonFeatures):
            transformation.retrieve_common_features(datasets, True)
        Xs_true = transformation.transform([adata.X for adata in datasets])

        for X_true, X_test in zip(Xs_true, Xs_test):
            np.testing.assert_array_almost_equal(X_true, X_test)

        # Cleaning
        for adata in datasets:
            adm.clean(adata, level="pipeline")


def test_layer_transformation_order():
    # Tests transformations are applied in the
    # appending order.
    # Loading datasets with base representation
    datasets = list(load_travaglini_10x().values())
    for adata in datasets:
        adm.set_value(
            adata=adata,
            key=AnnDataKeyIdentifiers.BaseRepresentation,
            field="obsm",
            value=adata.X,
            persist="pipeline",
        )
    linput = LayerInput()
    ltrans = LayerTransformation()
    # Creating and fitting test network
    for transformation_algo, kwargs in ALL_TRANSFORMATIONS:
        transformation = transformation_algo(**kwargs)
        ltrans.add_transformation(transformation=transformation)
    linput.connect(ltrans)
    linput.fit(datasets)
    ltrans.fit(datasets)
    Xs_test = [ltrans.get_representation(adata) for adata in datasets]

    # Retrieving true value
    feature_space = True
    Xs_true = [adata.X for adata in datasets]
    for transformation_algo, kwargs in ALL_TRANSFORMATIONS:
        transformation: Transformation = transformation_algo(**kwargs)
        if isinstance(transformation, HasMetadata):
            transformation.retrieve_all_metadata(datasets)
        if isinstance(transformation, UsesCommonFeatures):
            transformation.retrieve_common_features(datasets, feature_space)
        Xs_true = transformation.transform(Xs_true)
        feature_space = feature_space and transformation.preserves_space

    for X_true, X_test in zip(Xs_true, Xs_test):
        np.testing.assert_array_almost_equal(X_true, X_test)


if __name__ == "__main__":
    test_layer_transformation_order()
