#!/usr/bin/env python3

import numpy as np

from transmorph.datasets import load_test_datasets_small
from transmorph.engine.layers import LayerInput, LayerMatching
from transmorph.engine.matching import Matching, Labels, OT, GW, FusedGW, MNN
from transmorph.engine.traits import HasMetadata, UsesSampleLabels
from transmorph.utils import anndata_manager as adm, AnnDataKeyIdentifiers

ALL_MATCHINGS = [
    # constructor, parameters
    (FusedGW, {}),
    (GW, {}),
    (Labels, {"label_obs": "class"}),
    (MNN, {"n_neighbors": 3}),
    (OT, {}),
]


def test_layer_matching():
    # Tests all types of matchings in a layer to ensure
    # all needed traits are initialized.
    # We trust matching tests to ensure correctness of
    # Matchings.
    datasets = list(load_test_datasets_small().values())
    for adata in datasets:
        adm.set_value(
            adata=adata,
            key=AnnDataKeyIdentifiers.BaseRepresentation,
            field="obsm",
            value=adata.X,
            persist="pipeline",
        )
    for matching_algo, kwargs in ALL_MATCHINGS:
        linput = LayerInput()
        matching_lay: Matching = matching_algo(**kwargs)
        lmatching = LayerMatching(matching=matching_lay)
        linput.connect(lmatching)
        linput.fit(datasets)
        lmatching.fit(datasets)

        matching_ref: Matching = matching_algo(**kwargs)
        if isinstance(matching_ref, HasMetadata):
            matching_ref.retrieve_all_metadata(datasets)
        if isinstance(matching_ref, UsesSampleLabels):
            matching_ref.retrieve_labels(datasets)
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


if __name__ == "__main__":
    test_layer_matching()
