#!/usr/bin/env python3

from transmorph.datasets import load_test_datasets_small
from transmorph.engine.layers import LayerInput, LayerMatching
from transmorph.engine.matching import Labels, OT, GW, FusedGW, MNN
from transmorph.utils import anndata_manager as adm, AnnDataKeyIdentifiers

ALL_MATCHINGS = [
    (FusedGW, {}),
    (GW, {}),
    (Labels, {"label_obs": "class"}),
    (MNN, {"n_neighbors": 3}),
    (OT, {}),
]


def test_layer_matching():
    # Tests all types of matchings in a layer to ensure
    # all needed traits are initialized.
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
        matching_lay = matching_algo(**kwargs)
        lmatching = LayerMatching(matching=matching_lay)
        linput.connect(lmatching)
        linput.fit(datasets)
        lmatching.fit(datasets)


if __name__ == "__main__":
    test_layer_matching()
