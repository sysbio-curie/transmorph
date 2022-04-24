#!/usr/bin/env python3

from transmorph import settings
from transmorph.datasets import load_test_datasets_small
from transmorph.engine.checking import NeighborEntropy
from transmorph.engine.matching import Labels
from transmorph.engine.merging import Barycenter
from transmorph.engine.traits import UsesNeighbors, UsesReference
from transmorph.utils import anndata_manager as adm, AnnDataKeyIdentifiers


def test_checking_neighborentropy():
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
    settings.n_neighbors = 3
    UsesNeighbors.compute_neighbors_graphs(
        datasets=datasets,
        representation_key=AnnDataKeyIdentifiers.BaseRepresentation,
    )
    UsesReference.write_is_reference(datasets[1])
    matching = Labels(label_obs="class")
    matching.retrieve_labels(datasets)
    T = matching.fit([adata.X for adata in datasets])
    mg = Barycenter()
    mg.retrieve_reference_index(datasets)
    mg.set_matchings(T)
    Xs_out = mg.transform([adata.X for adata in datasets])
    check = NeighborEntropy()
    is_valid = check.check(Xs_out)
    assert check.score is not None and check.score > 0.16 and not is_valid


if __name__ == "__main__":
    test_checking_neighborentropy()
