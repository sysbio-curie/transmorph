#!/usr/bin/env python3

from transmorph import settings
from transmorph.datasets import load_test_datasets_small
from transmorph.engine.layers import (
    LayerInput,
    LayerChecking,
    LayerOutput,
)
from transmorph.engine.checking import NeighborEntropy
from transmorph.engine.traits import IsRepresentable, UsesNeighbors
from transmorph.utils import anndata_manager as adm


def test_layer_checking_simple():
    # Tests a very simple network
    # with two branches.
    settings.n_neighbors = 3
    datasets = list(load_test_datasets_small().values())
    UsesNeighbors.compute_neighbors_graphs(datasets)
    linput = LayerInput()
    lchecking = LayerChecking(checking=NeighborEntropy())
    louty = LayerOutput()
    loutn = LayerOutput()
    linput.connect(lchecking)
    lchecking.connect(louty)
    lchecking.connect_rejected(loutn)
    linput.fit(datasets)
    lchecking.fit(datasets)
    louty.fit(datasets)
    # We pretend we received the "yes" layer
    IsRepresentable.assert_representation_equals([linput, louty], datasets)
    UsesNeighbors.reset()
    adm.clean(datasets)


if __name__ == "__main__":
    test_layer_checking_simple()
