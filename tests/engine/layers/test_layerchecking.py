#!/usr/bin/env python3

from transmorph.datasets import load_test_datasets_small
from transmorph.engine.layers import (
    LayerInput,
    LayerChecking,
    LayerMatching,
    LayerMerging,
    LayerOutput,
)
from transmorph.engine.checking import NeighborEntropy
from transmorph.engine.matching import Labels
from transmorph.engine.merging import LinearCorrection
from transmorph.engine.traits import IsRepresentable, UsesNeighbors, UsesReference
from transmorph.utils import anndata_manager as adm

N_STEPS_MAX = 100
N_CHECKS_MAX = 5


def test_layer_checking_simple():
    # Tests a very simple network with two branches
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


def test_layer_checking_standard():
    # Tests a standard model with a single loop.
    datasets = list(load_test_datasets_small().values())
    UsesNeighbors.compute_neighbors_graphs(datasets)
    UsesReference.write_is_reference(datasets[1])
    linput = LayerInput()
    lmatching = LayerMatching(matching=Labels("class"))
    lmerging = LayerMerging(merging=LinearCorrection())
    lchecking = LayerChecking(
        checking=NeighborEntropy(threshold=1.0),
        n_checks_max=N_CHECKS_MAX,
    )
    lout = LayerOutput()
    linput.connect(lmatching)
    lmatching.connect(lmerging)
    lmerging.connect(lchecking)
    lchecking.connect(lout)
    lchecking.connect_rejected(lmatching)

    current_layer = linput
    n_checks, n_steps = 0, 0
    while current_layer is not None:
        n_steps += 1
        if n_steps >= N_STEPS_MAX:
            break
        next_layers = current_layer.fit(datasets)
        if len(next_layers) == 0:
            next_layer = None
        else:
            assert len(next_layers) == 1
            next_layer = next_layers[0]
        if current_layer is lchecking:
            n_checks += 1
            if next_layer is lmatching:
                next_layer.called_by_checking = True
        if current_layer is lmatching:
            current_layer.called_by_checking = False
        current_layer = next_layer

    assert n_checks == N_CHECKS_MAX
    assert n_steps < N_STEPS_MAX


if __name__ == "__main__":
    test_layer_checking_standard()
