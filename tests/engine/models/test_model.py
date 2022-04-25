#!/usr/bin/env python3

from transmorph.datasets import load_test_datasets_small
from transmorph.engine.checking import NeighborEntropy
from transmorph.engine.layers import (
    LayerInput,
    LayerMatching,
    LayerMerging,
    LayerChecking,
    LayerOutput,
)
from transmorph.engine.matching import Labels
from transmorph.engine.merging import Barycenter
from transmorph.engine.model import Model


def test_model_initialization():
    # Tests the initialization of a simple model
    # TODO: maybe also test a bigger model?
    linput = LayerInput()
    lmatching = LayerMatching(Labels("class"))
    lmerging = LayerMerging(Barycenter())
    lchecking = LayerChecking(NeighborEntropy())
    loutput = LayerOutput()

    linput.connect(lmatching)
    lmatching.connect(lmerging)
    lmerging.connect(lchecking)
    lchecking.connect(loutput)
    lchecking.connect_rejected(lmatching)

    model = Model(input_layer=linput)
    exp_layers = [linput, lmatching, lmerging, lchecking, loutput]
    assert model.input_layer is linput
    assert len(model.output_layers) == 1
    assert model.output_layers[0] is loutput
    for layer, layer_exp in zip(model.layers, exp_layers):
        assert layer is layer_exp


def test_model_simple():
    # Tests a simple match -> merge model on a
    # toy dataset.
    datasets = list(load_test_datasets_small().values())
    linput = LayerInput()
    lmatching = LayerMatching(Labels("class"))
    lmerging = LayerMerging(Barycenter())
    loutput = LayerOutput()
    linput.connect(lmatching)
    lmatching.connect(lmerging)
    lmerging.connect(loutput)
    model = Model(linput)
    model.fit(datasets, reference=datasets[1])


if __name__ == "__main__":
    test_model_simple()
