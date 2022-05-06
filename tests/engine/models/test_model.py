#!/usr/bin/env python3

from transmorph.datasets import load_test_datasets_small, load_travaglini_10x
from transmorph.engine.checking import NeighborEntropy
from transmorph.engine.layers import (
    LayerInput,
    LayerMatching,
    LayerMerging,
    LayerChecking,
    LayerOutput,
)
from transmorph.engine.layers.layertransformation import LayerTransformation
from transmorph.engine.matching import Labels, MNN
from transmorph.engine.merging import Barycenter, LinearCorrection, GraphEmbedding
from transmorph.engine.model import Model
from transmorph.engine.transforming import Standardize, PCA, CommonFeatures
from transmorph.utils.plotting import scatter_plot


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


def test_model_smalldata_simple():
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
    testname = "smalldata_simple"
    scatter_plot(
        datasets,
        color_by="class",
        title=testname,
        xlabel="Feature 1",
        ylabel="Feature 2",
        save=True,
        show=False,
        caller_path=__file__,
        suffix=testname,
    )


def test_model_smalldata_complex():
    # Tests a more complex model with several
    # transformations on a small dataset
    datasets = list(load_test_datasets_small().values())
    linput = LayerInput()
    lpreprocess = LayerTransformation()
    lpreprocess.add_transformation(Standardize())
    lmatching = LayerMatching(Labels("class"))
    lmerging = LayerMerging(LinearCorrection())
    lchecking = LayerChecking(NeighborEntropy(), n_checks_max=2)
    loutput = LayerOutput()
    linput.connect(lpreprocess)
    lpreprocess.connect(lmatching)
    lmatching.connect(lmerging)
    lmerging.connect(lchecking)
    lchecking.connect(loutput)
    lchecking.connect_rejected(lmatching)
    model = Model(linput)
    model.fit(datasets, reference=datasets[1])
    testname = "smalldata_complex"
    scatter_plot(
        datasets,
        color_by="class",
        title=testname,
        xlabel="Feature 1",
        ylabel="Feature 2",
        save=True,
        show=False,
        caller_path=__file__,
        suffix=testname,
    )


def test_model_largedata_simple():
    # Tests a simple match -> merge model on a
    # data bank with few, large datasets. This allows to
    # test integration with subsampling.
    datasets = list(load_travaglini_10x().values())
    linput = LayerInput()
    ltransformation = LayerTransformation()
    ltransformation.add_transformation(CommonFeatures())
    ltransformation.add_transformation(Standardize(center=True, scale=True))
    ltransformation.add_transformation(PCA(n_components=30))
    lmatching = LayerMatching(MNN(n_neighbors=30))
    lmerging = LayerMerging(GraphEmbedding(n_neighbors=10, matching_strength=0.8))
    loutput = LayerOutput()
    linput.connect(ltransformation)
    ltransformation.connect(lmatching)
    lmatching.connect(lmerging)
    lmerging.connect(loutput)
    model = Model(linput)
    model.fit(datasets)
    testname = "largedatasets_simple"
    scatter_plot(
        datasets,
        color_by="class",
        title=testname,
        xlabel="UMAP1",
        ylabel="UMAP2",
        save=True,
        show=False,
        caller_path=__file__,
        suffix=f"{testname}_class",
        use_cache=True,
    )
    scatter_plot(
        datasets,
        title=testname,
        xlabel="UMAP1",
        ylabel="UMAP2",
        save=True,
        show=False,
        caller_path=__file__,
        suffix=f"{testname}_dataset",
        use_cache=True,
    )


if __name__ == "__main__":
    test_model_largedata_simple()
