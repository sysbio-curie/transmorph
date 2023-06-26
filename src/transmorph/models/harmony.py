#!/usr/bin/env python3

from anndata import AnnData
from typing import Dict, List, Optional

from transmorph.engine import Model
from transmorph.engine.layers import (
    LayerInput,
    LayerTransformation,
    LayerOutput,
)
from transmorph.engine.transforming import CommonFeatures, PCA, Harmony as TrHarmony


class Harmony(Model):
    """
    Harmony algorithm, presented in [Korsunky 2019]. We use here the harmonypy
    implementation as a backend.

    Parameters
    ----------
    n_components: int, default = 30
        Number of principal components to use.

    verbose: bool, default = True
        Verbose (seems like all output cannot be removed from harmonypy)

    **kwargs: Parameters to be passed to scanpy.external.harmony.
    See at
        https://scanpy.readthedocs.io/en/stable/generated/
        scanpy.external.pp.harmony_integrate.html

    Reference
    ---------

    [Korsunky 2019] https://www.nature.com/articles/s41592-019-0619-0
    """

    def __init__(self, n_components: int = 30, verbose: bool = True, **kwargs) -> None:
        from .. import settings

        if verbose:
            settings.verbose = "INFO"
        else:
            settings.verbose = "WARNING"

        # Initializing layers
        linput = LayerInput()

        ltransform = LayerTransformation()
        ltransform.add_transformation(CommonFeatures())
        ltransform.add_transformation(PCA(n_components=n_components))
        ltransform.add_transformation(TrHarmony(verbose=False, **kwargs))

        loutput = LayerOutput()

        # Building model
        linput.connect(ltransform)
        ltransform.connect(loutput)

        Model.__init__(self, input_layer=linput, str_identifier="HARMONY")

    def transform(
        self,
        datasets: List[AnnData],
        use_representation: Optional[str] = None,
        output_representation: Optional[str] = None,
    ) -> None:
        """
        Carries out the model on a list of AnnData objects. Writes the result in
        .obsm fields.

        Parameters
        ----------
        datasets: List[AnnData]
            List of anndata objects, must have at least one common feature.

        use_representation: Optional[str]
            .obsm to use as input.

        output_representation: str
            .obsm destination key, "transmorph" by default.
        """
        if isinstance(datasets, Dict):
            datasets = list(datasets.values())
        self.fit(
            datasets,
            use_representation=use_representation,
            output_representation=output_representation,
        )
