#!/usr/bin/env python3

from anndata import AnnData
from typing import Dict, List, Optional

from transmorph.engine import Model
from transmorph.engine.layers import (
    LayerInput,
    LayerTransformation,
    LayerOutput,
)
from transmorph.engine.transforming import VAEscvi


class VAE(Model):
    """
    Wraps variational autoencoder-based data integration introduced in
    scvi [Gayoso2022]. These VAEs use raw, integer counts as an input.

    Parameters
    ----------
    use_anndata_layer: Optional[str], default = None
        Key of AnnData.layers containing raw counts. If no key is
        specified, VAEscvi will attempt to use the .X matrix as an input.

    n_layers: int, default = 2
        Number of hidden layers used for encoder and decoder NNs.

    n_latent: int, default = 30
        Number of neurons in the latent space, also defines output dimension.

    verbose: bool, default = True
        Verbose

    **kwargs
        Additional arguments to be passed to scvi.model.SCVI, see at
        https://docs.scvi-tools.org/en/0.9.1/api/reference/scvi.model.SCVI.html

    References
    ----------
    [Gayoso2022] https://www.nature.com/articles/s41587-021-01206-w
    """

    def __init__(
        self,
        use_anndata_layer: Optional[str] = None,
        n_layers: int = 2,
        n_latent: int = 30,
        verbose: bool = True,
        **kwargs
    ) -> None:
        from .. import settings

        if verbose:
            settings.verbose = "INFO"
        else:
            settings.verbose = "WARNING"

        # Initializing layers
        linput = LayerInput()

        ltransform = LayerTransformation()
        ltransform.add_transformation(
            VAEscvi(
                use_anndata_layer=use_anndata_layer,
                n_latent=n_latent,
                n_layers=n_layers,
                **kwargs
            )
        )

        loutput = LayerOutput()

        # Building model
        linput.connect(ltransform)
        ltransform.connect(loutput)

        Model.__init__(self, input_layer=linput, str_identifier="VAEscvi")

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
