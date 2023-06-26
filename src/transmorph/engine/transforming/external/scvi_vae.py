#!/usr/bin/env python3

from typing import List, Optional

import anndata as ad
import numpy as np

from ..transformation import Transformation
from ....utils.matrix import extract_chunks


class VAEscvi(Transformation):
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
        **kwargs,
    ):
        Transformation.__init__(
            self,
            str_identifier="VAEscvi",
            preserves_space=False,
        )
        self.use_anndata_layer = use_anndata_layer
        self.n_layers = n_layers
        self.n_latent = n_latent
        self.kwargs = kwargs

    def transform(
        self,
        datasets: List[ad.AnnData],
        embeddings: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Runs the VAE and returns the embedding result.
        """
        try:
            import scvi
        except ImportError:
            raise ImportError(
                "scvi is not installed. You can install it with !pip \
                    install scvi-tools."
            )

        if self.use_anndata_layer is None:
            assert all(
                np.array_equal(adata.X.astype(int), adata.X) for adata in datasets
            ), "scVI model needs raw counts. Please provide an AnnData with \
                raw counts, or specify a layer with raw counts into \
                'use_adata_layer'."
            self.use_anndata_layer = "raw_tmp"
            for adata in datasets:
                adata.layers["raw_tmp"] = adata.X.copy()

        adata_all = ad.concat(datasets, label="batch")
        adata_all.obs_names_make_unique()

        scvi.model.SCVI.setup_anndata(
            adata_all,
            layer=self.use_anndata_layer,
            batch_key="batch",
        )
        vae = scvi.model.SCVI(
            adata_all,
            n_layers=self.n_layers,
            n_latent=self.n_latent,
            **self.kwargs,
        )
        vae.train()

        if self.use_anndata_layer == "raw_tmp":
            del adata.layers["raw_tmp"]

        return extract_chunks(
            vae.get_latent_representation(),
            [adata.n_obs for adata in datasets],
        )
