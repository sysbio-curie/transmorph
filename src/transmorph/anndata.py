#!/usr/bin/env python3

from .transmorph import Transmorph
import anndata as ad
import numpy as np
import pandas as pd

def integrate_anndata(
        adata_qry,
        adata_ref,
        use_labels=False,
        key_labels=None,
        key_labels_ref_if_different=None,
        write_adata_qry=True,
        jitter_result=True,
        jitter_std=.01,
        **kwargs
):
    # Choosing labels
    if use_labels:
        assert key_labels in adata_qry.obs,\
            f"{key_labels} not in adata_qry.obs."
        ref_labels = (
            key_labels if key_labels_ref_if_different is None
            else key_labels_ref_if_different
        )
        assert ref_labels in adata_ref.obs,\
            f"{ref_labels} not in adata_ref.obs."
        X_labels = adata_qry.obs[key_labels]
        Y_labels = adata_ref.obs[ref_labels]

    # Choosing the right pipeline
    method = "ot"
    if "method" in kwargs:
        method = kwargs["method"]

    # Computing X and Y depending on the matrix
    if method == "ot":
        genes_final_space = adata_qry.var_names.intersection(adata_ref.var_names)
        if len(genes_final_space) < 100:
            print("Warning: Few genes (<100) in common between adatas. "\
                  "Integration results may be inaccurate.")
        X = adata_qry[:, genes_final_space].X
        Y = adata_ref[:, genes_final_space].X
    elif method == "gromov":
        genes_final_space = adata_ref.var_names
        X, Y = adata_qry.X, adata_ref.X
    else:
        raise ValueError(f"Invalid method: {method}.")

    # Integration
    transmorph = Transmorph(**kwargs)
    X_integrated = transmorph.fit_transform(
        X,
        Y,
        xs_labels=X_labels if use_labels else None,
        yt_labels=Y_labels if use_labels else None,
        jitter=jitter_result,
        jitter_std=jitter_std
    )

    # Returns
    if write_adata_qry:
        adata_qry.obsm['trph_X'] = X_integrated
    adata_final = ad.AnnData(
        np.concatenate( (X_integrated, Y) ),
        pd.concat( (adata_qry.obs, adata_ref.obs) ),
        adata_ref[:,genes_final_space].var # May be done cleaner
    )

    # Labeling cells by dataset of origin
    adata_final.obs['trph_origin'] = (
        ['qry']*len(X_integrated) +
        ['ref']*len(Y)
    )
    return adata_final
