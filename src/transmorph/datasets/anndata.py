from typing import Dict

import scanpy as sc


def preprocess_anndata(
    adata: sc.AnnData, parameters_matching: Dict = {}, parameters_merging: Dict = {}
) -> sc.AnnData:
    adata.uns["_transmorph"] = {
        "matching": parameters_matching,
        "merging": parameters_merging,
    }
    return adata
