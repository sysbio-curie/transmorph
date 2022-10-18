#!/usr/bin/env python3
# Contains high level functions to load datasets.

# TODO: process databanks before instead of on call

import anndata as ad
import gc
import logging
import numpy as np
import os
import scanpy as sc

from anndata import AnnData
from os.path import dirname
from scipy.sparse import csr_matrix, load_npz
from typing import Dict, Optional

from .databank_api import check_files, download_dataset, remove_dataset, unzip_file
from .._logging import logger
from ..utils.misc import generate_anndata, generate_str_elements

# Datasets are stored in data/ subdirectory.
DPATH_DATASETS = dirname(__file__) + "/data/"
AVAILABLE_BANKS = ["chen_10x", "pal_10x", "travaglini_10x", "zhou_10x", "cell_cycle"]


def load_dataset(source, filename, is_sparse=False) -> np.ndarray:
    """
    Loads a dataset and returns it as a numpy array.
    """
    if not is_sparse:
        return np.loadtxt(source + filename, delimiter=",")
    return load_npz(source + filename).toarray()


def load_test_datasets_random(
    n_adata: int = 5, n_obs: int = 100, n_var: int = 20
) -> Dict[str, AnnData]:
    """
    Loads a handful of random datasets for testing purposes.
    """
    commonf = generate_str_elements(int(n_var * 0.5))
    return {
        f"batch_{i}": generate_anndata(
            n_obs,
            np.concatenate(
                (commonf, generate_str_elements(n_var - commonf.shape[0])),
                axis=0,
            ),
        )
        for i in range(n_adata)
    }


def load_test_datasets_small() -> Dict[str, AnnData]:
    """
    Loads a small hand-crafted dataset for testing purposes.
    Samples are two dimensional, and are labeled by the
    observation "class" with two options, 0 and 1.
    """
    x1 = np.array(
        [
            # Upper cluster
            [1, 6],
            [2, 5],
            [3, 4],
            [3, 6],
            [4, 5],
            [2, 4],
            # Lower cluster
            [2, 0],
            [1, 2],
            [2, 2],
            [3, 0],
        ],
        dtype=np.float32,
    )
    a1 = ad.AnnData(x1, dtype=x1.dtype)
    a1.obs["class"] = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]

    x2 = np.array(
        [
            # Upper cluster
            [6, 5],
            [7, 5],
            [9, 6],
            [7, 6],
            # Lower cluster
            [8, 1],
            [6, 2],
            [7, 2],
            [7, 0],
            [7, 1],
        ],
        dtype=np.float32,
    )
    a2 = ad.AnnData(x2, dtype=x2.dtype)
    a2.obs["class"] = [0, 0, 0, 0, 1, 1, 1, 1, 1]
    return {"src": a1, "ref": a2}


def load_spirals():
    """
    Loads a pair of spiraling datasets of small/medium size, for
    testing purposes. Samples are embedded in a 3D space, and are
    labeled by a continuous observation "label".
    """
    xs = load_dataset(DPATH_DATASETS, "spirals/spiralA.csv")
    ys = load_dataset(DPATH_DATASETS, "spirals/spiralA_labels.csv")
    adata_s = ad.AnnData(xs, dtype=xs.dtype)
    adata_s.obs["label"] = ys

    xt = load_dataset(DPATH_DATASETS, "spirals/spiralB.csv")
    yt = load_dataset(DPATH_DATASETS, "spirals/spiralB_labels.csv")
    adata_t = ad.AnnData(xt, dtype=xt.dtype)
    adata_t.obs["label"] = yt

    return {"src": adata_s, "ref": adata_t}


def load_cell_cycle() -> Dict[str, AnnData]:
    """
    Loads three cell cycle datasets for label transfer purposes.

    TODO
    """
    return load_bank("cell_cycle")


def load_chen_10x(
    keep_sparse: bool = False,
    keep_scanpy_leftovers: bool = False,
    cache_databank: bool = True,
) -> Dict[str, AnnData]:
    """
    Loads 14 nasopharyngal carcinoma 10X datasets, gathered in  [Chen 2020].
    These datasets have been downloaded from Curated Cancer Cell atlas
    [3CA], and prepared by us to be transmorph ready. There are 14
    batches of a few thousands cells each, for a total of 71,896 cells.

    Each of these batches is expressed in the space of its 10,000 most
    variable genes, with pooling, log1p and scaling transformation.

    The observation 'class' contains inferred cell type. Existing cell
    types are the 14 following,
    [
        nan, 'Lymphovascular', 'Endothelial', 'B cell', 'Dendritic',
        'NK_cell', 'Plasma', 'Epithelial', 'Macrophage', 'Malignant',
        'Mast', 'Myofibroblast', 'Fibroblast', 'T cell'
    ].

    [Chen 2020] Chen, Y. P., Yin, J. H., Li, W. F., Li, H. J., Chen,
                D. P., Zhang, C. J., ... & Ma, J. (2020). Single-cell
                transcriptomics reveals regulators underlying immune
                cell diversity and immune subtypes associated with
                prognosis in nasopharyngeal carcinoma. Cell research,
                30(11), 1024-1042.
                https://www.nature.com/articles/s41422-020-0374-x

    [3CA]       https://www.weizmann.ac.il/sites/3CA/head-and-neck

    Parameters
    ----------
    keep_sparse: bool, default = False
        Keeps AnnData.X at csr_matrix format. Note it will be densified
        if it goes through a transmorph model.

    keep_scanpy_leftovers: bool, default = False
        Set it to true if you do not want the function to clean leftovers
        from scanpy preprocessing for some reason.

    cache_databank: bool, default = True
        Keeps a local copy of the data bank to avoid downloading it each
        time. Data bank size: 3.4G
    """
    datasets = load_bank("chen_10x", keep_sparse=keep_sparse)

    # Formatting adata properly, removing scanpy leftovers
    for key, adata in datasets.items():
        adata.obs["class"] = adata.obs["cell_type"]
        cl = adata.obs["class"]
        adata.obs["class"] = cl.cat.add_categories("n/a").fillna("n/a")
        # Cleaning artifactual labels
        adata = adata[adata.obs["class"] != "n/a", :]
        adata = adata[adata.obs["class"] != "Mast", :]
        adata = adata[adata.obs["class"] != "Myofibroblast", :]
        adata = adata[adata.obs["class"] != "Lymphovascular", :]
        adata = adata[adata.obs["class"] != "Fibroblast", :]
        adata = adata[adata.obs["class"] != "Dendritic", :]
        datasets[key] = adata
        if keep_scanpy_leftovers:
            continue
        del adata.obs["sample"]
        del adata.obs["cell_type"]
        del adata.obs["ebv"]
        del adata.var["highly_variable"]
        del adata.var["means"]
        del adata.var["dispersions"]
        del adata.var["dispersions_norm"]
        del adata.var["mean"]
        del adata.var["std"]
        del adata.uns["hvg"]
        del adata.uns["log1p"]
        del adata.uns["pca"]
        del adata.obsm["X_pca"]
        del adata.varm["PCs"]

    if not cache_databank:
        remove_bank("chen_10x")

    return datasets


def load_pal_10x(
    keep_sparse: bool = False,
    keep_scanpy_leftovers: bool = False,
    cache_databank: bool = True,
) -> Dict[str, AnnData]:
    """
    Loads 5 breast cancer 10X datasets, gathered in  [Pal 2021]. These datasets
    have been downloaded from Curated Cancer Cell atlas [3CA], and prepared
    by us to be transmorph ready. There are 5 batches of a few tens of
    thousands cells each, for a total of 130,258 cells.

    Each of these batches is expressed in the space of its 10,000 most
    variable genes, with pooling, log1p and scaling transformation.

    The observation 'class_type' contains inferred cell type. Existing cell
    types are the 8 following,
    [
        'Epithelial', 'Endothelial', 'Pericyte', 'Myeloid', 'Unknown',
        'Fibroblast', 'Lymphiod', 'Normal epithelium'
    ].

    The boolean observation 'class_iscancer' indicates if each cell has
    been inferred as malignant or not.

    [Pal 2021] Pal, B., Chen, Y., Vaillant, F., Capaldo, B. D., Joyce,
               R., Song, X., ... & Visvader, J. E. (2021). A single‐cell
               RNA expression atlas of normal, preneoplastic and tumorigenic
               states in the human breast. The EMBO journal, 40(11), e107333.

               https://embopress.org/doi/full/10.15252/embj.2020107333

    [3CA]      https://www.weizmann.ac.il/sites/3CA/head-and-neck

    Parameters
    ----------
    keep_sparse: bool, default = False
        Keeps AnnData.X at csr_matrix format. Note it will be densified
        if it goes through a transmorph model.

    keep_scanpy_leftovers: bool, default = False
        Set it to true if you do not want the function to clean leftovers
        from scanpy preprocessing for some reason.

    cache_databank: bool, default = True
        Saves the data bank on the HDD to avoid downloading it each
        time. Data bank size: 6.2G
    """
    datasets = load_bank("pal_10x", keep_sparse=keep_sparse)

    # Formatting adata properly, removing scanpy leftovers
    for key in datasets:
        adata = datasets[key]
        adata.obs["class_type"] = adata.obs["Annotation"]
        adata.obs["class_iscancer"] = adata.obs["Is_Cancer"]
        adata = adata[adata.obs["class_type"] != "Unknown", :].copy()
        adata.obs["class_type"] = adata.obs["class_type"].replace(
            "Normal epithelium", "Epithelial"
        )
        adata.obs["class_type"] = adata.obs["class_type"].replace(
            "Lymphiod", "Lymphoid"
        )
        datasets[key] = adata
        if keep_scanpy_leftovers:
            continue
        del adata.obs["Sample"]
        del adata.obs["Annotation"]
        del adata.obs["Is_Cancer"]
        del adata.var["highly_variable"]
        del adata.var["means"]
        del adata.var["dispersions"]
        del adata.var["dispersions_norm"]
        del adata.var["mean"]
        del adata.var["std"]
        del adata.uns["hvg"]
        del adata.uns["log1p"]
        del adata.uns["pca"]
        del adata.obsm["X_pca"]
        del adata.varm["PCs"]

    if not cache_databank:
        remove_bank("pal_10x")

    return datasets


def load_travaglini_10x(
    keep_sparse: bool = False,
    keep_scanpy_leftovers: bool = False,
    cache_databank: bool = True,
) -> Dict[str, AnnData]:
    """
    Loads 3 10X lung datasets, gathered in  [Travaglini 2020]. These datasets
    have been prepared by us to be transmorph ready. There are 3 batches
    of a few tens of thousands cells each, for a total of 65,662 cells.

    Each of these batches is expressed in the space of its 10,000 most
    variable genes, with pooling, log1p and scaling transformation.

    The observation 'class' contains inferred cell compartment. Existing
    cell compartments are the 4 following,
    ['epithelial', 'endothelial', 'stromal', 'immune'].

    [Travaglini 2020] Travaglini, Kyle J., et al. "A molecular cell atlas
                      of the human lung from single-cell RNA sequencing."
                      Nature 587.7835 (2020): 619-625.

                      https://www.nature.com/articles/s41586-020-2922-4

    Parameters
    ----------
    keep_sparse: bool, default = False
        Keeps AnnData.X at csr_matrix format. Note it will be densified
        if it goes through a transmorph model.

    keep_scanpy_leftovers: bool, default = False
        Set it to true if you do not want the function to clean leftovers
        from scanpy preprocessing for some reason.

    cache_databank: bool, default = True
        Saves the data bank on the HDD to avoid downloading it each
        time. Data bank size: 389M
    """
    datasets = load_bank("travaglini_10x", keep_sparse=keep_sparse)

    # Formatting adata properly, removing scanpy leftovers
    for adata in datasets.values():
        adata.obs["class"] = adata.obs["compartment"]
        if keep_scanpy_leftovers:
            continue
        del adata.obs["nGene"]
        del adata.obs["patient"]
        del adata.obs["sample"]
        del adata.obs["location"]
        del adata.obs["compartment"]
        del adata.obs["magnetic.selection"]
        del adata.obs["nUMI"]
        del adata.obs["orig.ident"]
        del adata.obs["channel"]
        del adata.obs["tissue"]
        del adata.obs["region"]
        del adata.obs["percent.ribo"]
        del adata.obs["free_annotation"]
        del adata.obs["preparation.site"]
        del adata.var["highly_variable"]
        del adata.var["means"]
        del adata.var["dispersions"]
        del adata.var["dispersions_norm"]
        del adata.uns["hvg"]
        del adata.obsm["X_Compartment_tSNE"]
        del adata.obsm["X_tSNE"]

    if not cache_databank:
        remove_bank("travaglini_10x")

    return datasets


def load_zhou_10x(
    keep_sparse: bool = False,
    keep_scanpy_leftovers: bool = False,
    cache_databank: bool = True,
) -> Dict[str, AnnData]:
    """
    Loads 11 10X osteosarcoma datasets, gathered in  [Zhou 2020]. These datasets
    have been downloaded from Curated Cancer Cell atlas [3CA], and prepared
    by us to be transmorph ready. There are 11 batches of a few
    thousands cells each, for a total of 64,557 cells.

    Each of these batches is expressed in the space of its 10,000 most
    variable genes, with pooling, log1p and scaling transformation.

    The observation 'class_type' contains inferred cell compartment. Existing
    cell compartments are the 11 following,
    [
        'Osteoblast', 'Osteoclast', 'Myeloid', 'Osteoblast_proli',
        'Pericyte', 'Fibroblast', 'Chondrocyte', 'T_cell', 'MSC',
        'Myoblast', 'Endothelial'
    ]

    The boolean observation 'class_iscancer' indicates if each cell has
    been inferred as malignant or not.

    [Zhou 2020] Zhou, Yan, et al. "Single-cell RNA landscape of intratumoral
                heterogeneity and immunosuppressive microenvironment in advanced
                osteosarcoma." Nature communications 11.1 (2020): 1-17.

                https://www.nature.com/articles/s41467-020-20059-6

    [3CA]       https://www.weizmann.ac.il/sites/3CA/head-and-neck

    Parameters
    ----------
    keep_sparse: bool, default = False
        Keeps AnnData.X at csr_matrix format. Note it will be densified
        if it goes through a transmorph model.

    keep_scanpy_leftovers: bool, default = False
        Set it to true if you do not want the function to clean leftovers
        from scanpy preprocessing for some reason.

    cache_databank: bool, default = True
        Saves the data bank on the HDD to avoid downloading it each
        time. Data bank size: 3.1G
    """
    datasets = load_bank("zhou_10x", keep_sparse=keep_sparse)

    # Formatting adata properly, removing scanpy leftovers
    for adata in datasets.values():
        adata.obs["class_type"] = adata.obs["cell_type"]
        adata.obs["class_type"] = adata.obs["class_type"].replace(
            "Osteoblast_proli",
            "Osteoblast",
        )
        adata.obs["class_iscancer"] = adata.obs["malignant"]
        if keep_scanpy_leftovers:
            continue
        del adata.obs["sample"]
        del adata.obs["cell_type"]
        del adata.obs["malignant"]
        del adata.obs["n_genes"]
        del adata.var["n_counts"]
        del adata.var["highly_variable"]
        del adata.var["means"]
        del adata.var["dispersions"]
        del adata.var["dispersions_norm"]
        del adata.var["mean"]
        del adata.var["std"]
        del adata.uns["hvg"]
        del adata.uns["log1p"]
        del adata.uns["pca"]
        del adata.obsm["X_pca"]
        del adata.varm["PCs"]

    if not cache_databank:
        remove_bank("zhou_10x")

    return datasets


def load_bank(
    dataset_name: str,
    keep_sparse: bool = False,
    n_samples: Optional[int] = None,
):
    """
    Parameters
    ----------
    dataset_name: str
        "name" value in the json file for the bank of datasets.

    keep_sparse: bool, default = False
        Prevents AnnData.X to be converted to ndarray.

    n_samples: Optional[int], default = None
        If set and lesser than the total number of samples, selects at random
        a subset of them with an average size of n_samples.
    """
    assert (
        dataset_name in AVAILABLE_BANKS
    ), f"Unknown bank: {dataset_name}. Available banks are {', '.join(AVAILABLE_BANKS)}"
    logger.log(logging.INFO, f"databank_api > Loading bank {dataset_name}.")
    download_needed = not check_files(dataset_name)
    if download_needed:
        zip_path = download_dataset(dataset_name)
        unzip_file(zip_path, dataset_name)
        check_files(dataset_name)
    dataset_root = DPATH_DATASETS + f"{dataset_name}/"
    datasets = {}
    # Loading datasets
    for fname in os.listdir(dataset_root):
        adata = sc.read_h5ad(dataset_root + fname)
        if not keep_sparse and isinstance(adata.X, csr_matrix):
            logger.log(
                logging.DEBUG,
                f"databank_api > Removing sparsity of {dataset_name}.",
            )
            adata.X = adata.X.toarray()
        pid = fname.split(".")[0]
        datasets[pid] = adata

    logger.log(logging.INFO, f"databank_api > Bank {dataset_name} successfully loaded.")

    # Subsampling if necessary
    if n_samples is None:
        return datasets

    total_samples = sum(adata.n_obs for adata in datasets.values())
    if total_samples <= n_samples:
        return datasets

    frequency = n_samples / total_samples
    for key in datasets:
        adata = datasets[key]
        datasets[key] = adata[np.random.random(adata.n_obs) < frequency, :].copy()

    gc.collect()

    return datasets


def remove_bank(dataset_name: str):
    """
    Removes a data banks from local storage.
    """
    assert (
        dataset_name in AVAILABLE_BANKS
    ), f"Unknown bank: {dataset_name}. Available banks are {', '.join(AVAILABLE_BANKS)}"
    remove_dataset(dataset_name)
