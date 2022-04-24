#!/usr/bin/env python3
# Contains high level functions to load datasets.

import anndata as ad
import logging
import numpy as np
import os
import scanpy as sc

from os.path import dirname
from scipy.sparse import load_npz
from typing import Dict, Optional

from .databank_api import check_files, download_dataset, remove_dataset, unzip_file
from .._logging import logger

# GIT: small datasets, can be hosted on Git
# ONLINE: bigger datasets, are downloaded if necessary
#   by the transmorph http API.
DPATH_DATASETS = dirname(__file__) + "/data/"


def load_dataset(source, filename, is_sparse=False) -> np.ndarray:
    """
    Loads a dataset and returns it as a numpy array.
    """
    if not is_sparse:
        return np.loadtxt(source + filename, delimiter=",")
    return load_npz(source + filename).toarray()


def load_test_datasets_small() -> Dict:
    """
    Loads a small hand-crafted dataset for testing purposes.

    Dataset
    -------
    - Number of datasets: 2
    - Embedding dimension: 2
    - Sizes: (10,2) and (9,2)
    - Number of labels: 2
    - Number of clusters: 2 per dataset

    Format
    ------
    {
        "src": AnnData(obs: "class"),
        "ref": AnnData(obs: "class"),
        "errors": np.array[i,j] = class_i != class_j
    }
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
            [7, 2],
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
    testing purposes.

    Dataset
    -------
    - Number of datasets: 2
    - Embedding dimension: 3
    - Sizes: (433,3) and (663,3)
    - Continuous labels

    Format
    ------
    {
        "src": AnnData(obs: "label"),
        "ref": AnnData(obs: "label")
    }
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


def load_chen_10x():
    """
    Dataset
    -------
    - Number of datasets: 14
    - Embedding dimension: 10000
    - Number of cell types: ? TODO
    """
    return load_bank("chen_10x")


def load_pal_10x():
    """
    Dataset
    -------
    - Number of datasets: 14
    - Embedding dimension: 10000
    - Number of cell types: ? TODO
    """
    return load_bank("pal_10x")


def load_travaglini_10x():
    """
    Dataset
    -------
    - Number of datasets: 3
    - Embedding dimension: 10000
    - Number of labels: 4
    """
    return load_bank("travaglini_10x")


def load_zhou_10x():
    """
    Dataset
    -------
    - Number of datasets: 14
    - Embedding dimension: 10000
    - Number of cell types: ? TODO
    """
    return load_bank("zhou_10x")


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
        if not keep_sparse:
            logger.log(
                logging.DEBUG,
                f"databank_api > Removing sparsity of {dataset_name}.",
            )
            adata.X = adata.X.toarray()
        pid = fname.split(".")[0]
        datasets[pid] = adata

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
    return datasets


def remove_bank(dataset_name: str):
    """
    Removes all data banks.
    """
    remove_dataset(dataset_name)
