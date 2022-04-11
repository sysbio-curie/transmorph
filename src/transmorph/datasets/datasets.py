#!/usr/bin/env python3
# Contains high level functions to load datasets.

import anndata as ad
import scanpy as sc
import numpy as np
import os

from os.path import dirname
from scipy.sparse import load_npz
from typing import Dict

from .databank_api import check_files, download_dataset, remove_dataset, unzip_file


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
        ]
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
        ]
    )
    a2 = ad.AnnData(x2, dtype=x2.dtype)
    a2.obs["class"] = [0, 0, 0, 0, 1, 1, 1, 1, 1]
    errors = np.array(a1.obs["class"])[:, None] != np.array(a2.obs["class"])
    return {"src": a1, "ref": a2, "error": errors}


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


def load_bank(dataset_name: str, keep_sparse: bool = False):
    """
    Parameters
    ----------
    dataset_name: str
        "name" value in the json file for the bank of datasets.

    keep_sparse: bool, default = False
        Prevents AnnData.X to be converted to ndarray.
    """
    # TODO: print information about dataset here
    download_needed = not check_files(dataset_name)
    if download_needed:
        zip_path = download_dataset(dataset_name)
        unzip_file(zip_path, dataset_name)
        check_files(dataset_name)
    dataset_root = DPATH_DATASETS + f"{dataset_name}/"
    data = {}
    for fname in os.listdir(dataset_root):
        adata = sc.read_h5ad(dataset_root + fname)
        if not keep_sparse:
            adata.X = adata.X.toarray()
        pid = fname.split(".")[0]
        data[pid] = adata
    return data


def remove_bank(dataset_name: str):
    """
    Removes all data banks.
    """
    remove_dataset(dataset_name)
