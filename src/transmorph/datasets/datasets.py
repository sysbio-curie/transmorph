#!/usr/bin/env python3

import anndata as ad
import numpy as np

from os.path import dirname
from scipy.sparse import load_npz
from typing import Dict

from .http_dl import download_dataset

# GIT: small datasets, can be hosted on Git
# ONLINE: bigger datasets, are downloaded if necessary
#   by the transmorph http API.
DPATH_GIT = f"{dirname(__file__)}/data/"
DPATH_ONLINE = "/".join(__file__.split("/")[:-4]) + "/"


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
    a1 = ad.AnnData(x1)
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
    a2 = ad.AnnData(x2)
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
    xl = load_dataset(DPATH_GIT, "spiralA_labels.csv")
    yl = load_dataset(DPATH_GIT, "spiralB_labels.csv")
    xs = ad.AnnData(load_dataset(DPATH_GIT, "spiralA.csv"))
    xs.obs["label"] = xl
    yt = ad.AnnData(load_dataset(DPATH_GIT, "spiralB.csv"))
    yt.obs["label"] = yl
    return {"src": xs, "ref": yt}


def load_travaglini_10x():
    """
    Loads a large single-cell lung dataset. Reference:

    https://www.nature.com/articles/s41586-020-2922-4

    Dataset
    -------
    - Number of datasets: 3
    - Embedding dimension: 500
    - Sizes: (9744,500), (28793,500) and (27125,500)
    - Number of labels: 4

    Format
    ------
    {
        "patient_1": AnnData(obs: "cell_type"),
        "patient_2": AnnData(obs: "cell_type"),
        "patient_3": AnnData(obs: "cell_type"),
    }
    """
    download_dataset("travaglini_10x")  # TODO handle network exceptions
    dataset_root = DPATH_ONLINE + "data/travaglini_10x/"
    data = {}
    for patient_id in (1, 2, 3):
        counts = load_dataset(
            dataset_root,
            f"P{patient_id}_counts.npz",
            is_sparse=True,
        )
        cell_types = load_dataset(
            dataset_root,
            f"P{patient_id}_labels.csv",
        ).astype(str)
        cell_types[cell_types == "0.0"] = "Endothelial"
        cell_types[cell_types == "1.0"] = "Stromal"
        cell_types[cell_types == "2.0"] = "Epithelial"
        cell_types[cell_types == "3.0"] = "Immune"
        adata = ad.AnnData(counts)
        adata.obs["cell_type"] = cell_types
        data[f"patient_{patient_id}"] = adata
    return data
