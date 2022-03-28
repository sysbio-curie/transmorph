#!/usr/bin/env python3

import anndata as ad
import numpy as np
from scipy.sparse import load_npz
from os.path import dirname

DPATH = f"{dirname(__file__)}/data/%s"
DPATH_LOCAL = "/home/risitop/Documents/PHD/coding/%s"  # TODO: this is dirty


def load_dataset(dpath, filename, is_sparse=False):
    if not is_sparse:
        return np.loadtxt(dpath % filename, delimiter=",")
    return load_npz(dpath % filename).toarray()


def load_test_datasets_small():
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
    xl = load_dataset(DPATH, "spiralA_labels.csv")
    yl = load_dataset(DPATH, "spiralB_labels.csv")
    xs = ad.AnnData(load_dataset(DPATH, "spiralA.csv"))
    xs.obs["label"] = xl
    yt = ad.AnnData(load_dataset(DPATH, "spiralB.csv"))
    yt.obs["label"] = yl
    return {"src": xs, "ref": yt}


def load_travaglini_10x():
    data = {}
    for patient_id in (1, 2, 3):
        counts = load_dataset(
            DPATH_LOCAL,
            f"transmorph_local/notebooks/examples/data/P{patient_id}_counts.npz",
            is_sparse=True,
        )
        cell_types = load_dataset(
            DPATH_LOCAL,
            f"transmorph_local/notebooks/examples/data/P{patient_id}_labels.csv",
        )
        adata = ad.AnnData(counts)
        adata.obs["cell_type"] = cell_types
        data[f"patient_{patient_id}"] = adata
    return data
