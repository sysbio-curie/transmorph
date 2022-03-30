#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os
import umap

from anndata import AnnData
from matplotlib import cm
from numbers import Number
from os.path import exists
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from typing import List, Union

MARKERS = "osv^<>pP*hHXDd"


def plot_result(
    datasets: List[AnnData],
    matching_mtx: Union[csr_matrix, None] = None,
    reducer: str = "umap",
    color_by: str = "__dataset__",
    palette: str = "rainbow",
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    show: bool = True,
    save: bool = False,
    caller_path: str = "",
    suffix: str = "__color_by__",
    dpi=200,
):
    # Checking parameters
    if matching_mtx is not None:
        assert len(datasets) == 2, "Drawing matching requires exactly two datasets."
        assert type(matching_mtx) is csr_matrix, "Matching must come as a csr_matrix."

    # Guessing continous or discrete labels
    if color_by == "__dataset__":
        n_labels = len(datasets)
        continuous = False
        all_labels = {}
    else:
        all_labels = set()
        for adata in datasets:
            all_labels = all_labels | set(adata.obs[color_by])
        n_labels = len(all_labels)
        continuous = (n_labels > 20) and all(isinstance(y, Number) for y in all_labels)

    # Reducing dimension if necessary
    if all("transmorph" in adata.obsm for adata in datasets):
        representations = [adata.obsm["transmorph"] for adata in datasets]
    else:
        representations = [adata.X for adata in datasets]
    assert all(type(X) is np.ndarray for X in representations)
    repr_dim = representations[0].shape[1]
    assert all(X.shape[1] == repr_dim for X in representations)
    if repr_dim > 2:
        all_X = np.concatenate(representations, axis=0)
        if reducer == "umap":
            if repr_dim > 30:
                all_X = PCA(n_components=30).fit_transform(all_X)
            all_X = umap.UMAP(min_dist=0.5).fit_transform(all_X)
            if xlabel == "":
                xlabel = "UMAP1"
            if ylabel == "":
                ylabel = "UMAP2"
        elif reducer == "pca":
            all_X = PCA(n_components=2).fit_transform(all_X)
            if xlabel == "":
                xlabel = "PC1"
            if ylabel == "":
                ylabel = "PC2"
        else:
            raise ValueError(
                f"Unrecognized reducer: {reducer}. Expected 'umap' or 'pca'."
            )
        offset = 0
        for i, adata in enumerate(datasets):
            nobs = adata.n_obs
            representations[i] = all_X[offset : offset + nobs]
            offset += nobs

    # Guess plotting parameters
    npoints = sum(X.shape[0] for X in representations)
    size = 100
    if npoints > 100:
        size = 60
    if npoints > 1000:
        size = 40
    if npoints > 10000:
        size = 20

    ndatasets = len(datasets)
    alpha = 0.8
    if ndatasets >= 3:
        alpha = 0.6
    if ndatasets >= 10:
        alpha = 0.4

    # Prepare the palette
    cmap = cm.get_cmap(palette, n_labels)

    # Do the plotting
    plt.figure(figsize=(6, 6), dpi=dpi)
    for i, adata in enumerate(datasets):
        X = representations[i]
        mk = MARKERS[i]
        if color_by == "__dataset__":
            plt.scatter(
                *X.T, marker=mk, s=size, label=f"Dataset {i}", ec="k", alpha=alpha
            )
            continue
        else:
            plt.scatter([], [], marker=mk, s=40, c="k", label=f"Dataset {i}")
        if continuous:
            plt.scatter(
                *X.T, c=adata.obs[color_by], alpha=alpha, marker=mk, ec="k", s=size
            )
        else:
            for k, label in enumerate(all_labels):
                color = cmap(k / (n_labels - 1))
                if i == 0:
                    plt.scatter(
                        [], [], marker="o", s=40, color=color, label=label, ec="k"
                    )
                plt.scatter(
                    *X[adata.obs[color_by] == label].T,
                    alpha=alpha,
                    marker=mk,
                    ec="k",
                    s=size,
                    color=color,
                )

    # Drawing matching if necessary
    if matching_mtx is not None:
        Tcoo = matching_mtx.tocoo()
        X1, X2 = representations
        for i, j, v in zip(Tcoo.row, Tcoo.col, Tcoo.data):
            plt.plot(
                [X1[i][0], X2[j][0]],
                [X1[i][1], X2[j][1]],
                alpha=np.clip(v * X2.shape[0], 0, 1),
                c="k",
            )

    # Reordering legend
    handles, labels = plt.gca().get_legend_handles_labels()
    order = np.argsort(labels)
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    # Adding text pieces
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks([])
    plt.yticks([])
    if color_by == "__dataset__":
        color_by = "dataset"
    plt.title(title + f" -- Color: {color_by}")

    # Saving, showing and closing
    if save:
        save_path = "/".join(caller_path.split("/")[:-1]) + "/figures/"
        if not exists(save_path):
            os.mkdir(save_path)
        fname = caller_path.split("/")[-1].split(".")[0]
        if suffix == "__color_by__":
            suffix = color_by
        if suffix != "":
            suffix = "_" + suffix
        plt.savefig(save_path + f"{fname}{suffix}.png")
    if show:
        plt.show()
    plt.close()
