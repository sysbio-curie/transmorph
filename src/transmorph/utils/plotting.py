#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os

from anndata import AnnData
from matplotlib import cm
from numbers import Number
from os.path import exists
from scipy.sparse import csr_matrix
from typing import Dict, List, Literal, Optional, Union

from ..engine import Model
from ..engine.evaluators import (
    evaluate_matching_layer,
    matching_edge_accuracy_discrete,
    matching_edge_penalty_continuous,
)
from ..engine.layers import LayerMatching
from ..utils.anndata_manager import (
    anndata_manager as adm,
    AnnDataKeyIdentifiers,
    slice_common_features,
)
from ..utils.dimred import pca, umap
from ..utils.matrix import extract_chunks, guess_is_discrete

MARKERS = "osv^<>pP*hHXDd"


def reduce_dimension(
    datasets: Union[AnnData, List[AnnData], Dict[str, AnnData]],
    reducer: Literal["umap", "pca"] = "umap",
    input_obsm: Optional[str] = None,
    output_obsm: Optional[str] = None,
) -> None:
    """
    Applies a dimensionality reduction algorithm to a list of AnnData objects
    that must have a non-empty common feature space, for plotting purposes.

    Parameters
    ----------
    datasets: Union[AnnData, List[AnnData]]
        List of AnnData objects to embed in a plottable 2D space.

    reducer: Literal["umap", "pca"]
        Dimension reduction algorithm to use.

    input_obsm: str, default = "tr_plot"
        .obsm key to save representation in. Can then be provided
        to scatter_plot.
    """
    if isinstance(datasets, AnnData):
        datasets = [datasets]
    if isinstance(datasets, Dict):
        datasets = list(datasets.values())

    if input_obsm is None:
        representations = slice_common_features(datasets)
        assert (
            representations[0].shape[1] > 0
        ), "No common gene space found for datasets. Try providing a custom obsm."
    else:
        representations = [
            adm.get_value(adata, key=input_obsm, field_str="obsm") for adata in datasets
        ]

    for i, X in enumerate(representations):
        if type(X) is csr_matrix:
            representations[i] = X.toarray()

    # Dimred if necessary
    assert all(isinstance(X, np.ndarray) for X in representations)
    repr_dim = representations[0].shape[1]
    assert all(X.shape[1] == repr_dim for X in representations)
    all_X = np.concatenate(representations, axis=0)
    if repr_dim > 2:
        if reducer == "umap":
            all_X = umap(all_X, embedding_dimension=2)
        elif reducer == "pca":
            all_X = pca(all_X, n_components=2)
        else:
            raise ValueError(
                f"Unrecognized reducer: {reducer}. Expected 'umap' or 'pca'."
            )

    if output_obsm is None:
        output_obsm = AnnDataKeyIdentifiers.PlotRepresentation

    # Saving representations
    for adata, X in zip(
        datasets,
        extract_chunks(all_X, [adata.n_obs for adata in datasets]),
    ):
        adm.set_value(
            adata=adata,
            key=output_obsm,
            field="obsm",
            value=X,
            persist="output",
        )
        adm.set_value(
            adata=adata,
            key=f"reducer_{output_obsm}",
            field="uns",
            value=reducer,
            persist="output",
        )


def scatter_plot(
    datasets: Union[AnnData, List[AnnData], Dict[str, AnnData]],
    matching_mtx: Optional[csr_matrix] = None,
    input_obsm: Optional[str] = None,
    color_by: str = "__dataset__",
    palette: str = "rainbow",
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    batch_names: Optional[List[int]] = None,
    show_title: bool = True,
    show_legend: bool = True,
    plot_cluster_names: bool = False,
    show: bool = True,
    save: bool = False,
    extension: str = "png",
    caller_path: str = "",
    suffix: str = "__color_by__",
    dpi: int = 100,
):
    """
    Advanced plotting function for transmorph results, handling parameters
    guessing, representation cacheing, automatic dimensionality reduction,
    continuous or linear labeling, figure saving.

    IMPORTANT: If save=True, see parameter caller_path.

    Parameters
    ----------
    datasets: List[AnnData]
        List of datasets to display, represented as annotated data. If they
        have been processed by transmorph, integrated embedding will be used.
        Otherwise, AnnData.X is chosen as fallback representation (in this
        case, all datasets must be embedded in the same space).

    matching_mtx: csr_matrix, default = None
        This option is only valid for two datasets. You can provide a boolean
        or probabilistic matching between datasets represented as a sparse
        matrix (CSR format), which will be displayed on the plot. Be careful,
        for too large datasets this will slow down plotting and yield results
        difficult to read.

    reducer: str, default = "umap"
        Algorithm to use if embedding dimensionality exceeds 2. Valid options
        are "pca" for linear dimensionality reduction and "umap" for embedding
        high dimensional datasets such as single-cell data.

    color_by: str, default = "__dataset__"
        Labeling colors to use in the figure. "__dataset__" will color each
        point depending of its dataset of origin. Otherwise, all keys of
        AnnData.obs are valid. The function will try to guess discrete- or
        continuous-ness of the labels, and color accordingly.

    palette: str, default = "rainbow"
        Matplotlib colormap to pick colors from.

    title: str, default = ""
        Title of the plot.

    xlabel: str, default = ""
        Labeling of the x axis.

    ylabel: str, default = ""
        Labeling of the y axis.

    show_title: bool, default = True
        Shows plot title.

    show_legend: bool, default = True
        Shows the legend panel.

    plot_cluster_names: bool, default = False
        Shows cluster names on the plot, gathered from color_by.

    show: bool, default = True
        Call plt.show() at the end to display the figure in a separate
        window.

    save: bool, default = False
        Save the plot in the same directory as the file calling the
        functions, in a figures/ subdirectory (created if necessary).

    extension: str, default = "png"
        If save=True, valid image file extension to use.

    caller_path:
        If save=True, pass f"__file__" to indicate file location.

    suffix: str = "__color_by__":
        If save=True, suffix to append to the file name to differentiate
        between image files of the same source. "__color_by__" will use
        labels as suffix (e.g. "plot_cell_type.png")
                                     ^^^^^^^^^

    dpi: int, default = 200,
        Dot per inch to use for the figure.
    """
    if isinstance(datasets, AnnData):
        datasets = [datasets]
    if isinstance(datasets, Dict):
        batch_names = list(datasets.keys())
        datasets = list(datasets.values())

    # Checking parameters
    assert all(
        isinstance(adata, AnnData) for adata in datasets
    ), "All datasets must be AnnData."
    if matching_mtx is not None:
        assert len(datasets) == 2, "Drawing matching requires exactly two datasets."
        assert type(matching_mtx) is csr_matrix, "Matching must come as a csr_matrix."

    if input_obsm is None:
        input_obsm = AnnDataKeyIdentifiers.PlotRepresentation

    # Retrieving representation
    default_xlabel = "Feature 1"
    default_ylabel = "Feature 2"
    if not all(adm.isset_value(adata, key=input_obsm) for adata in datasets):
        assert all(
            adata.X.shape[1] == 2 for adata in datasets
        ), "Make sure to call reduce_dimension first."
        representations = [adata.X for adata in datasets]
    else:
        representations = [adm.get_value(adata, key=input_obsm) for adata in datasets]
        reducer = adm.get_value(datasets[0], key=f"reducer_{input_obsm}")
        if reducer == "umap":
            default_xlabel = "UMAP1"
            default_ylabel = "UMAP2"
        else:
            default_xlabel = "PC1"
            default_ylabel = "PC2"

    if xlabel is None:
        xlabel = default_xlabel
    if ylabel is None:
        ylabel = default_ylabel

    # Guessing continous or discrete labels
    if color_by == "__dataset__":
        n_labels = len(datasets)
        continuous = False
        if batch_names is not None:
            all_labels = batch_names
        else:
            all_labels = [f"Batch {i + 1}" for i in range(n_labels)]
    else:
        all_labels = set()
        for adata in datasets:
            all_labels = all_labels | set(adata.obs[color_by])
        all_labels = sorted(all_labels)
        n_labels = len(all_labels)
        continuous = (n_labels > 20) and all(isinstance(y, Number) for y in all_labels)

    # Guess plotting parameters
    npoints = sum(X.shape[0] for X in representations)
    size = 100
    if npoints > 100:
        size = 60
    if npoints > 500:
        size = 50
    if npoints > 1000:
        size = 40
    if npoints > 5000:
        size = 30
    if npoints > 10000:
        size = 20
    if npoints > 50000:
        size = 10

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
        mk = MARKERS[i % len(MARKERS)]  # Loops to avoid oob error
        if color_by == "__dataset__":
            if ndatasets == 1:
                color = cmap(0.5)
            else:
                color = cmap(i / (ndatasets - 1))
            plt.scatter(*X.T, marker=mk, s=size, alpha=alpha, color=color)
            plt.scatter(
                [],
                [],
                marker=mk,
                s=40,
                label=all_labels[i],
                color=color,
            )
            continue
        if continuous:
            show_legend = False
            plt.scatter(
                *X.T,
                c=adata.obs[color_by],
                alpha=alpha,
                marker=mk,
                s=size,
                cmap=palette,
            )
        else:
            for k, label in enumerate(all_labels):
                if n_labels == 1:
                    color = cmap(0.5)
                else:
                    color = cmap(k / (n_labels - 1))
                if i == 0:
                    plt.scatter(
                        [],
                        [],
                        marker="o",
                        s=40,
                        color=color,
                        label=label,
                    )
                plt.scatter(
                    *X[adata.obs[color_by] == label].T,
                    alpha=alpha,
                    marker=mk,
                    s=size,
                    color=color,
                )
    if continuous:
        plt.colorbar()

    # Plotting cluster names if necessary
    if plot_cluster_names:
        for ilabel, label in enumerate(all_labels):
            nobs, cl_pos = 0, np.zeros((2,), dtype=np.float32)
            for i, adata in enumerate(datasets):
                if color_by == "__dataset__":
                    if i == ilabel:
                        selector = np.ones(adata.n_obs, dtype=bool)
                    else:
                        selector = np.zeros(adata.n_obs, dtype=bool)
                else:
                    selector = adata.obs[color_by] == label
                nobs += selector.sum()
                cl_pos += representations[i][selector].sum(axis=0)
            cl_pos /= nobs
            plt.text(
                *cl_pos,
                s=str(label),
                bbox={"edgecolor": "none", "facecolor": "white", "alpha": 0.5},
                fontsize=12,
                ha="center",
                va="center",
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
    if show_legend:
        handles, labels = plt.gca().get_legend_handles_labels()
        if len(handles) < 5:
            legendsize = 12
        elif len(handles) < 10:
            legendsize = 8
        else:
            legendsize = 5
        order = np.argsort(labels)
        plt.legend(
            [handles[idx] for idx in order],
            [labels[idx] for idx in order],
            fontsize=legendsize,
        )

    # Adding text pieces
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.xticks([])
    plt.yticks([])
    if color_by == "__dataset__":
        color_by = "dataset"
    if show_title:
        if title is None:
            title = ""
        plt.title(title, fontsize=18)

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
        plt.savefig(save_path + f"{fname}{suffix}.{extension}")
    if show:
        plt.show()
    plt.close()


def plot_matching_eval(
    model: Model,
    datasets: Union[List[AnnData], Dict[str, AnnData]],
    obs: str,
    dataset_keys: Optional[List[str]] = None,
    title: Optional[str] = None,
    matshow_kwargs: Dict = {},
) -> None:
    """
    Plots matching evaluation between datasets as a heatmap.

    Parameters
    ----------
    layer_matching: LayerMatching
        LayerMatching that has been evaluated.

    evaluator: str
        String identifier of the evaluator.

    dataset_keys: Optional[List[str]], default = None
        Dataset labels to add to the plot.

    title: Optional[str], default = None
        Title to use for the plot. If None, generates one.

    matshow_kwargs: Dict[str, Any], default = {}
        Additional matshow parameters.
    """
    if isinstance(datasets, Dict):
        dataset_keys = list(datasets.keys())
        datasets = list(datasets.values())
    layer_matchings = model.get_layers_by_type(LayerMatching)
    assert len(layer_matchings) > 0, "No layer of type LayerMatching found."
    layer_matching = layer_matchings[0]
    is_discrete = all(guess_is_discrete(adata.obs[obs]) for adata in datasets)
    if is_discrete:
        evaluator = matching_edge_accuracy_discrete(obs)
    else:
        evaluator = matching_edge_penalty_continuous(obs)
    scores = evaluate_matching_layer(layer_matching, datasets, evaluator)
    ndatasets = scores.shape[0]
    if dataset_keys is not None:
        assert len(dataset_keys) == ndatasets, (
            f"Inconsistent number of datasets in dataset_keys, expected {ndatasets}, "
            f"found {len(dataset_keys)}."
        )

    plt.matshow(scores, **matshow_kwargs)
    vmin, vmax = plt.gci().get_clim()
    mid = 0.5 * (vmax - vmin)
    for i in range(ndatasets):
        for j in range(ndatasets):
            if scores[i, j] > mid:
                fontcolor = "k"
            else:
                fontcolor = "w"
            plt.text(
                i,
                j,
                "{:.1f}".format(scores[i, j]),
                va="center",
                ha="center",
                c=fontcolor,
            )
        values = [scores[i, k] for k in range(ndatasets) if k != i]
        plt.text(ndatasets, i, "{:.1f}".format(np.mean(values)), va="center")
        plt.text(ndatasets + 1, i, "{:.1f}".format(np.std(values)), va="center")

    plt.text(ndatasets, -0.5, "Mean", rotation=60, va="bottom")
    plt.text(ndatasets + 1, -0.5, "STD", rotation=60, va="bottom")
    if dataset_keys is not None:
        plt.xticks(range(ndatasets), dataset_keys, rotation=60)
        plt.yticks(range(ndatasets), dataset_keys)
    else:
        plt.xticks([])
        plt.yticks([])
    if title is None:
        title = "Pairwise matching evaluation"
    plt.title(title)


def plot_label_distribution_heatmap(
    datasets: Union[List[AnnData], Dict[str, AnnData]],
    label: str,
    dataset_keys: Optional[List[str]] = None,
    title: Optional[str] = None,
) -> None:
    """ """
    if isinstance(datasets, Dict):
        dataset_keys = list(datasets.keys())
        datasets = list(datasets.values())
    all_labels = set()
    for adata in datasets:
        all_labels = all_labels | set(adata.obs[label])
    all_labels = list(sorted(all_labels))

    ndatasets, nlabels = len(datasets), len(all_labels)

    labels_counts = np.zeros(nlabels, dtype=int)
    labels_dist = np.zeros((ndatasets, nlabels), dtype=np.float32)
    for i, adata in enumerate(datasets):
        nsamples = adata.n_obs
        for j, lb in enumerate(all_labels):
            v = np.sum(adata.obs[label] == lb)
            labels_dist[i, j] = v / nsamples
            labels_counts[j] += v

    vmin, vmax = 0.0, 1 / nlabels
    plt.matshow(labels_dist, vmin=vmin, vmax=vmax)

    for i, adata in enumerate(datasets):
        nsamples = adata.n_obs
        for j, lb in enumerate(all_labels):
            val = np.sum(adata.obs[label] == lb) / nsamples
            fc = "k" if val > vmax / 2 else "w"
            plt.text(j, i, f"{int(val*100)}%", ha="center", va="center", c=fc)

    for i, adata in enumerate(datasets):
        plt.text(nlabels, i, adata.n_obs, ha="left", va="center")

    for j, lb in enumerate(all_labels):
        plt.text(j, ndatasets, labels_counts[j], ha="center", va="top", rotation=30)

    plt.xticks(np.arange(nlabels), all_labels, rotation=60)
    if dataset_keys is not None:
        plt.yticks(np.arange(ndatasets), dataset_keys)
    if title is not None:
        plt.title(title)
