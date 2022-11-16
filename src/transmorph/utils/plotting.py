#!/usr/bin/env python3

import matplotlib as mt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from anndata import AnnData
from matplotlib import cm
from numbers import Number
from scipy.sparse import csr_matrix
from typing import Dict, List, Literal, Optional, Union


from ..engine import Model
from ..engine.evaluators import (
    evaluate_matching_layer,
    matching_edge_accuracy_discrete,
    matching_edge_penalty_continuous,
)
from ..engine.layers import (
    Layer,
    LayerMatching,
    LayerMerging,
)
from ..engine.matching import CombineMatching
from ..engine.traits import ContainsTransformations
from ..utils.anndata_manager import (
    anndata_manager as adm,
    AnnDataKeyIdentifiers,
    slice_common_features,
)
from ..utils.dimred import pca, umap
from ..utils.matrix import extract_chunks, guess_is_discrete


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
    use_rep: Optional[str] = None,
    color_by: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    dataset_labels: Optional[List[str]] = None,
    labels_on_plot: bool = False,
    palette: str = "rainbow",
    dot_size: float = 0.5,
    dpi: int = 100,
):
    """
    Advanced plotting function for transmorph results.

    Parameters
    ----------
    datasets: List[AnnData]
        List of datasets to display, represented as annotated data. If they
        have been processed by transmorph, integrated embedding will be used.
        Otherwise, AnnData.X is chosen as fallback representation (in this
        case, all datasets must be embedded in the same space).

    use_rep: Optional[str]
        AnnData.obsm key containing data embedding. If None, attempts to use
        AnnData.X instead if dimensionality is 2.

    color_by: Optional[str]
        Labeling colors to use in the figure. None will color each
        point depending of its dataset of origin. Otherwise, all keys of
        AnnData.obs are valid. The function will try to guess discrete- or
        continuous-ness of the labels, and color accordingly.

    xlabel: Optional[str]
        Labeling of the x axis.

    ylabel: Optional[str]
        Labeling of the y axis.

    title: Optional[str]
        Title of the plot.

    dataset_labels: Optional[List[str]]
        Labels to use in legend to describe each AnnData, if datasets
        is not a Dict.

    labels_on_plot: bool, default = False
        Shows labels on plot instead of in legend.

    palette: str, default = "rainbow"
        Matplotlib colormap to pick colors from.

    dot_size: float, default = 0.5
        Scatter "size" parameter.

    dpi: int, default = 200,
        Dot per inch to use for the figure.
    """
    if isinstance(datasets, AnnData):
        datasets = [datasets]
    if isinstance(datasets, Dict):
        dataset_labels = list(datasets.keys())
        datasets = list(datasets.values())

    # Checking parameters
    assert all(
        isinstance(adata, AnnData) for adata in datasets
    ), "All datasets must be AnnData."

    if use_rep is None:
        assert all(
            adata.X.shape[1] == 2 for adata in datasets
        ), "No use_rep specified, and some adata.X are not 2 dimensional."

    # Retrieving representation
    default_xlabel = "Feature 1"
    default_ylabel = "Feature 2"
    if not all(adm.isset_value(adata, key=use_rep) for adata in datasets):
        representations = [adata.X for adata in datasets]
    else:
        representations = [adm.get_value(adata, key=use_rep) for adata in datasets]
        reducer = adm.get_value(datasets[0], key=f"reducer_{use_rep}")
        if reducer == "umap":
            default_xlabel = "UMAP1"
            default_ylabel = "UMAP2"
        elif reducer == "pca":
            default_xlabel = "PC1"
            default_ylabel = "PC2"

    if xlabel is None:
        xlabel = default_xlabel
    if ylabel is None:
        ylabel = default_ylabel

    if color_by is None:  # Color by dataset
        continuous_palette = False
        if dataset_labels is not None:
            all_labels = dataset_labels
        else:
            all_labels = [f"Dataset {i + 1}" for i, _ in enumerate(datasets)]
    else:  # Colour by custom label
        all_labels = set()
        for adata in datasets:
            all_labels = all_labels | set(adata.obs[color_by])
        all_labels = sorted(all_labels)
        # simple heuristic
        continuous_palette = (len(all_labels) > 20) and all(
            isinstance(y, Number) for y in all_labels
        )

    dot_alpha = 1
    n_labels = len(all_labels)
    n_datasets = len(datasets)

    # Prepare the palette
    cmap = cm.get_cmap(palette, n_labels)

    # Do the plotting
    fig = plt.figure(dpi=dpi)
    ax_scatter = fig.add_subplot(111, aspect="equal")

    scatter = None
    for i, adata in enumerate(datasets):
        X = representations[i]
        if color_by is None:
            if n_datasets == 1:
                color = cmap(0.5)
            else:
                color = cmap(i / (n_datasets - 1))
            scatter = ax_scatter.scatter(
                *X.T, s=dot_size, alpha=dot_alpha, color=color, label=all_labels[i]
            )
        elif continuous_palette:
            scatter = ax_scatter.scatter(
                *X.T,
                c=adata.obs[color_by],
                alpha=dot_alpha,
                s=dot_size,
                cmap=palette,
            )
        else:
            for k, label in enumerate(all_labels):
                if n_labels == 1:
                    color = cmap(0.5)
                else:
                    color = cmap(k / (n_labels - 1))
                if i == 0:
                    scatter = ax_scatter.scatter(
                        *X[adata.obs[color_by] == label].T,
                        alpha=dot_alpha,
                        s=dot_size,
                        color=color,
                        label=label,
                    )
                else:
                    scatter = ax_scatter.scatter(
                        *X[adata.obs[color_by] == label].T,
                        alpha=dot_alpha,
                        s=dot_size,
                        color=color,
                    )

    if continuous_palette:
        divider = make_axes_locatable(ax_scatter)
        ax_legend = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(scatter, cax=ax_legend).set_label(color_by, fontsize=14)
    elif labels_on_plot:
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
    else:
        divider = make_axes_locatable(ax_scatter)
        ax_legend = divider.append_axes("right", size="5%", pad=0.1)
        handles, labels = ax_scatter.get_legend_handles_labels()
        order = np.argsort(labels)
        for label in order:
            ax_legend.scatter([], [], label=label)
        legend = ax_legend.legend(
            [handles[idx] for idx in order],
            [labels[idx] for idx in order],
            fontsize=12,
            ncols=(1 + len(order) // 10),
            loc="center left",
        )
        legend.legendHandles[0]._sizes = [15]
        legend.legendHandles[1]._sizes = [15]
        ax_legend.axis("off")

    # Adding text pieces
    ax_scatter.set_xlabel(xlabel, fontsize=16)
    ax_scatter.set_ylabel(ylabel, fontsize=16)
    ax_scatter.set_xticks([])
    ax_scatter.set_yticks([])

    if title is not None:
        ax_scatter.set_title(title, fontsize=18)

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
    if isinstance(datasets, AnnData):
        datasets = {"Batch": datasets}
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


def plot_model(
    model: Model,
    layer_edgecolor: str = "royalblue",
    layer_facecolor: str = "lightsteelblue",
    algorithm_edgecolor: str = "darkorange",
    algorithm_facecolor: str = "bisque",
) -> None:
    """
    Plots a visual representation of a model.
    """

    BOX_WIDTH = 40
    BOX_HEIGHT = 20

    def _retrieve_elements(layer: Layer):
        elements = []
        if isinstance(layer, ContainsTransformations):
            for transformation in layer.transformations:
                elements.append(type(transformation).__name__)
        if isinstance(layer, LayerMatching):
            elements.append(type(layer.matching).__name__)
            if isinstance(layer.matching, CombineMatching):
                for matching in layer.matching.matchings:
                    elements.append(type(matching).__name__)
        if isinstance(layer, LayerMerging):
            elements.append(type(layer.merging).__name__)
        return elements

    def _retrieve_n_rows(model: Model) -> int:
        max_elements = 0
        for layer in model.layers:
            max_elements = max(max_elements, len(_retrieve_elements(layer)))
        return max_elements

    def _plot_textbox(x: int, y: int, text: str, ax: mt.axes.Axes, **rect_kwargs):
        rect = mt.patches.Rectangle((x, y), BOX_WIDTH, BOX_HEIGHT, **rect_kwargs)
        ax.add_patch(rect)
        ax.text(
            x + BOX_WIDTH / 2,
            y + BOX_HEIGHT / 2,
            text,
            ha="center",
            va="center",
            fontsize=9,
        )

    def _plot_layer(layer: Layer, x_offset: int, ax: mt.axes.Axes):
        _plot_textbox(
            x_offset,
            0,
            type(layer).__name__[5:],
            ax,
            linewidth=1,
            edgecolor=layer_edgecolor,
            facecolor=layer_facecolor,
        )
        next_layers = layer.output_layers
        if len(next_layers) > 1:
            raise NotImplementedError
        if len(next_layers) == 0:
            return
        plt.arrow(
            x_offset + BOX_WIDTH,
            BOX_HEIGHT / 2,
            BOX_WIDTH / 2,
            0,
            length_includes_head=True,
            facecolor="k",
            head_width=3,
            head_length=1.5,
        )
        y_offset = 0
        for element in _retrieve_elements(layer):
            plt.arrow(
                x_offset + BOX_HEIGHT,
                y_offset,
                0,
                -BOX_HEIGHT,
                length_includes_head=True,
                facecolor="k",
                head_width=2,
            )
            _plot_textbox(
                x_offset,
                y_offset - BOX_HEIGHT * 2,
                element,
                ax,
                linewidth=1,
                edgecolor=algorithm_edgecolor,
                facecolor=algorithm_facecolor,
            )
            y_offset -= BOX_HEIGHT * 2
        _plot_layer(next_layers[0], x_offset + BOX_WIDTH * 1.5, ax)

    assert isinstance(model, Model), f"Model expected, found {type(model)}."

    n_layers = len(model.layers)
    n_rows = _retrieve_n_rows(model)

    if n_layers == 0:
        return

    fig = plt.figure(figsize=(2.5 * n_layers, 0.7 * (1 + n_rows)))
    ax = fig.add_subplot(111)
    plt.axis("off")

    _plot_layer(model.layers[0], 0, ax)
