#!/usr/bin/env python3

from collections.abc import Collection
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Optional, Tuple, TypeVar

import anndata as ad
import numpy as np

from .. import InputType, check_input_transmorph

T = TypeVar("T")


def _get_top_label(
    labels: Collection[T], priors: Optional[Dict[T, float]] = None
) -> Tuple[T, float]:
    # Returns most represented label from a set as well as
    # its frequency
    counts = {}
    for label_i in labels:
        if label_i not in counts:
            counts[label_i] = 0
        if priors is None:
            counts[label_i] += 1
        else:
            counts[label_i] += 1 / priors[label_i]

    top_label = max(counts.keys(), key=lambda k: counts[k])
    return top_label, counts[top_label] / sum(counts.values())


def _set_labels_inplace(adata_qry, adata_ref, label, new_values):
    # Small pandas helper

    # Writing values
    adata_qry.obs[label] = new_values
    adata_qry.obs[label] = adata_qry.obs[label].astype("category")

    # Adding missing categories
    to_add = []
    for c in adata_ref.obs[label].cat.categories:
        if c not in adata_qry.obs[label].cat.categories:
            to_add.append(c)
    adata_qry.obs[label].cat.add_categories(to_add, inplace=True)

    # Reordering
    adata_qry.obs[label].cat.reorder_categories(
        adata_ref.obs[label].cat.categories, inplace=True
    )
    if f"{label}_colors" in adata_ref.uns:
        adata_qry.uns[f"{label}_colors"] = adata_ref.uns[f"{label}_colors"]


def label_transfer_knn(
    datasets: InputType,
    reference: ad.AnnData,
    label: str,
    use_rep: str = "X_transmorph",
    n_neighbors: int = 10,
    inplace: bool = True,
    adjust_prior: bool = True,
) -> Optional[Tuple[List[np.ndarray], List[np.ndarray]]]:
    """
    Transfers ref_adata labels onto all AnnDatas after integration via nearest
    neighbors.
    """
    # Type checking
    check_input_transmorph(datasets)
    if isinstance(datasets, Dict):
        datasets = list(datasets.values())

    # Prediction
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(reference.obsm[use_rep])
    if adjust_prior:
        priors = {
            lb: sum(reference.obs[label] == lb) for lb in set(reference.obs[label])
        }
    else:
        priors = {lb: 1.0 for lb in set(reference.obs[label])}

    # if inplace = False
    predictions, confidences = [], []
    for adata in datasets:
        if adata is reference:
            predictions.append(reference.obs[label].to_numpy())
            confidences.append(np.ones(reference.n_obs, dtype=np.float32))
            continue
        indices = nn.kneighbors(
            adata.obsm[use_rep],
            return_distance=False,
        )
        prediction = []
        prediction_confidence = []
        for i in range(adata.n_obs):
            top_label, confidence = _get_top_label(
                np.array(reference.obs[label][indices[i]]),
                priors,
            )
            prediction.append(top_label)
            prediction_confidence.append(confidence)

        # Writing if necessary
        if inplace:
            _set_labels_inplace(adata, reference, label, prediction)
            adata.obs[f"{label}_confidence"] = prediction_confidence

        predictions.append(np.array(prediction))
        confidences.append(np.array(prediction_confidence))

    if not inplace:
        return predictions, confidences


def label_transfer_cluster(
    datasets: InputType,
    reference: ad.AnnData,
    label: str,
    cluster: str = "louvain",
    use_rep: str = "X_transmorph",
    n_neighbors: int = 10,
    inplace: bool = True,
    adjust_prior: bool = True,
) -> Optional[Tuple[List[np.ndarray], List[np.ndarray]]]:
    """
    Transfers ref_adata labels onto all AnnDatas after integration
    via nearest neighbors, with a correction step guaranteeing
    homogeneous label within a cluster.
    """
    # Type checking
    check_input_transmorph(datasets)
    if isinstance(datasets, Dict):
        datasets = list(datasets.values())

    # Prediction
    prediction_result = label_transfer_knn(
        datasets,
        reference,
        label,
        use_rep,
        n_neighbors,
        False,
        adjust_prior,
    )
    assert prediction_result is not None
    predictions, confidences = [], []

    # Correction
    for adata, prediction, confidence in zip(datasets, *prediction_result):
        if adata is reference:
            continue

        new_prediction = np.empty(adata.n_obs, dtype=np.chararray)
        new_confidence = np.zeros(adata.n_obs, dtype=np.float32)
        for cl_i in set(adata.obs[cluster]):
            selector = adata.obs[cluster] == cl_i
            labels = prediction[selector]
            top_label, confidence = _get_top_label(labels)
            new_prediction[selector] = top_label
            new_confidence[selector] = confidence

        # Writing if necessary
        if inplace:
            _set_labels_inplace(adata, reference, label, new_prediction)
            adata.obs[f"{label}_confidence"] = new_confidence

        predictions.append(np.array(prediction))
        confidences.append(np.array(new_confidence))

    if not inplace:
        return predictions, confidences
