#!/usr/bin/env python3

from __future__ import annotations

import logging
import numpy as np
import warnings

from abc import ABC, abstractmethod
from anndata import AnnData
from scipy.sparse import csr_matrix
from transmorph import logger
from transmorph.utils import anndata_manager as adm
from transmorph.utils import AnnDataKeyIdentifiers
from typing import Any, Dict, Hashable, List, Literal, Optional, Tuple, Type, Union


# A trait is a small module of features that can be added
# to a class using inheritance. It allows code factorization,
# and easier compatibility checking. The base trait does nothing
# but checking if an object is endowed with it.


def assert_trait(obj: Any, traits: Union[Type, Tuple[Type, ...]]):
    """
    Raises an exception if $obj is not endowed with the
    trait $trait.
    """
    if isinstance(obj, traits):
        return
    if isinstance(traits, Type):
        all_traits: str = traits.__name__
    else:
        all_traits: str = ", ".join([trait.__name__ for trait in traits])
    raise TypeError(
        f"Object {obj} of type {type(obj)} is not endowed"
        f" with trait(s) {all_traits}."
    )


class CanLog:
    """
    This trait allows a class to send messages to the logging system.
    """

    def __init__(self, str_identifier: str):
        self.str_identifier = str_identifier

    def log(self, msg: str, level: int = logging.DEBUG) -> None:
        """
        Transmits a message to the logging module.

        Parameters
        ----------
        msg: str
            Message to print

        leve: int, default = logging.DEBUG
            Message priority. Set it higher to make it pass filters.
        """
        logger.log(level, f"{self.str_identifier} > {msg}")

    def warn(self, msg: str) -> None:
        """
        Emits a warning message that will both reach the logger and the warning
        console stream.

        Parameters
        ----------
        msg: str
            Message to print
        """
        warnings.warn(msg)
        self.log(msg, level=logging.WARNING)

    def raise_error(self, error_type: Type, msg: str = "") -> None:
        """
        Raises an error of the specified type, and prints the message both in
        the console and in the logging stream.
        """
        self.log(f"{error_type.__name__} -- {msg}")
        raise error_type(msg)

    def __str__(self) -> str:
        return self.str_identifier


class IsRepresentable:
    """
    A representable object is able to provide a matrix
    representation of an AnnData object.
    """

    def __init__(self, repr_key: Union[str, AnnDataKeyIdentifiers]):
        self.repr_key = repr_key

    def write(self, adata: AnnData, X: Union[np.ndarray, csr_matrix]) -> None:
        """
        Inserts a new AnnData representation.
        """
        adm.set_value(
            adata=adata, key=self.repr_key, field="obsm", value=X, persist="pipeline"
        )

    def get(self, adata: AnnData) -> np.ndarray:
        """
        Returns a matrix view of a given AnnData.
        """
        X = adm.get_value(adata=adata, key=self.repr_key)
        assert X is not None, "Representation has not been computed."
        if isinstance(X, csr_matrix):
            X = X.toarray()
        return X


class HasMetadata(ABC, CanLog):
    """
    This trait allows a module to retrieve and store metadata
    from an AnnData object.
    TODO handle subsampling when metadata is related to samples
    """

    def __init__(self):
        CanLog.__init__(self, "TraitHasMetadata")
        self.metadata: List[Dict[Hashable, Any]] = []

    def retrieve_all_metadata(self, datasets: List[AnnData]) -> None:
        """
        Retrieves necessary information from a list of AnnData objects.
        """
        for adata in datasets:
            self.metadata.append(self.retrieve_metadatata(adata))

    @abstractmethod
    def retrieve_metadatata(self, adata: AnnData) -> Dict[Hashable, Any]:
        """
        This must be implemented by child classes.
        """
        pass

    def get_metadata(self, index: int, key: Hashable) -> Optional[Any]:
        """
        Returns a stored information, or None if it has not be
        registered.
        """
        if index >= len(self.metadata):
            self.warn(
                f"List index {index} out of range for list "
                f"of size {len(self.metadata)}."
            )
            return None
        return self.metadata[index].get(key, None)


class UsesCommonFeatures:
    """
    This trait will allow an object to retrieve feature names from an AnnData
    object. They will then be used to slice count matrices in order to select
    pairwise or total common genes intersection.
    """

    def __init__(self, mode: Literal["pairwise", "total"]):
        assert mode in ("pairwise", "total")
        self.mode = mode
        # TFS is used for mode "total", PFS for mode "pairwise"
        self.total_feature_slices: List[np.ndarray] = []
        self.pairwise_feature_slices: Dict[
            Tuple[int, int], Tuple[np.ndarray, np.ndarray]
        ]
        self.fitted = False

    @staticmethod
    def generate_slice(features: np.ndarray, selected: np.ndarray) -> np.ndarray:
        """
        Returns a boolean selector of features so that only features belonging
        to selected are set to True.
        """
        fslice = np.zeros(features.shape).astype(bool)
        for i, fname in enumerate(features):
            fslice[i] = fname in selected
        return fslice

    def retrieve_common_features(self, datasets: List[AnnData]) -> None:
        """
        Stores gene names for later use.
        """
        assert len(datasets) > 0, "No dataset provided."
        if self.mode == "pairwise":
            for i, adata_i in enumerate(datasets):
                features_i = adata_i.var_names.to_numpy()
                for j, adata_j in enumerate(datasets):
                    if j <= i:
                        continue
                    features_j = adata_j.var_names.to_numpy()
                    common_features = np.intersect1d(features_i, features_j)
                    assert (
                        common_features.shape[0] > 0
                    ), f"No common feature found between datasets {i} and {j}."
                    slice_i = UsesCommonFeatures.generate_slice(
                        features=features_i,
                        selected=common_features,
                    )
                    slice_j = UsesCommonFeatures.generate_slice(
                        features=features_j,
                        selected=common_features,
                    )
                    self.pairwise_feature_slices[i, j] = (slice_i, slice_j)
                    self.pairwise_feature_slices[j, i] = (slice_j, slice_i)
        elif self.mode == "total":
            common_features = datasets[0].var_names.to_numpy()
            for adata in datasets[1:]:
                common_features = np.intersect1d(
                    common_features,
                    adata.var_names.to_numpy(),
                )
            for adata in datasets:
                self.total_feature_slices.append(
                    UsesCommonFeatures.generate_slice(
                        adata.var_names.to_numpy(), common_features
                    )
                )
        else:
            raise ValueError(f"Unknown mode {self.mode}.")
        self.fitted = True

    def get_common_features(
        self, idx_1: int, idx_2: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns a tuple containing feature slices to use between two datasets
        identified as their index. Raises a ValueError if idx_1, idx_2 is unknown.
        """
        assert self.fitted, "UsesCommonFeatures trait has not retrieved features."
        if self.mode == "pairwise":
            slices = self.pairwise_feature_slices.get((idx_1, idx_2), None)
            if slices is None:
                raise ValueError(f"No feature slice found for {idx_1}, {idx_2}.")
        elif self.mode == "total":
            assert idx_1 < len(self.total_feature_slices), f"{idx_1} out of bounds."
            assert idx_2 < len(self.total_feature_slices), f"{idx_2} out of bounds."
            slices = self.total_feature_slices[idx_1], self.total_feature_slices[idx_2]
        else:
            raise ValueError(f"Unknown mode {self.mode}.")
        return slices

    def slice_features(
        self,
        X1: np.ndarray,
        idx_1: int,
        X2: Optional[np.ndarray] = None,
        idx_2: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns a sliced view of datasets X1 and X2, indexed by idx_1 and idx_2. Raises
        a ValueError if indices are not found, or if slice size does not coincidate.
        """
        if X2 is None and idx_2 is None:
            assert (
                self.mode == "total"
            ), "Calling slice_features with one dataset is only"
            " valid for mode == 'total'."
            return X1[self.total_feature_slices[idx_1]]
        assert X2 is not None and idx_2 is not None
        s1, s2 = self.get_common_features(idx_1, idx_2)
        assert s1.shape[0] == X1.shape[1], (
            f"Unexpected matrix features number. Expected {s1.shape[0]}, "
            f"found {X1.shape[1]}."
        )
        assert s2.shape[0] == X2.shape[1], (
            f"Unexpected matrix features number. Expected {s2.shape[0]}, "
            f"found {X2.shape[1]}."
        )
        return X1[:, s1], X2[:, s2]


class UsesReference:
    """
    This trait is shared by objects that must use a reference
    dataset.
    """

    def __init__(self):
        pass

    @staticmethod
    def get_reference_index(datasets: List[AnnData]) -> int:
        """
        Returns index of AnnData that has been chosen as a reference. If
        found none or several, returns -1.
        """
        ref_id = -1
        for k, adata in enumerate(datasets):
            is_ref = adm.get_value(adata, AnnDataKeyIdentifiers.IsReference)
            if is_ref is not None:
                if ref_id != -1:
                    return -1  # Several found
                ref_id = k
        return ref_id

    @staticmethod
    def get_reference(datasets: List[AnnData]) -> Optional[AnnData]:
        """
        Returns AnnData that has been chosen as a reference. If
        found none or several, returns None.
        """
        ref_id = UsesReference.get_reference_index(datasets)
        if ref_id == -1:
            return None
        return datasets[ref_id]


class UsesMetric:
    """
    Objects with this trait can set and get internal metrics of
    AnnData objects.
    """

    def __init__(self):
        pass

    @staticmethod
    def set_metric(
        adata: AnnData,
        metric: str,
        metric_kwargs: Optional[Dict] = None,
    ) -> None:
        """
        Set AnnData internal metric.
        """
        if metric_kwargs is None:
            metric_kwargs = {}
        adm.set_value(
            adata=adata,
            key=AnnDataKeyIdentifiers.Metric,
            field="uns",
            value=metric,
            persist="output",
        )
        adm.set_value(
            adata=adata,
            key=AnnDataKeyIdentifiers.MetricKwargs,
            field="uns",
            value=metric_kwargs,
            persist="output",
        )

    @staticmethod
    def get_metric(adata: AnnData) -> Optional[Tuple[str, Dict]]:
        """
        Returns metric and metric kwargs contained in anndata,
        or None if not set.
        """
        metric = adm.get_value(adata, AnnDataKeyIdentifiers.Metric)
        metric_kwargs = adm.get_value(adata, AnnDataKeyIdentifiers.MetricKwargs)
        if metric is None:
            return None
        if metric_kwargs is None:
            metric_kwargs = {}
        return metric, metric_kwargs
