#!/usr/bin/env python3

import logging
import numpy as np
import pandas as pd

from anndata import AnnData
from collections import namedtuple
from enum import Enum
from scipy.sparse import csr_matrix
from transmorph import logger
from .type import assert_type
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

_TypeAnnDataFields = Literal["obs", "var", "obsm", "varm", "obsp", "varp", "uns"]
_TypePairwiseSlice = Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]
_TypeTotalSlice = List[np.ndarray]
_TypePersistLevels = Literal["layer", "pipeline", "output"]


def generate_features_slice(features: np.ndarray, selected: np.ndarray) -> np.ndarray:
    """
    Returns a boolean selector of features so that only features belonging
    to selected are set to True.
    """
    fslice = np.zeros(features.shape).astype(bool)
    for i, fname in enumerate(features):
        fslice[i] = fname in selected
    return fslice


def get_pairwise_feature_slices(datasets: List[AnnData]) -> _TypePairwiseSlice:
    """
    Returns a dictionary where index (i, j) corresponds to boolean
    slices to use to put datasets i and j to the same feature space.
    """
    assert len(datasets) > 0, "No dataset provided."
    result: _TypePairwiseSlice = {}
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
            slice_i = generate_features_slice(
                features=features_i,
                selected=common_features,
            )
            slice_j = generate_features_slice(
                features=features_j,
                selected=common_features,
            )
            result[i, j] = (slice_i, slice_j)
            result[j, i] = (slice_j, slice_i)
    return result


def get_total_feature_slices(datasets: List[AnnData]) -> _TypeTotalSlice:
    """
    Returns a dictionary where index (i, j) corresponds to boolean
    slices to use to put datasets i and j to the same feature space.
    """
    assert len(datasets) > 0, "No dataset provided."
    result: _TypeTotalSlice = []
    common_features = datasets[0].var_names.to_numpy()
    for adata in datasets[1:]:
        common_features = np.intersect1d(
            common_features,
            adata.var_names.to_numpy(),
        )
    for adata in datasets:
        result.append(
            generate_features_slice(adata.var_names.to_numpy(), common_features)
        )
    return result


def slice_common_features(datasets: List[AnnData]) -> List[np.ndarray]:
    """
    Returns a list of AnnData objects in a common features space.
    """
    slices = get_total_feature_slices(datasets)
    return [adata.X[:, sl] for adata, sl in zip(datasets, slices)]


class AnnDataKeyIdentifiers(Enum):
    """
    String constants to pass to AnnDataManager to easily manage
    data storing in AnnData objects.
    """

    # Default representation keys
    BaseRepresentation = "base_representation"
    TransmorphRepresentation = "transmorph"

    # AnnData metadata
    AnnDataId = "adata_id"
    IsReference = "is_reference"
    Metric = "default_metric"
    MetricKwargs = "default_metric_kwargs"

    # Structural keys
    DistanceMatrix = "distance_matrix"
    SimilarityMatrix = "similarity_matrix"
    SubsamplingAnchors = "ssp_anchors"
    SubsamplingReferences = "ssp_references"

    # Plotting keys
    PlotRepresentation = "plot_representation"


AnnDataKey = namedtuple("AnnDataKey", ["identifier", "field", "persist"])


class AnnDataManager:
    """ """

    AnnDataId: int = 0

    def __init__(self):
        self.keys: Dict[Union[str, AnnDataKeyIdentifiers], AnnDataKey] = {}

    @staticmethod
    def gen_keystring(base: Union[str, AnnDataKeyIdentifiers]) -> str:
        """
        Adds a prefix to a given key to decrease collision cases
        with other packages.
        """
        if base == "transmorph":
            return "transmorph"
        return f"tr_{base}"

    @staticmethod
    def to_delete(query: _TypePersistLevels, target: _TypePersistLevels) -> bool:
        """
        Returns true if query level <= target level.
        """
        if target == "output":
            return True
        if target == "pipeline":
            return query != "output"
        if target == "layer":
            return query == "target"
        raise ValueError(f"Unknown target persist level: {target}.")

    @staticmethod
    def insert(
        field: Union[pd.DataFrame, Dict], field_str: str, str_key: str, value: Any
    ) -> None:
        """
        Inserts an entry to an AnnData component, and logs it.
        """
        AnnDataManager._log(f"Inserting {field_str} {str_key}.")
        field[str_key] = value

    @staticmethod
    def get(
        field: Union[pd.DataFrame, Dict], str_key: Union[str, AnnDataKeyIdentifiers]
    ) -> Any:
        """
        Retrieves information for an AnnData field, returns None if
        not present.
        """
        if str_key not in field:
            return None
        return field[str_key]

    @staticmethod
    def delete(field: Union[pd.DataFrame, Dict], field_str: str, str_key: str) -> None:
        """
        Deletes an AnnData entry if it is present, and logs it.
        """
        if str_key not in field:
            return
        AnnDataManager._log(f"Deleting {field_str} {str_key}.")
        del field[str_key]

    @staticmethod
    def _log(msg: str, level: int = logging.DEBUG) -> None:
        logger.log(level, f"ADManager > {msg}")

    def get_anndata_id(self, adata: AnnData) -> int:
        """
        Creates a new identifier for specified AnnData if necessary,
        then retrieves its identifier.
        """
        adata_id = self.get_anndata_id(adata)
        if adata_id is None:
            adata_id = AnnDataManager.AnnDataId
            self.set_value(adata, AnnDataKeyIdentifiers.AnnDataId, "uns", adata_id)
            AnnDataManager.AnnDataId += 1
        return adata_id

    def set_value(
        self,
        adata: AnnData,
        key: Union[str, AnnDataKeyIdentifiers],
        field: _TypeAnnDataFields,
        value: Any,
        persist: _TypePersistLevels = "pipeline",
    ) -> None:
        """
        Stores information in an AnnData object, with a few sanity
        checks.

        Parameters
        ----------
        adata: AnnData
            AnnData object to store information in.

        key: AnnDataKeys
            AnnDataKey identifier to safely store the information.

        field: Literal["obs", "var", "obsm", "varm", "obsp", "varp", "uns"]
            AnnData subfield to store the information.
            - "obs" vector-like data to label observations
            - "var" vector-like data to label features
            - "obsm" matrix-like data to represent observations
            - "varm" matrix-like data to represent features
            - "obsp" matrix-like data containing pairwise information
              between observations
            - "varp" matrix-like data containing pairwise information
              between features
            - "uns" any type of data to be stored (parameters, metadata,
              local metric...)

        value: Any
            Data to store.

        persist: Literal["layer", "pipeline", "output"], default = "pipeline"
            Life duration of the information.
            - "layer": Information is deleted at the end of the layer that
              created it.
            - "pipeline": Information is deleted at the end of the pipeline.
            - "output": Information is not deleted at the end of the pipeline.
        """
        str_key = AnnDataManager.gen_keystring(key)
        self.keys[key] = AnnDataKey(key, field, persist)
        if field == "uns":
            if "transmorph" not in adata.uns:
                adata.uns["transmorph"] = {}
            AnnDataManager.insert(adata.uns["transmorph"], "uns", str_key, value)
        elif field == "obs":
            assert_type(value, (np.ndarray, pd.Series))
            AnnDataManager.insert(adata.obs, "obs", str_key, value)
        elif field == "var":
            assert_type(value, (np.ndarray, pd.Series))
            AnnDataManager.insert(adata.var, "var", str_key, value)
        elif field == "obsm":
            assert_type(value, (np.ndarray, csr_matrix))
            assert value.shape[0] == adata.n_obs
            AnnDataManager.insert(adata.obsm, "obsm", str_key, value)
        elif field == "varm":
            assert_type(value, (np.ndarray, csr_matrix))
            assert value.shape[1] == adata.n_vars
            AnnDataManager.insert(adata.varm, "varm", str_key, value)
        elif field == "obsp":
            assert_type(value, (np.ndarray, csr_matrix))
            assert value.shape[0] == value.shape[1] == adata.n_obs
            AnnDataManager.insert(adata.obsp, "obsp", str_key, value)
        elif field == "varp":
            assert_type(value, (np.ndarray, csr_matrix))
            assert value.shape[0] == value.shape[1] == adata.n_vars
            AnnDataManager.insert(adata.varp, "varp", str_key, value)
        else:
            raise ValueError(f"Unrecognized field: {field}.")

    def isset_value(
        self,
        adata: AnnData,
        key: Union[str, AnnDataKeyIdentifiers],
        transmorph_key: bool = True,
        field: Optional[
            Literal["obs", "var", "obsm", "varm", "obsp", "varp", "uns"]
        ] = None,
    ) -> bool:
        """
        Detects if the desired key is contained in an AnnData object.
        """
        return self.get_value(adata, key, transmorph_key, field) is not None

    def get_value(
        self,
        adata: AnnData,
        key: Union[str, AnnDataKeyIdentifiers],
        transmorph_key: bool = True,
        field: Optional[
            Literal["obs", "var", "obsm", "varm", "obsp", "varp", "uns"]
        ] = None,
    ) -> Optional[Any]:
        """
        Retrieves value previously stored. Returns None if nothing is found.
        """
        if transmorph_key:
            str_key = AnnDataManager.gen_keystring(key)
            ad_key = self.keys.get(key, None)
            if ad_key is None:
                return None
            field = ad_key.field
        else:
            assert field is not None, "Field must be specified for non-transmorph keys."
            str_key = key
        if field == "obs":
            return AnnDataManager.get(adata.obs, str_key)
        if field == "var":
            return AnnDataManager.get(adata.var, str_key)
        if field == "obsm":
            return AnnDataManager.get(adata.obsm, str_key)
        if field == "varm":
            return AnnDataManager.get(adata.varm, str_key)
        if field == "obsp":
            return AnnDataManager.get(adata.obsp, str_key)
        if field == "varp":
            return AnnDataManager.get(adata.varp, str_key)
        if field == "uns":
            if "transmorph" not in adata.uns:
                return None
            return AnnDataManager.get(adata.uns["transmorph"], str_key)
        # We should not reach this
        logger.warning(f"WARNING - Unrecognized field {field}.")
        return None

    def clean(self, adata: AnnData, level: _TypePersistLevels) -> None:
        """
        Deletes transmorph keys of the given persist level and below.
        """
        for admkey in self.keys.values():
            key, field, persist = admkey
            str_key = AnnDataManager.gen_keystring(key)
            if not AnnDataManager.to_delete(persist, level):
                continue
            if field == "uns":
                if "transmorph" not in adata.uns:
                    continue
                AnnDataManager.delete(adata.uns["transmorph"], "uns", str_key)
                if not adata.uns["transmorph"]:  # Test empty dict
                    del adata.uns["transmorph"]
            elif field == "obs":
                AnnDataManager.delete(adata.obs, "obs", str_key)
            elif field == "var":
                AnnDataManager.delete(adata.var, "var", str_key)
            elif field == "obsm":
                AnnDataManager.delete(adata.obsm, "obsm", str_key)
            elif field == "varm":
                AnnDataManager.delete(adata.varm, "varm", str_key)
            elif field == "obsp":
                AnnDataManager.delete(adata.obsp, "obsp", str_key)
            elif field == "varp":
                AnnDataManager.delete(adata.varp, "varp", str_key)
            else:
                raise ValueError(f"Unrecognized field: {field}.")


anndata_manager = AnnDataManager()
