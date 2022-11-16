#!/usr/bin/env python3

import logging
import numpy as np
import pandas as pd

from anndata import AnnData
from collections import namedtuple
from enum import Enum
from scipy.sparse import csr_matrix
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from .._logging import logger

_TypeAnnDataFields = Literal["obs", "var", "obsm", "varm", "obsp", "varp", "uns"]
_TypePairwiseSlice = Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]
_TypeTotalSlice = List[np.ndarray]
_TypePersistLevels = Literal["layer", "pipeline", "output"]


def generate_features_slice(features: np.ndarray, selected: np.ndarray) -> np.ndarray:
    """
    Returns an integer selector to slice {features} to {selected}
    """
    assert (
        np.unique(features).shape[0] == features.shape[0]
    ), "Duplicated features detected. Please fix this ambiguity."
    assert (
        np.unique(selected).shape[0] == selected.shape[0]
    ), "Duplicated selected features detected. Please report this issue."
    fslice = np.zeros(selected.shape).astype(int)
    for i, fname in enumerate(selected):
        feature_idx = np.where(features == fname)[0]
        assert feature_idx.shape[0] == 1, f"Missing features: {fname}."
        fslice[i] = np.where(features == fname)[0]  # Yuk, could be done better
    return fslice


def get_pairwise_feature_slices(datasets: List[AnnData]) -> _TypePairwiseSlice:
    """
    Returns a dictionary where index (i, j) corresponds to integer
    slices to use to put datasets i and j to the same feature space.
    """
    result: _TypePairwiseSlice = {}
    for i, adata_i in enumerate(datasets):
        features_i = adata_i.var_names.to_numpy()
        slice_ii = generate_features_slice(
            features=adata_i.var_names.to_numpy(),
            selected=np.sort(adata_i.var_names.to_numpy()),
        )
        result[i, i] = (slice_ii, slice_ii)
        for j, adata_j in enumerate(datasets):
            if j <= i:
                continue
            features_j = adata_j.var_names.to_numpy()
            common_features = np.intersect1d(features_i, features_j)
            common_features = np.sort(common_features)
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
    Returns a list where array at index i corresponds to boolean
    slice to use to slice dataset i in a common feature space.
    """
    result: _TypeTotalSlice = []
    if len(datasets) == 0:
        return []
    common_features = datasets[0].var_names.to_numpy()
    for adata in datasets[1:]:
        common_features = np.intersect1d(
            common_features,
            adata.var_names.to_numpy(),
        )
    common_features = np.sort(common_features)
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


# FIXME: this will cause issues in the future.
# It was bad design
class AnnDataKeyIdentifiers(Enum):
    """
    String constants to pass to AnnDataManager to easily manage
    data storing in AnnData objects.

    tr_... suggests the key is internal to the engine, and
    helps avoiding key collisions with other packages.
    """

    # Default representation keys
    BaseRepresentation = "tr_base_representation"
    TransmorphRepresentation = "X_transmorph"

    # AnnData metadata
    AnnDataId = "tr_adata_id"
    IsReference = "tr_is_reference"
    Metric = "tr_default_metric"
    MetricKwargs = "tr_default_metric_kwargs"

    # Structural keys
    DistanceMatrix = "tr_distance_matrix"
    SimilarityMatrix = "tr_similarity_matrix"
    SubsamplingAnchors = "tr_ssp_anchors"
    SubsamplingReferences = "tr_ssp_references"

    # Plotting keys
    PlotRepresentation = "tr_plot_representation"


AnnDataKey = namedtuple("AnnDataKey", ["identifier", "field", "persist"])


class AnnDataManager:
    """
    This class allows to safely handle AnnData objects, either through
    its static methods or using the global anndata manager object.
    """

    def __init__(self):
        adataid_str = AnnDataKeyIdentifiers.AnnDataId.value
        self.keys: Dict[str, AnnDataKey] = {
            adataid_str: AnnDataKey(adataid_str, "uns", "pipeline")
        }
        self.current_id = 0

    @staticmethod
    def log(msg: str, level: int = logging.DEBUG) -> None:
        logger.log(level, f"ADManager > {msg}")

    @staticmethod
    def gen_keystring(key: Union[str, AnnDataKeyIdentifiers]) -> str:
        """
        Adds a prefix to a given key to decrease collision cases
        with other packages.
        """
        if isinstance(key, AnnDataKeyIdentifiers):
            return key.value
        return key

    @staticmethod
    def to_delete(query: _TypePersistLevels, target: _TypePersistLevels) -> bool:
        """
        Returns true if query level <= target level.
        """
        levels = ["layer", "pipeline", "output"]
        assert query in levels, f"Unknown query level: {query}"
        assert target in levels, f"Unknown target level: {target}"
        if target == "output":
            return True
        if target == "pipeline":
            return query != "output"
        if target == "layer":
            return query == target
        raise ValueError(f"Unknown target persist level: {target}.")

    @staticmethod
    def get(field: Union[pd.DataFrame, Dict], str_key: str) -> Any:
        """
        Retrieves information for an AnnData field, returns None if
        not present.
        """
        if str_key not in field:
            return None
        return field[str_key]

    @staticmethod
    def delete(field: Union[pd.DataFrame, Dict], str_key: str) -> None:
        """
        Deletes an AnnData entry if it is present, and logs it.
        """
        if str_key not in field:
            return
        del field[str_key]

    @staticmethod
    def get_field_from_str(adata: AnnData, field_str: str) -> Union[pd.DataFrame, Dict]:
        """
        Returns field {field_str} of {adata}. Raises an exception if {field_str} is
        invalid.
        """
        if field_str == "obs":
            return adata.obs
        if field_str == "var":
            return adata.var
        if field_str == "obsm":
            return adata.obsm
        if field_str == "varm":
            return adata.varm
        if field_str == "obsp":
            return adata.obsp
        if field_str == "varp":
            return adata.varp
        if field_str == "uns":
            if "transmorph" not in adata.uns:
                adata.uns["transmorph"] = {}
            return adata.uns["transmorph"]
        raise ValueError(f"Unknown field {field_str}.")

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

        value: Any
            Data to store.

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


        persist: Literal["layer", "pipeline", "output"], default = "pipeline"
            Life duration of the information.
            - "layer": Information is deleted at the end of the layer that
              created it.
            - "pipeline": Information is deleted at the end of the pipeline.
            - "output": Information is not deleted at the end of the pipeline.
        """
        str_key = AnnDataManager.gen_keystring(key)
        if str_key not in self.keys:
            self.keys[str_key] = AnnDataKey(str_key, field, persist)
        else:  # We check consistance
            _, old_field, old_persist = self.keys[str_key]
            assert old_field == field
            assert old_persist == persist
        field_obj = AnnDataManager.get_field_from_str(adata, field)
        field_obj[str_key] = value
        str_descriptor = value
        if isinstance(value, (np.ndarray, csr_matrix)):
            str_descriptor = f"{type(value)} ({value.shape})"
        AnnDataManager.log(f"Inserting {str_descriptor} in {field}[{str_key}]")

    def get_value(
        self,
        adata: AnnData,
        key: Union[str, AnnDataKeyIdentifiers],
        field_str: Optional[_TypeAnnDataFields] = None,
    ) -> Optional[Any]:
        """
        Retrieves value previously stored. Returns None if nothing is found.

        Parameters
        ----------
        adata: AnnData
            AnnData object to retrieve information from.

        key: AnnDataKeys
            Key to retrieve. if it was stored by transmorph, the field will
            be remembered. Otherwise, you must provide it explicitly.

        field_str: Literal["obs", "var", "obsm", "varm", "obsp", "varp", "uns"]
            AnnData subfield containing the information.
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
        """
        str_key = AnnDataManager.gen_keystring(key)

        # By default, ADKI.BaseRepresentation returns .X if not set.
        base_repr = AnnDataKeyIdentifiers.BaseRepresentation.value
        if str_key == base_repr and AnnDataManager.get(adata.obsm, base_repr) is None:
            return adata.X

        if str_key in self.keys:
            field_str = self.keys[str_key].field
        if field_str is None:
            return None

        field = AnnDataManager.get_field_from_str(adata, field_str)
        return AnnDataManager.get(field, str_key)

    def isset_value(
        self,
        adata: AnnData,
        key: Union[str, AnnDataKeyIdentifiers],
        field: Optional[_TypeAnnDataFields] = None,
    ) -> bool:
        """
        Detects if a key is stored in an AnnData object.

        Parameters
        ----------
        adata: AnnData
            AnnData object to retrieve information from.

        key: AnnDataKeys
            Key to retrieve. if it was stored by transmorph, the field will
            be remembered. Otherwise, you must provide it explicitly.

        field_str: Literal["obs", "var", "obsm", "varm", "obsp", "varp", "uns"]
            AnnData subfield containing the information.
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
        """
        return self.get_value(adata, key, field) is not None

    def clean(
        self,
        datasets: Union[AnnData, List[AnnData]],
        level: _TypePersistLevels = "pipeline",
    ) -> None:
        """
        Deletes transmorph keys of the given persist level and below.

        Parameters
        ----------
        datasets: Union[AnnData, List[AnnData]]
            AnnData object(s) to clean.

        level: Literal["output", "pipeline", "layer"]
            All values with persist below this {level} are deleted.
        """
        if isinstance(datasets, AnnData):
            datasets = [datasets]

        for adata in datasets:
            for admkey in self.keys.values():
                key, field_str, persist = admkey
                str_key = AnnDataManager.gen_keystring(key)
                if not AnnDataManager.to_delete(persist, level):
                    continue
                AnnDataManager.log(f"Deleting entry {field_str}[{str_key}].")
                field = AnnDataManager.get_field_from_str(adata, field_str)
                AnnDataManager.delete(field, str_key)
                if field_str == "uns" and field == {}:
                    del adata.uns["transmorph"]

    def get_anndata_id(self, adata: AnnData) -> int:
        """
        Creates a new identifier for specified AnnData if necessary,
        then retrieves its identifier.
        """
        adata_id = self.get_value(adata, AnnDataKeyIdentifiers.AnnDataId)
        if adata_id is None:
            adata_id = self.current_id
            self.set_value(
                adata,
                AnnDataKeyIdentifiers.AnnDataId,
                "uns",
                adata_id,
            )
            self.current_id += 1
        return adata_id


anndata_manager = AnnDataManager()
