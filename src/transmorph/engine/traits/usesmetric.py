#!/usr/bin/env python3

from anndata import AnnData
from typing import Dict, List, Optional, Tuple

from ...utils.anndata_manager import anndata_manager as adm, AnnDataKeyIdentifiers

_TypeMetric = Tuple[str, Dict]


class UsesMetric:
    """
    Objects with this trait can set and get internal metrics of
    AnnData objects.

    Parameters
    ----------
    default_metric: str, default = "sqeuclidean"
        Scipy-compatible metric to use if no metric is found in an AnnData object.

    default_kwargs: Dict, default = {}
        Additional parameters to provide to metric

    Attributes
    ----------
    _stored_metrics: List[Tuple[str, Dict]]
        List of cached metrics to use in methods that do not have direct access
        to AnnData objects.
    """

    def __init__(self, default_metric: str, default_kwargs: Optional[Dict] = None):
        self._default_metric = default_metric
        if default_kwargs is None:
            default_kwargs = {}
        self._default_metric_kwargs = default_kwargs
        self._stored_metrics: List[_TypeMetric] = []

    @staticmethod
    def set_adata_metric(
        adata: AnnData,
        metric: str,
        metric_kwargs: Optional[Dict] = None,
    ) -> None:
        """
        Sets AnnData internal metric inside a AnnData object. Modules which use
        internal metric will use this one in priority.

        Parameters
        ----------
        adata: AnnData
            AnnData object to write metric in.

        metric: str
            Scipy-compatible metric name.

        metric_kwargs: Optional[Dict]
            Optional additional parameters to provide to cdist() function.
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
    def get_adata_metric(adata: AnnData) -> Optional[_TypeMetric]:
        """
        Returns AnnData internal metric if set, None otherwise.

        Parameters
        ----------
            adata: AnnData to retrieve metric from.
        """
        metric = adm.get_value(adata, AnnDataKeyIdentifiers.Metric)
        if metric is None:
            return None
        kwargs = adm.get_value(adata, AnnDataKeyIdentifiers.MetricKwargs)
        if kwargs is None:
            kwargs = {}
        return metric, kwargs

    def retrieve_all_metrics(self, datasets: List[AnnData]) -> None:
        """
        Retrieves metrics from a list of AnnData datasets and saves them
        in objsct cache. If information is missing, fills with default value.

        Parameters
        ----------
        datasets: AnnData
            AnnData objects to extract metric information from.
        """
        for adata in datasets:
            adata_metric = UsesMetric.get_adata_metric(adata)
            if adata_metric is None:
                metric, kwargs = self._default_metric, self._default_metric_kwargs
            else:
                metric, kwargs = adata_metric
            self._stored_metrics.append((metric, kwargs))

    def get_metric(self, index: int) -> _TypeMetric:
        """
        Returns metric and metric kwargs contained in anndata, or default ones
        if information is missing.

        Parameters
        ----------
        index: int
            Dataset index corresponding to the list order passed to
            retrieve_all_metrics
        """
        assert index < len(
            self._stored_metrics
        ), f"{index} out of range, make sure retrieve_all_metrics have been called."
        return self._stored_metrics[index]
