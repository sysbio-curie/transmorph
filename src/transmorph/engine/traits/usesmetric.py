#!/usr/bin/env python3

from anndata import AnnData
from typing import Dict, Optional, Tuple

from ...utils import anndata_manager as adm, AnnDataKeyIdentifiers


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
