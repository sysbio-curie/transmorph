#!/usr/bin/env python3

# All types are string by default
from __future__ import annotations

import logging
import numpy as np

from abc import abstractmethod
from anndata import AnnData
from scipy.sparse import csr_matrix
from transmorph import logger
from typing import List, Optional, Type, Union, TYPE_CHECKING
from warnings import warn

if TYPE_CHECKING:
    from .watchers import Watcher

from ..checking.checkingABC import CheckingABC
from ..matching.matchingABC import MatchingABC
from ..merging.mergingABC import MergingABC
from ..preprocessing.preprocessingABC import PreprocessingABC
from ..utils.anndata_interface import (
    delete_matrix,
    get_info,
    get_matrix,
    set_matrix,
)

from .profiler import Profiler


# Profiling decorator for class methods
# allows to measure time elapsed in a layer
def profile_method(method):
    def wrapper(*args):
        self = args[0]
        assert self is not None
        profiler: Profiler = self.profiler
        tid = profiler.task_start(str(self))
        result = method(*args)
        elapsed = profiler.task_end(tid)
        self.time_elapsed = elapsed
        return result

    return wrapper


class Layer:
    """
    A Layer wraps an integration module, and manage its connections
    with other modules. All Layers derive from this class.

    Parameters
    ----------
    compatible_inputs: List[Type]
        List of type identifiers of compatible input layers

    verbose: bool, default = False
        Displays debugging information

    Attributes
    ----------
    output_layers: List[Layer]
        Set of next layers in the pipeline.

    layer_id: int
        Unique integer id indentifying the layer.

    Todo
    ----
    TODO: cacheing
    TODO: toggle cleaning
    """

    LayerID = 0

    def __init__(
        self,
        compatible_inputs: List[Type] = [],
        verbose: Optional[bool] = None,
    ) -> None:
        self.str_rep = ""
        self.compatible_inputs = compatible_inputs
        self.output_layers: List[Layer] = []
        self.watchers: List[Watcher] = []
        self.preprocessings: List[PreprocessingABC] = []
        self.verbose = verbose
        self.profiler = None
        self.layer_id = Layer.LayerID
        self.embedding_layer = None
        self.time_elapsed = -1
        Layer.LayerID += 1
        self._log("Initialized.")

        if verbose is not None:
            warn(
                "Using 'verbose' parameter for layers is deprecated."
                "It uses TransmorphPipeline's verbose parameter instead."
            )

    def _log(self, msg: str, level: int = logging.DEBUG) -> None:
        logger.log(level, f"{self} > {msg}")

    def __str__(self) -> str:
        if self.str_rep == "":
            typestr = "ABS"  # Abstract
            if type(self) is LayerChecking:
                typestr = "CHK"
            elif type(self) is LayerInput:
                typestr = "INP"
            elif type(self) is LayerOutput:
                typestr = "OUT"
            elif type(self) is LayerMatching:
                typestr = "MTC"
            elif type(self) is LayerMerging:
                typestr = "MRG"
            elif type(self) is LayerPreprocessing:
                typestr = "PRP"
            else:
                raise NotImplementedError  # wth am I?
            self.str_rep = f"{typestr}#{self.layer_id}"
        return self.str_rep

    def set_verbose(self, verbose: bool = False) -> None:
        self.verbose = verbose
        for watcher in self.watchers:
            watcher.verbose = verbose
        for preprocessing in self.preprocessings:
            preprocessing.verbose = verbose

    def add_preprocessing(self, preprocessing: PreprocessingABC) -> None:
        self.preprocessings.append(preprocessing)

    def connect(self, layer: Layer) -> None:
        """
        Connects the current layer to an output layer, if compatible.

        Parameters
        ----------
        layer: Layer
            Output layer of compatible type.
        """
        assert (
            type(self) in layer.compatible_inputs
        ), f"Incompatible connection: {self} -> {layer}"
        assert layer not in self.output_layers, "{self} already connected to {layer}."
        self.output_layers.append(layer)
        self._log(f"Connected to layer {layer}.")

    def add_watcher(self, watcher: Watcher) -> None:
        """
        Adds a watcher to the layer to monitor it. Only the Watcher
        class should call this function, and is trusted to do so.
        """
        assert watcher not in self.watchers
        self.watchers.append(watcher)

    @abstractmethod
    def fit(self, caller: Layer, datasets: List[AnnData]) -> List[Layer]:
        """
        This is the computational method, running an internal module.
        It then should write its output in the AnnDatas, and callback
        the downstream fit() methods.

        Parameters
        ----------
        caller: Layer
            Reference to current layer, used to retrieve relevant information
            relative to computation results.

        datasets: List[AnnData]
            List of datasets to process.
        """
        pass

    def clean(self, datasets: List[AnnData]):
        """
        Deletes information written in AnnDatas, then delegate the
        rest to next layers.
        """
        raise NotImplementedError

    def get_representation(self) -> str:
        """
        Provides a keyword to the latest matrix representation of
        datasets.
        """
        raise NotImplementedError

    def set_profiler(self, profiler: Profiler):
        self._log("Connecting to profiler.")
        self.profiler = profiler

    def set_embedding_reference(self, layer):
        self.embedding_layer = layer

    def get_time_spent(self) -> float:
        return self.time_elapsed


class LayerInput(Layer):
    """
    Every pipeline must contain exactly one input layer, followed by an
    arbitrary network structure. Every pipeline is initialized using this
    input layer.
    """

    def __init__(self, verbose: Optional[bool] = None) -> None:
        super().__init__(compatible_inputs=[], verbose=verbose)
        self.use_rep = ""

    @profile_method
    def fit(self, caller: Layer, datasets: List[AnnData]) -> List[Layer]:
        """
        Simply calls the downstream layers.
        """
        assert caller is None, f"{caller} called {self}."
        self._log("Calling next layers.")
        return self.output_layers

    def clean(self, datasets: List[AnnData]):
        self._log("Cleaning.")
        for output in self.output_layers:
            output.clean(datasets)

    def get_representation(self) -> str:
        """
        Returns a matrix representation of AnnData.
        """
        return self.use_rep

    def add_preprocessing(self, preprocessing: PreprocessingABC) -> None:
        raise NotImplementedError


class LayerOutput(Layer):
    """
    Simple layer to manage network outputs. There can be several output layers.
    (actually not for now)
    """

    def __init__(self, verbose: Optional[bool] = None) -> None:
        super().__init__(
            compatible_inputs=[
                LayerChecking,
                LayerInput,
                LayerMerging,
                LayerPreprocessing,
            ],
            verbose=verbose,
        )
        self.representation_kw = ""

    @profile_method
    def fit(self, caller: Layer, datasets: List[AnnData]) -> List[Layer]:
        """
        Runs the upstream pipeline and stores results in AnnData objects.
        """
        self._log("Retrieving keyword.")
        if self.embedding_layer is None:
            self.representation_kw = caller.get_representation()
        else:
            self.representation_kw = self.embedding_layer.get_representation()
        self._log(f"Found '{self.representation_kw}'. Terminating the branch.")
        return []

    def clean(self, datasets: List[AnnData]):
        self._log("Cleaning ended for this branch.")

    def get_representation(self) -> str:
        return self.representation_kw

    def add_preprocessing(self, preprocessing: PreprocessingABC) -> None:
        raise NotImplementedError


class LayerMatching(Layer):
    """
    This layer performs a matching between two or more datasets.
    It wraps an object derived from MatchingABC.
    """

    def __init__(self, matching: MatchingABC, verbose: Optional[bool] = None) -> None:
        super().__init__(
            compatible_inputs=[
                LayerChecking,
                LayerInput,
                LayerMerging,
                LayerPreprocessing,
            ],
            verbose=verbose,
        )
        self.matching = matching
        self.representation_kw = ""

    @profile_method
    def fit(self, caller: Layer, datasets: List[AnnData]) -> List[Layer]:
        self._log("Requesting keyword.")
        if self.embedding_layer is None:
            self.representation_kw = caller.get_representation()
        else:
            self.representation_kw = self.embedding_layer.get_representation()
        self._log(f"Found '{self.representation_kw}'")
        self._log("Calling matching.", level=logging.INFO)
        for preprocessing in self.preprocessings:
            self.matching.add_preprocessing(preprocessing)
        self.matching.fit(datasets, self.representation_kw)
        self._log("Fitted.", level=logging.INFO)
        return self.output_layers

    def clean(self, datasets: List[AnnData]):
        self._log("Cleaning.")
        for output in self.output_layers:
            output.clean(datasets)

    def get_matching(self, adata1: AnnData, adata2: AnnData) -> csr_matrix:
        return self.matching.get_matching(adata1, adata2)

    def get_anchors(self, adata: AnnData) -> np.ndarray:
        return self.matching.get_anchors(adata)

    def get_representation(self) -> str:
        return self.representation_kw

    def set_verbose(self, verbose: bool = False) -> None:
        super().set_verbose(verbose)
        self.matching.verbose = verbose  # FIXME all this is dirty
        self.matching.subsampling.verbose = verbose


class LayerMerging(Layer):
    """
    This layer performs a merging between two or more datasets and their matchings.
    It wraps an object derived from MergingABC.
    """

    def __init__(self, merging: MergingABC, verbose: Optional[bool] = None) -> None:
        """
        TODO
        """
        super().__init__(
            compatible_inputs=[LayerMatching],
            verbose=verbose,
        )
        self.merging = merging
        self.use_reference = merging.use_reference
        self.mtx_id = f"merging_{self.layer_id}"  # To write results

    @profile_method
    def fit(self, caller: LayerMatching, datasets: List[AnnData]) -> List[Layer]:
        self._log("Requesting keyword.")
        if self.embedding_layer is None:
            representation_kw = caller.get_representation()
        else:
            representation_kw = self.embedding_layer.get_representation()
        self._log(f"Found '{representation_kw}'.")
        self._log("Running preprocessing...", level=logging.INFO)
        ref_id = -1
        if self.use_reference:
            for k, adata in enumerate(datasets):
                if get_info(adata, "is_reference"):
                    ref_id = k
                    break
            assert (
                ref_id != -1
            ), "Error: No reference found in TransmorphPipeline.fit()."
        for preprocessing in self.preprocessings:
            representations = preprocessing.transform(datasets, representation_kw)
            for adata, X in zip(datasets, representations):
                set_matrix(adata, "tmp", X)
            representation_kw = "tmp"
        self._log("Running merging...", level=logging.INFO)
        X_transform = self.merging.fit(
            datasets,
            matching=caller.matching,
            X_kw=representation_kw,
            reference_idx=ref_id,
        )
        for adata, X_after in zip(datasets, X_transform):
            set_matrix(adata, self.mtx_id, X_after)
        self._log("Fitted.", level=logging.INFO)
        return self.output_layers

    def clean(self, datasets: List[AnnData]):
        self._log("Cleaning.")
        for adata in datasets:
            delete_matrix(adata, self.mtx_id)
        for output in self.output_layers:
            output.clean(datasets)

    def get_representation(self) -> str:
        return self.mtx_id


class LayerChecking(Layer):
    """
    Conditional layers with exactly two outputs. Performs a statistical test
    on its input data (typically the result of a merging), then
    > if accepted, calls layer_yes
    > if rejected, calls layer_no (possibly upstream)
    Useful to create "until convergence" loops.
    Encapsulates a CheckingABC module.
    """

    def __init__(
        self,
        checking: CheckingABC,
        n_checks_max: int = 10,
        verbose: Optional[bool] = None,
    ) -> None:
        super().__init__(
            compatible_inputs=[
                LayerInput,
                LayerChecking,
                LayerMerging,
                LayerPreprocessing,
            ],  # Test CHECK
            verbose=verbose,
        )
        self.checking = checking
        self.n_checks = 0
        self.n_checks_max = n_checks_max
        self.layer_yes: Union[None, Layer] = None
        self.layer_no: Union[None, Layer] = None
        self.mtx_id = f"checking_{self.layer_id}"
        self.cleaned = False  # FIXME: investigate this to avoid looping

    def connect(self, layer: Layer):
        raise NotImplementedError(
            "Please use instead connect_yes and connect_no for LayerChecking."
        )

    def connect_yes(self, layer: Layer):
        assert self.layer_yes is None, "Error: Only one layer 'YES' is allowed."
        super().connect(layer)
        self.layer_yes = layer

    def connect_no(self, layer: Layer):
        assert self.layer_no is None, "Error: Only one layer 'NO' is allowed."
        super().connect(layer)
        self.layer_no = layer

    @profile_method
    def fit(self, caller: Layer, datasets: List[AnnData]) -> List[Layer]:
        assert self.layer_yes is not None, "Error: No layer found for 'YES' path."
        assert self.layer_no is not None, "Error: No layer found for 'NO' path."
        self._log("Requesting keyword.")
        if self.embedding_layer is None:
            representation_kw = caller.get_representation()
        else:
            representation_kw = self.embedding_layer.get_representation()
        self._log(f"Found '{representation_kw}'.")
        self._log("Checking representation validity...", level=logging.INFO)
        for adata in datasets:
            set_matrix(adata, self.mtx_id, get_matrix(adata, representation_kw))
        valid = self.checking.check(datasets, self.mtx_id)
        self._log(f"Thr: {self.checking.threshold}, Val: {self.checking.last_value}")
        self.n_checks += 1
        if valid or self.n_checks >= self.n_checks_max:
            if not valid:
                prefix = "Maximum number of checks reached."
            else:
                prefix = "Checking is valid."
            self._log(f"{prefix} Pursuing.", level=logging.INFO)
            return [self.layer_yes]
        else:
            self._log("Check fail, retrying.", level=logging.INFO)
            return [self.layer_no]

    def clean(self, datasets: List[AnnData]):
        if self.cleaned:
            return
        self.cleaned = True
        self._log("Cleaning.")
        for adata in datasets:
            delete_matrix(adata, self.mtx_id)
        for output in self.output_layers:
            output.clean(datasets)

    def get_representation(self) -> str:
        return self.mtx_id

    def add_preprocessing(self, preprocessing: PreprocessingABC) -> None:
        raise NotImplementedError


class LayerPreprocessing(Layer):
    """
    This layer encapsulates a preprocessing algorithm derived
    from PreprocessingABC.
    """

    def __init__(
        self,
        preprocessing: PreprocessingABC,
        verbose: Optional[bool] = None,
    ) -> None:
        super().__init__(
            compatible_inputs=[
                LayerInput,
                LayerPreprocessing,
                LayerMerging,
                LayerChecking,
            ],
            verbose=verbose,
        )
        self.preprocessing = preprocessing
        self.mtx_id = f"preprocessing_{self.layer_id}"

    @profile_method
    def fit(self, caller: Layer, datasets: List[AnnData]) -> List[Layer]:
        self._log("Requesting keyword.")
        if self.embedding_layer is None:
            self.representation_kw = caller.get_representation()
        else:
            self.representation_kw = self.embedding_layer.get_representation()
        self._log(f"Found '{self.representation_kw}'.")
        self._log("Preprocessing...", level=logging.INFO)
        Xs = self.preprocessing.transform(datasets, self.representation_kw)
        self.fitted = True
        for adata, X in zip(datasets, Xs):
            set_matrix(adata, self.mtx_id, X)
        self._log("Fitted.", level=logging.INFO)
        return self.output_layers

    def clean(self, datasets: List[AnnData]):
        self._log("Cleaning.")
        for adata in datasets:
            delete_matrix(adata, self.mtx_id)
        for output in self.output_layers:
            output.clean(datasets)

    def get_representation(self) -> str:
        assert self.fitted, "{self} must be fitted to access its representation."
        return self.mtx_id

    def add_preprocessing(self, preprocessing: PreprocessingABC) -> None:
        raise NotImplementedError
