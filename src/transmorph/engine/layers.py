#!/usr/bin/env python3

from abc import abstractmethod
from anndata import AnnData

from typing import List, Type, Union

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
        profiler.task_end(tid)
        return result

    return wrapper


class LayerTransmorph:
    """
    A LayerTransmorph wraps an integration module, and manage its connections
    with other modules. All Layers derive from this class.

    Parameters
    ----------
    compatible_types: List[LayerType]
        List of type identifiers of compatible input layers

    verbose: bool, default = False
        Displays debugging information

    Attributes
    ----------
    output_layers: List[LayerTransmorph]
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
        compatible_inputs: List[Type] = [], #L Why Type as type and not LayerTransmorph because it is the same class ?
        verbose: bool = False,
    ) -> None:
        self.str_rep = ""
        self.compatible_inputs = compatible_inputs
        self.output_layers: List[LayerTransmorph] = [] # Why "" ?
        self.verbose = verbose
        self.profiler = None
        self.layer_id = LayerTransmorph.LayerID
        self.embedding_layer = None
        LayerTransmorph.LayerID += 1
        self._log("Initialized.")

    def _log(self, msg: str):
        if not self.verbose:
            return
        print(f"{self} >", msg)

    def __str__(self):
        if self.str_rep == "":
            typestr = "ABS"  # Abstract #L
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

    def connect(self, layer: "LayerTransmorph") -> None:
        """
        Connects the current layer to an output layer, if compatible.

        Parameters
        ----------
        layer: LayerTransmorph
            Output layer of compatible type.
        """
        assert (
            type(self) in layer.compatible_inputs
        ), f"Incompatible connection: {self} -> {layer}"
        assert layer not in self.output_layers, "{self} already connected to {layer}."
        self.output_layers.append(layer)
        self._log(f"Connected to layer {layer}.")

    @abstractmethod
    def fit(
        self, caller: "LayerTransmorph", datasets: List[AnnData]
    ) -> List["LayerTransmorph"]:
        """
        This is the computational method, running an internal module.
        It then should write its output in the AnnDatas, and callback
        the downstream fit() methods.

        Parameters
        ----------
        caller: LayerTransmorph
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


class LayerInput(LayerTransmorph):
    """
    Every pipeline must contain exactly one input layer, followed by an
    arbitrary network structure. Every pipeline is initialized using this
    input layer.
    """

    def __init__(self, verbose: bool = False) -> None:
        super().__init__(compatible_inputs=[], verbose=verbose)
        self.use_rep = ""

    @profile_method
    def fit(
        self, caller: LayerTransmorph, datasets: List[AnnData]
    ) -> List[LayerTransmorph]:
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


class LayerOutput(LayerTransmorph):
    """
    Simple layer to manage network outputs. There can be several output layers.
    (actually not for now)
    """

    def __init__(self, verbose: bool = False) -> None:
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
    def fit(
        self, caller: LayerTransmorph, datasets: List[AnnData]
    ) -> List[LayerTransmorph]:
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


class LayerMatching(LayerTransmorph):
    """
    This layer performs a matching between two or more datasets.
    It wraps an object derived from MatchingABC.
    """

    def __init__(self, matching: MatchingABC, verbose: bool = False) -> None:
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
    def fit(
        self, caller: LayerTransmorph, datasets: List[AnnData]
    ) -> List[LayerTransmorph]:
        self._log("Requesting keyword.")
        if self.embedding_layer is None:
            self.representation_kw = caller.get_representation()
        else:
            self.representation_kw = self.embedding_layer.get_representation()
        self._log(f"Found '{self.representation_kw}'. Calling matching.")
        self.matching.fit(datasets, self.representation_kw)
        self._log("Fitted.")
        return self.output_layers

    def clean(self, datasets: List[AnnData]):
        self._log("Cleaning.")
        for output in self.output_layers:
            output.clean(datasets)

    def get_representation(self) -> str:
        return self.representation_kw


class LayerMerging(LayerTransmorph):
    """
    This layer performs a merging between two or more datasets and their matchings.
    It wraps an object derived from MergingABC.
    """

    def __init__(self, merging: MergingABC, verbose: bool = False) -> None:
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
    def fit(
        self, caller: LayerMatching, datasets: List[AnnData]
    ) -> List[LayerTransmorph]:
        self._log("Requesting keyword.")
        if self.embedding_layer is None:
            representation_kw = caller.get_representation()
        else:
            representation_kw = self.embedding_layer.get_representation()
        self._log(f"Found '{representation_kw}'. Calling merging.")
        ref_id = -1
        if self.use_reference:
            for k, adata in enumerate(datasets):
                if get_info(adata, "is_reference"):
                    ref_id = k
                    break
            assert (
                ref_id != -1
            ), "Error: No reference found in TransmorphPipeline.fit()."
        X_transform = self.merging.fit(
            datasets,
            matching=caller.matching,
            X_kw=representation_kw,
            reference_idx=ref_id,
        )
        for adata, X_after in zip(datasets, X_transform):
            set_matrix(adata, self.mtx_id, X_after)
        self._log("Fitted.")
        return self.output_layers

    def clean(self, datasets: List[AnnData]):
        self._log("Cleaning.")
        for adata in datasets:
            delete_matrix(adata, self.mtx_id)
        for output in self.output_layers:
            output.clean(datasets)

    def get_representation(self) -> str:
        return self.mtx_id


class LayerChecking(LayerTransmorph):
    """
    Conditional layers with exactly two outputs. Performs a statistical test
    on its input data (typically the result of a merging), then
    > if accepted, calls layer_yes
    > if rejected, calls layer_no (possibly upstream)
    Useful to create "until convergence" loops.
    Encapsulates a CheckingABC module.
    """

    def __init__(
        self, checking: CheckingABC, n_checks_max: int = 10, verbose: bool = False
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
        self.layer_yes: Union[None, LayerTransmorph] = None
        self.layer_no: Union[None, LayerTransmorph] = None
        self.mtx_id = f"checking_{self.layer_id}"
        self.cleaned = False  # FIXME: investigate this to avoid looping

    def connect(self, layer: LayerTransmorph):
        raise NotImplementedError(
            "Please use instead connect_yes and connect_no for LayerChecking."
        )

    def connect_yes(self, layer: LayerTransmorph):
        assert self.layer_yes is None, "Error: Only one layer 'YES' is allowed."
        super().connect(layer)
        self.layer_yes = layer

    def connect_no(self, layer: LayerTransmorph):
        assert self.layer_no is None, "Error: Only one layer 'NO' is allowed."
        super().connect(layer)
        self.layer_no = layer

    @profile_method
    def fit(
        self, caller: LayerTransmorph, datasets: List[AnnData]
    ) -> List[LayerTransmorph]:
        assert self.layer_yes is not None, "Error: No layer found for 'YES' path."
        assert self.layer_no is not None, "Error: No layer found for 'NO' path."
        self._log("Requesting keyword.")
        if self.embedding_layer is None:
            representation_kw = caller.get_representation()
        else:
            representation_kw = self.embedding_layer.get_representation()
        self._log(f"Found '{representation_kw}'. Checking validity.")
        for adata in datasets:
            set_matrix(adata, self.mtx_id, get_matrix(adata, representation_kw))
        valid = self.checking.check(datasets, self.mtx_id)
        self._log(f"Thr: {self.checking.threshold}, Val: {self.checking.last_value}")
        self.n_checks += 1
        if valid or self.n_checks >= self.n_checks_max:
            if not valid:
                self._log("Warning, number of checks exceeded, validating by default.")
            self._log("Checking loop ended, pursuing.")
            return [self.layer_yes]
        else:
            self._log("Check fail, retrying.")
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


class LayerPreprocessing(LayerTransmorph):
    """
    This layer encapsulates a preprocessing algorithm derived
    from PreprocessingABC.
    """

    def __init__(
        self,
        preprocessing: PreprocessingABC,
        verbose: bool = False,
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
    def fit(
        self, caller: LayerTransmorph, datasets: List[AnnData]
    ) -> List[LayerTransmorph]:
        self._log("Requesting keyword.")
        if self.embedding_layer is None:
            self.representation_kw = caller.get_representation()
        else:
            self.representation_kw = self.embedding_layer.get_representation()
        self._log(f"Found '{self.representation_kw}'. Preprocessing.")
        Xs = self.preprocessing.transform(datasets, self.representation_kw)
        self.fitted = True
        for adata, X in zip(datasets, Xs):
            set_matrix(adata, self.mtx_id, X)
        self._log("Fitted.")
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
