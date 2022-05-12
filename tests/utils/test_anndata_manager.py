#!/usr/bin/env python3

import anndata
import numpy as np
import pytest
import random

from transmorph.utils.anndata_manager import (
    AnnDataKeyIdentifiers as adk,
    AnnDataManager as Adm,
    generate_features_slice,
    get_pairwise_feature_slices,
    get_total_feature_slices,
    slice_common_features,
)
from transmorph.utils.misc import rand_str

NADATAS = 5
NTRIES = 100
MAX_ITER = 500

adm = Adm()


def generate_features(
    n: int = 100,
    ln: int = 8,
    leave_duplicates: bool = False,
) -> np.ndarray:
    # Helper function generating random features
    features = np.array([rand_str(ln) for _ in range(n)])
    if leave_duplicates:
        return features
    return np.unique(features)


def generate_anndata(
    nobs: int,
    nvars: int,
    features_set: np.ndarray,
    force_shape: bool = False,
):
    # Helper function generating random anndatas of size approx.
    # nobs x nvars
    for _ in range(MAX_ITER):
        features = np.unique(random.choices(features_set, k=nvars))
        X = np.random.random(size=(nobs, features.shape[0]))
        adata = anndata.AnnData(X, dtype=X.dtype)
        adata.var[None] = features
        adata.var = adata.var.set_index(None)
        if not force_shape or adata.X.shape == (nobs, nvars):
            return adata
    raise ValueError("Could not generate an AnnData of the requested shape.")


def test_generate_feature_slice_success():
    # Tests feature slice selection on correct inputs
    for _ in range(NTRIES):
        features = generate_features(500, 8)
        selected = np.unique(random.choices(features, k=100))
        fsliced = features[generate_features_slice(features, selected)]
        assert fsliced.shape[0] == selected.shape[0]
        np.testing.assert_array_equal(np.sort(fsliced), np.sort(selected))


def test_generate_feature_slice_failure():
    # Tests feature slice selection on incorrect inputs

    # Duplicates in features
    for _ in range(NTRIES):
        features = generate_features(500, 8, leave_duplicates=True)
        selected = np.unique(random.choices(features, k=100))
        if np.unique(features).shape == features.shape:
            continue
        with pytest.raises(AssertionError):
            generate_features_slice(features, selected)

    # Duplicates in selection
    for _ in range(NTRIES):
        features = generate_features(500, 8)
        selected = np.array(random.choices(features, k=100))
        if np.unique(selected).shape == selected.shape:
            continue
        with pytest.raises(AssertionError):
            generate_features_slice(features, selected)

    # Missing features
    for _ in range(NTRIES):
        features = generate_features(500, 8)
        selected = generate_features(100, 8)
        with pytest.raises(AssertionError):
            generate_features_slice(features, selected)


def test_get_pairwise_feature_slices():
    # Tests correct pairwise feature slices between anndatas

    # No dataset
    assert get_pairwise_feature_slices([]) == {}

    # Existing intersection
    for _ in range(NTRIES):
        features = generate_features(500, 8)
        adatas = [generate_anndata(100, 200, features) for _ in range(NADATAS)]
        slices = get_pairwise_feature_slices(adatas)
        for i in range(NADATAS):
            si1, si2 = slices[i, i]
            assert si1.shape == si2.shape
            np.testing.assert_array_equal(si1, si2)
            for j in range(i + 1, NADATAS):
                si, sj = slices[i, j]
                np.testing.assert_array_equal(
                    adatas[i].var_names.to_numpy()[si],
                    adatas[j].var_names.to_numpy()[sj],
                )
                sj, si = slices[j, i]
                np.testing.assert_array_equal(
                    adatas[i].var_names.to_numpy()[si],
                    adatas[j].var_names.to_numpy()[sj],
                )

    # No intersection
    for _ in range(NTRIES):
        adatas = [
            generate_anndata(100, 200, generate_features(500, i + 4))
            for i in range(NADATAS)
        ]
        slices = get_pairwise_feature_slices(adatas)
        for i in range(NADATAS):
            si1, si2 = slices[i, i]
            assert si1.shape == si2.shape
            np.testing.assert_array_equal(si1, si2)
            for j in range(i + 1, NADATAS):
                si, sj = slices[i, j]
                np.testing.assert_array_equal(
                    adatas[i].var_names.to_numpy()[si],
                    adatas[j].var_names.to_numpy()[sj],
                )
                assert si.shape[0] == 0
                assert sj.shape[0] == 0
                sj, si = slices[j, i]
                np.testing.assert_array_equal(
                    adatas[i].var_names.to_numpy()[si],
                    adatas[j].var_names.to_numpy()[sj],
                )
                assert si.shape[0] == 0
                assert sj.shape[0] == 0


def test_get_total_feature_slices():
    # Tests correct pairwise feature slices between anndatas

    # No dataset
    assert get_total_feature_slices([]) == []

    # Existing intersection
    for _ in range(NTRIES):
        features = generate_features(500, 8)
        adatas = [generate_anndata(100, 200, features) for _ in range(NADATAS)]
        slices = get_total_feature_slices(adatas)
        for i in range(NADATAS):
            for j in range(i + 1, NADATAS):
                si, sj = slices[i], slices[j]
                np.testing.assert_array_equal(
                    adatas[i].var_names.to_numpy()[si],
                    adatas[j].var_names.to_numpy()[sj],
                )

    # No intersection
    for _ in range(NTRIES):
        adatas = [
            generate_anndata(100, 200, generate_features(500, i + 4))
            for i in range(NADATAS)
        ]
        slices = get_total_feature_slices(adatas)
        for i in range(NADATAS):
            si1, si2 = slices[i], slices[i]
            assert si1.shape == si2.shape
            np.testing.assert_array_equal(si1, si2)
            for j in range(i + 1, NADATAS):
                si, sj = slices[i], slices[j]
                np.testing.assert_array_equal(
                    adatas[i].var_names.to_numpy()[si],
                    adatas[j].var_names.to_numpy()[sj],
                )
                assert si.shape[0] == 0
                assert sj.shape[0] == 0


def test_slice_common_features():
    # Similar to previous test, testing matrices instead

    # No dataset
    assert slice_common_features([]) == []

    # Existing intersection
    for _ in range(NTRIES):
        features = generate_features(500, 8)
        adatas = [generate_anndata(100, 200, features) for _ in range(NADATAS)]
        slices = get_total_feature_slices(adatas)
        matrices = slice_common_features(adatas)
        assert len(slices) == len(matrices)
        for i in range(NADATAS):
            for j in range(i + 1, NADATAS):
                Xi, Xj = matrices[i], matrices[j]
                assert Xi.shape[1] == Xj.shape[1]
                Xi_true, Xj_true = (
                    adatas[i].X[:, slices[i]],
                    adatas[j].X[:, slices[j]],
                )
                np.testing.assert_array_equal(Xi, Xi_true)
                np.testing.assert_array_equal(Xj, Xj_true)

    # No intersection
    for _ in range(NTRIES):
        adatas = [
            generate_anndata(100, 200, generate_features(500, i + 4))
            for i in range(NADATAS)
        ]
        matrices = slice_common_features(adatas)
        for i in range(NADATAS):
            for j in range(i + 1, NADATAS):
                Xi, Xj = matrices[i], matrices[j]
                assert Xi.shape[1] == Xj.shape[1] == 0


@pytest.mark.parametrize(
    "key,keystring",
    [
        (adk.BaseRepresentation, adk.BaseRepresentation.value),
        (adk.BaseRepresentation.value, adk.BaseRepresentation.value),
        (adk.TransmorphRepresentation.value, adk.TransmorphRepresentation.value),
        ("renalla", "renalla"),
    ],
)
def test_anndata_manager_gen_keystring(key, keystring):
    # Tests conversion key -> str and str -> str
    assert Adm.gen_keystring(key) == keystring


@pytest.mark.parametrize(
    "query,target,expected",
    [
        pytest.param("caria", "output", True, marks=pytest.mark.xfail),
        pytest.param("output", "caria", True, marks=pytest.mark.xfail),
        ("output", "output", True),
        ("pipeline", "output", True),
        ("layer", "output", True),
        ("output", "pipeline", False),
        ("pipeline", "pipeline", True),
        ("layer", "pipeline", True),
        ("output", "layer", False),
        ("pipeline", "layer", False),
        ("layer", "layer", True),
    ],
)
def test_anndata_manager_to_delete(query, target, expected):
    # Tests persistence levels
    assert Adm.to_delete(query, target) == expected


@pytest.mark.parametrize(
    "field_str,str_key,value",
    [
        ("obs", "test_obs", np.random.random(size=(100,))),
        ("var", "test_var", np.random.random(size=(50))),
        ("obsm", "test_obsm", np.random.random(size=(100, 20))),
        ("varm", "test_varm", np.random.random(size=(50, 20))),
        ("obsp", "test_obsp", np.random.random(size=(100, 100))),
        ("varp", "test_varp", np.random.random(size=(50, 50))),
        ("uns", "test_uns", 42),
    ],
)
def test_anndata_manager_insert_get_delete(field_str, str_key, value):
    # Tests correct insertion/retrieval
    features = generate_features(n=500)
    adata = generate_anndata(100, 50, features, force_shape=True)
    field = Adm.get_field_from_str(adata, field_str)
    field[str_key] = value
    v_after = Adm.get(field, str_key)
    if isinstance(value, np.ndarray):
        np.testing.assert_array_equal(value, v_after)
    else:
        assert value == v_after
    Adm.delete(field, str_key)
    assert Adm.get(field, str_key) is None


@pytest.mark.parametrize(
    "field_str,str_key,value,persist",
    [
        ("obs", "test_obs", np.random.random(size=(100,)), "layer"),
        ("var", "test_var", np.random.random(size=(50)), "layer"),
        ("obsm", "test_obsm", np.random.random(size=(100, 20)), "pipeline"),
        ("varm", "test_varm", np.random.random(size=(50, 20)), "pipeline"),
        ("obsp", "test_obsp", np.random.random(size=(100, 100)), "output"),
        ("varp", "test_varp", np.random.random(size=(50, 50)), "output"),
        ("uns", "test_uns", 42, "output"),
    ],
)
def test_anndata_manager_set_isset_get_clean(field_str, str_key, value, persist):
    # Tests correct high level API
    features = generate_features(n=500)
    adata = generate_anndata(100, 50, features, force_shape=True)
    adm.set_value(adata, str_key, field_str, value, persist)
    assert adm.isset_value(adata, str_key)
    v_after = adm.get_value(adata, str_key)
    if isinstance(value, np.ndarray):
        np.testing.assert_array_equal(value, v_after)
    else:
        assert value == v_after
    adm.clean(adata, "layer")
    assert adm.isset_value(adata, str_key) != Adm.to_delete(persist, "layer")
    adm.clean(adata, "pipeline")
    assert adm.isset_value(adata, str_key) != Adm.to_delete(persist, "pipeline")
    adm.clean(adata, "output")
    assert adm.isset_value(adata, str_key) is False


def test_anndata_id():
    # Tests correct attribution of IDs
    features = generate_features(n=500)
    adatas = [generate_anndata(100, 50, features) for _ in range(NADATAS)]
    for i, adata in enumerate(adatas):
        assert adm.get_anndata_id(adata) == i
