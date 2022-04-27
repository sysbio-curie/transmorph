#!/usr/bin/env python3

from transmorph.datasets import load_travaglini_10x
from transmorph.engine.transforming import ICA

N_ICS = 20


def test_transform_ica():
    # We cannot test against a reference due to stochasticity
    # of sICA solver :(
    datasets = list(load_travaglini_10x().values())
    transform = ICA(n_components=N_ICS)
    transform.retrieve_common_features(datasets, True)
    _ = transform.transform([adata.X for adata in datasets])


if __name__ == "__main__":
    test_transform_ica()
