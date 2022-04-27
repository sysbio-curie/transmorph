#!/usr/bin/env python3

from transmorph.datasets import load_travaglini_10x
from transmorph.utils.dimred import ica


def test_ica():
    X = list(load_travaglini_10x().values())[0].X
    X_ica = ica(X, n_components=30)
    assert X_ica.shape == (X.shape[0], 30)


if __name__ == "__main__":
    test_ica()
