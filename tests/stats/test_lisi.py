import numpy as np

from transmorph.stats.lisi import compute_lisi


def load_simulated_data():
    n_cells = 100
    n_features = 20
    X = np.random.rand(n_cells, n_features)
    labels = np.random.randint(0, 3, size=(n_cells,))
    return X, labels


def test_lisi():
    X, labels = load_simulated_data()
    lisi = compute_lisi(X, labels, 30)
    assert lisi.shape == (X.shape[0],)


if __name__ == "__main__":
    test_lisi()
