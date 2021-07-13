# <img alt="Transmorph" src="img/logo.png" height="90">

[![PyPI version](https://badge.fury.io/py/transmorph.svg)](https://badge.fury.io/py/transmorph)

**Transmorph** is a python package dedicated to transportation theory-based
data analysis, with a particular focus on data integration, which 
describes the problem of tying several datasets together. Our priority
is scalability, and the current version can handle datasets of **tens of
thousands data points** in **under a minute**, thanks to various 
optimization and approximation tricks. 

***Warning:*** This package is still in an early stage of its
development. Please open an issue in case of unexpected behavior.

## Changelog (v0.0.8)

+ Structure: Improved overall code documentation
+ Structure: Added numba support
+ Structure: Unit tests
+ Fix: Various fixes for corner cases
+ Fix: Various fixes and asserts for more robust code
+ New feature: geodesic distance (Gromov-Wasserstein only)
+ New feature: lower dimensional representation
+ New feature: approximation using vertex covering
+ New feature: unbalanced case (optimal transport only)
+ New feature: label weighting (supervised case)
+ New feature: label-dependent cost function

## Installation

Two main options exist to install transmorph, from source of from PyPi. 
PyPi version should be more stable, but may not contain latest features.
Using a python environment is highly recommended (for instance 
[pipenv](https://pypi.org/project/pipenv/)) in order to better handle
dependency versioning.

### Dependencies (automatically installed via $pip)

+ [cython](https://cython.org/)
+ [numba](https://numba.pydata.org/)
+ [numpy](https://numpy.org/) 
+ [scipy](https://www.scipy.org/) 
+ [scikit-learn](https://scikit-learn.org/stable/)
+ [osqp](https://github.com/osqp/osqp-python) (quadratic program solver)
+ [POT](https://github.com/PythonOT/POT) (optimal transport in python)

### Install from source (latest version)

```sh
git clone https://github.com/Risitop/transmorph
pip install ./transmorph
```

### Install from PyPi (recommended, latest stable version)

``` sh
pip install transmorph
```

## Usage

### Model fitting

We choose to adopt a philosophy similar to `sklearn`'s package, 
with numerical methods encapsulated in a python object. The main
class here is the `Transmorph`, and should be fitted prior to any 
analysis. First, you need to create a Transmorph object, and choose 
its parameters. The most important parameters to monitor are the 
following. Of course, default values are available.

+ method: `'ot'` or `'gromov'`. Optimal transport or Gromov-Wasserstein
approach. As a rule of thumbs, if both datasets are in the same feature space,
optimal transport is probably the better choice. Otherwise, or in case of severe
rotation Gromov-Wasserstein is to privilege.
+ geodesic: True or False. **Only available for Gromov-Wasserstein method.** Use
distance along the edges of a graph instead of vector-based distance in order to
build cost matrices. This better approximates distance along the underlying 
manifold, but is prone to hubness effects.
+ entropy: True or False. Wether to use entropy regularization described in 
(Cuturi 2013). It costs one extra parameter and an approximate result, but
can really speedup the resolution for large problems.
+ unbalanced: True or False. **Only available for Optimal Transport method.** Wether
to adopt the unbalanced formulation, in the case of severe unbalance in data 
types among datasets. If data types are known, see `weighting\_strategy` instead.
+ weighting\_strategy: `'uniform'`, `'woti'` or `'labels'`. Strategy to select
point weights before integration. Uniform equally weights each point, this is the 
standard approach which is prone to overfitting in real-life dataset. WOTi approach
is described in (Fouché, bioRxiv 2021), and aims to correct weighting with respect
to local density. Labels-based approach works the fastest and the best, but point 
labels are to be known prior to integration.
+ n\_hops: n > 0. This approach is not described in the literature yet to our
knowledge. It uses a vertex cover to approximate the datasets. Increase this number
to speedup algorithms, at precision cost. In our testing, it starts being relevant
for datasets above 10,000 data points.

``` python
import transmorph as tr

t = tr.Transmorph(
    method = 'ot', # 'ot' or 'gromov'
    metric = 'sqeuclidean', # any scipy metric, e.g. 'euclidean', 'cosine'...
    geodesic = False, # kNN-graph distance instead of vector based, gromov-only
    normalize = True, # column-normalize data matrices
    n_comps = 5, # use a 5-PC pca. Set it to -1 (default) to avoid this step
    entropy = False, # use Sinkhorn-Knopp procedure (see Cuturi 2013)
    hreg = 1e-3, # regularization parameter for entropy term, if entropy = True
    unbalanced = False, # use unbalanced formulation
    mreg = 1e-3, # unbalanced terms regularizers, if unbalanced = True
    weighting_strategy = 'uniform', # in 'uniform', 'woti', 'labels'
    label_dependency = 0, # penalty for supervised formulation
    n_hops = 0, # increase it for large datasets
    max_iter = 1e6 # maximum number of iterations for OT solvers
)
```

You can then load your two datasets and fit the Transmorph.

``` python
X, Y = ... # datasets, np.ndarrays
t.fit(
    X, # source matrix 
    Y, # reference matrix
    xs_labels = None, # source points labels (not mandatory)
    yt_labels = None, # reference points labels (not mandatory)
    xs_weights = None, # source points custom weights (inferred if None)
    yt_weights = None, # reference points custom weights (inferred if None)
    Mx = None, # If method='gromov', custom cost matrix for X (inferred if None)
    My = None, # If method='gromov', custom cost matrix for Y (inferred if None)
    Mxy = None, # If method='ot, custom cost matrix for X -> Y (inferred if None)
)
```

### Data integration

Once the Transmorph is fitted, data integration is straightforward through
the `transform` method, inspired by the
([Ferradans 2013](https://hal.archives-ouvertes.fr/hal-00797078/document))
methodology. 

``` python
X_integrated = t.transform(
    jitter = True, # Adds a little jittering to the result
    jitter_std = .01 # jittering strength
)
```

## Examples

See three example notebooks in `examples/` directory (might not always be up to date,
but we do our best to do so).

## Contributors

+ Aziz Fouché, (Institut Curie Paris, ENS Paris-Saclay)
+ Andrei Zinovyev, (Institut Curie Paris, INSERM, Mines ParisTech)
+ Special thanks to Nicolas Captier (Institut Curie Paris) and Maxime Roméas (LIX, École polytechnique, INRIA, Institut Polytechnique de Paris) for the insightful discussions and proofreading.

## Reference

https://www.biorxiv.org/content/10.1101/2021.05.12.443561v1

## Documentation

Work in progress.
