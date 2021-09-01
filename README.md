# <img alt="Transmorph" src="img/logo.png" height="90">

[![PyPI version](https://badge.fury.io/py/transmorph.svg)](https://badge.fury.io/py/transmorph)
[![GitHub license](https://img.shields.io/github/license/Risitop/transmorph.svg)](https://github.com/Risitop/transmorph/blob/main/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/transmorph/badge/?version=latest)](https://transmorph.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://pepy.tech/badge/transmorph)](https://pepy.tech/project/transmorph)
[![Downloads](https://pepy.tech/badge/transmorph/month)](https://pepy.tech/project/transmorph)

**Transmorph** is a python package dedicated to data integration, with a
particular focus on single-cell applications. Dataset integration describes
the problem of tying several datasets together, across different samples or
modalities. Our priority is scalability, as we aim to integrate datasets of
**tens of thousands data points in under a minute**. We use an efficient
optimal transport-based approach to do so, supported by a variety of
computational tricks.


## Documentation

https://transmorph.readthedocs.io/en/latest/

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

## Contributors

+ Aziz Fouché, (Institut Curie Paris, ENS Paris-Saclay)
+ Andrei Zinovyev, (Institut Curie Paris, INSERM, Mines ParisTech)
+ Special thanks to Nicolas Captier (Institut Curie Paris) and Maxime Roméas (LIX, École polytechnique, INRIA, Institut Polytechnique de Paris) for the insightful discussions and proofreading.

## References 

+ Weighted optimal transport integration: https://www.biorxiv.org/content/10.1101/2021.05.12.443561v1
+ Package: preprint to come

