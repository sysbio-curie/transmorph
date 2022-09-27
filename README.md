# [<img alt="Transmorph" src="img/logo_v2.png" height="90">](https://transmorph.readthedocs.io/en/latest/index.html)

[![PyPI version](https://badge.fury.io/py/transmorph.svg)](https://badge.fury.io/py/transmorph)
[![GitHub license](https://img.shields.io/github/license/Risitop/transmorph.svg)](https://github.com/Risitop/transmorph/blob/main/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/transmorph/badge/?version=latest)](https://transmorph.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://pepy.tech/badge/transmorph)](https://pepy.tech/project/transmorph)
[![Downloads](https://pepy.tech/badge/transmorph/month)](https://pepy.tech/project/transmorph)

**transmorph** is a python framework dedicated to data integration, with a focus on single-cell applications. Dataset integration describes the problem of embedding two or more datasets together, across different batches or feature spaces, so that similar samples end up close from one another. In transmorph we aim to provide a comprehensive framework to design, apply, report and benchmark data integration models using a system of interactive building blocks supported by statistical and plotting tools. We included pre-built models as well as benchmarking databanks in order to easily set up integration tasks. This package can be used in compatibility with **scanpy** and **anndata** packages, and works in jupyter notebooks.

Transmorph is also computationally efficient, and can scale to large datasets with competitive integration quality. 

## Documentation

https://transmorph.readthedocs.io/en/latest/

## Installation

**transmorph** can be installed either from source of from the python repository PyPi. PyPi version is commonly more stable, but may not contain latest features, while you can find the development version on GitHub. Using a python environment is highly recommended (for instance  [pipenv](https://pypi.org/project/pipenv/)) in order to easily handle dependencies and versions. **transmorph** has only be tested for python 3.9.

### Notable dependencies (automatically installed via $pip)

+ [anndata](https://anndata.readthedocs.io/en/latest/)
+ [igraph](https://igraph.org/)
+ [leidenalg](https://leidenalg.readthedocs.io/en/stable/intro.html)
+ [numba](https://numba.pydata.org/)
+ [numpy](https://numpy.org/) 
+ [osqp](https://github.com/osqp/osqp-python) (quadratic program solver)
+ [POT](https://github.com/PythonOT/POT) (optimal transport in python)
+ [pymde](https://pymde.org/)
+ [pynndescent](https://pynndescent.readthedocs.io/en/latest/)
+ [scipy](https://www.scipy.org/) 
+ [scikit-learn](https://scikit-learn.org/stable/)
+ [stabilized-ica](https://stabilized-ica.readthedocs.io/en/latest/)
+ [umap-learn](https://umap-learn.readthedocs.io/en/latest/)

### Install from source (latest version)

```sh
git clone https://github.com/Risitop/transmorph
pip install ./transmorph
```

### Install from PyPi (recommended, latest stable version)

```sh
pip install transmorph
```

### Quick starting with a pre-built model

All **transmorph** models take a list of AnnData objects as input
for data integration. Let us start by loading some benchmarking 
data, gathered from [Chen 2020] (3.4GB size).

```python
from transmorph.datasets import load_chen_10x
chen_10x = load_chen_10x()
```

One can then either create a custom integration model, or load 
a pre-built transmorph model. We will choose the *EmbedMNN* model with
default parameters for this example, which embeds all datasets into 
a common abstract 2D space. 

```python
from transmorph.models import EmbedMNN
model = EmbedMNN()
model.fit(chen_10x)
```

Integration embedding coordinates can be gathered in each AnnData object,
in AnnData.obsm['transmorph'].

```python
chen_10x['P01'].obsm['transmorph']
```

[Chen 2020] [Chen, Y. P., Yin, J. H., Li, W. F., Li, H. J., Chen, D. P., Zhang, C. J., ... & Ma, J. (2020). Single-cell transcriptomics reveals regulators underlying immune cell diversity and immune subtypes associated with prognosis in nasopharyngeal carcinoma. Cell research, 30(11), 1024-1042.](https://www.nature.com/articles/s41422-020-0374-x)
