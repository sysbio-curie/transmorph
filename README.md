# Transmorph (anciently WOTi)

*Warning* This package will soon be released on PyPi. It is unstable 
right now due to structural changes needed to do so.

![](img/logo.png)

This python package aims to provide an easy interface for integrating
datasets using optimal transport (OT)- and Gromov-Wasserstein (GW)-based
methods. We plan to extend the package beyond data integration, with
additional OT-related unsupervised and semi-supervised methods.
*Warning:* This package is still in an early stage. Feel free to
open an issue in case of unexpected behvior.

## Installation

Install from source (latest version, may be unstable)
```sh
git clone https://github.com/Risitop/transmorph
pip install ./transmorph
```

Install from PyPi (stable version)

``` sh
pip install transmorph
```
# Examples

See three example notebooks in `examples/` directory.

# Usage

This package offers four main integration techniques, two based on
OT and two based on GW. Both OT and GW comes in two variants, balanced
(similar to [SCOT](https://github.com/rsinghlab/SCOT "SCOT project")) 
and unbalanced, using a quadratic program
in order to estimate data points weights. These weights are chosen
so that the weighted Gaussian mixture distribution is close to be
uniform over the dataset.

Assuming two numpy arrays *X* and *Y* representing source and target
datasets, WOTi can be used in the following way. First, create a
Woti object. The scale parameter adjusts kernel bandwidth, and needs
some tuning according to cloud sparsity.

``` python
import transmorph as tr

X, Y = ... # datasets, np.ndarray
t = tr.Transmorph(method='ot')
```

Then, simply apply the integration method to project *X* onto *Y*.

``` python
X_integrated = t.fit_transform(X, Y)
```

# Reference

https://www.biorxiv.org/content/10.1101/2021.05.12.443561v1

# Documentation

Work in progress.
