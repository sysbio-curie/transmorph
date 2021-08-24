.. transmorph documentation master file, created by
   sphinx-quickstart on Mon Aug 23 10:36:26 2021.

=================================
Data integration using transmorph
=================================

**Transmorph** is a python package dedicated to data integration, with a
particular focus on single-cell applications. Dataset integration describes the
problem of tying several datasets together, across different samples or
modalities. Our priority is **scalability**, as we aim to integrate datasets of
tens of thousands data points in under a minute. We use an efficient optimal
transport-based approach to do so, supported by a variety of computational
tricks.


.. toctree::
   :maxdepth: 1

   install
   getting_started
   tutorials
   api


Contribute to transmorph
========================

For any issue with transmorph, or if you want to see a feature implemented,
do not hesitate to open an issue on this project's
`GitHub <https://github.com/Risitop/transmorph>`_. We are trying to
gather user experience on various datasets in order to improve our package robustness!


Changelog
=========

v0.0.8
------

   * Structure: Improved overall code documentation
   * Structure: Added numba support
   * Structure: Unit tests
   * Fix: Various fixes for corner cases
   * Fix: Various fixes and asserts for more robust code
   * New feature: geodesic distance (Gromov-Wasserstein only)
   * New feature: lower dimensional representation
   * New feature: approximation using vertex covering
   * New feature: unbalanced case (optimal transport only)
   * New feature: label weighting (supervised case)
   * New feature: label-dependent cost function

