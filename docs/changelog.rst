=========
Changelog
=========

v0.1.2
------

    + Dev: Switched to log-stabilized versions of Sinkhorn's algorithms
    + Fix: transform() throws an error if the result includes NaN
    + Fix: Mixing up strings and constants
    + Fix: fit_transform now properly passes weights parameters to fit
    + New: Time profiling of integration
    + New: scanpy/anndata interfacing via transmorph.anndata.integrate_anndata

v0.1.1
------

    + Dev: Replaced strings by f-strings
    + Dev: Improved logs and error messages.
    + Doc: Improved code documentation
    + Fix: Removing alias strategies for clarity
    + Fix: A negative number of PCs is no longer accepted
    + Fix: Checking metric is scipy-compatible
    + Fix: Checking entropy regularizer positive
    + Fix: Checking marginal penalty positive
    + Fix: String type consistency
    + Fix: Improved string representation
    + Fix: Gromov compatible with label weighting
    + Fix: Gromov incompatible with label dependency
    + Fix: General label-dependent parameters consistency
    + Fix: Adapted unit tests to these parameters
    + Fix: check_array caused an error when performed on a 1D array

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
