Installing transmorph
=====================

**transmorph** can be installed either from source of from the python repository PyPi. PyPi version is commonly more stable, but may not contain latest features, while you can find the development version on GitHub. Using a python environment is highly recommended (for instance `pipenv`_) in order to easily handle dependencies and versions.

.. contents:: Contents
   :local:
   :backlinks: none
   :depth: 3


Notable dependencies (automatically installed via $pip)
-------------------------------------------------------

+ `anndata`_
+ `igraph`_
+ `leidenalg`_
+ `numba`_
+ `numpy`_
+ `osqp`_
+ `POT`_
+ `pymde`_
+ `pynndescent`_
+ `scipy`_
+ `scikit-learn`_
+ `stabilized-ica`_
+ `umap-learn`_

Install from source (dev version)
------------------------------------

.. code-block:: console

    git clone https://github.com/Risitop/transmorph
    pip install ./transmorph


Install from PyPi (recommended, latest stable version)
------------------------------------------------------

.. code-block:: console

    pip install transmorph


What's next
-----------

Once the installation is completed, you can move on to :ref:`tutorials_section`.

.. _pipenv: https://pypi.org/project/pipenv/
.. _anndata: https://anndata.readthedocs.io/en/latest/
.. _igraph: https://igraph.org/
.. _leidenalg: https://leidenalg.readthedocs.io/en/stable/intro.html
.. _numba: https://numba.pydata.org/
.. _numpy: https://numpy.org/
.. _osqp: https://github.com/osqp/osqp-python
.. _POT: https://github.com/PythonOT/POT
.. _pymde: https://pymde.org/
.. _pynndescent: https://pynndescent.readthedocs.io/en/latest/
.. _scipy: https://www.scipy.org/
.. _scikit-learn: https://scikit-learn.org/stable/
.. _stabilized-ica: https://stabilized-ica.readthedocs.io/en/latest/
.. _umap-learn: https://umap-learn.readthedocs.io/en/latest/
