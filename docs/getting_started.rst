===============================
Getting started with transmorph
===============================

.. highlight:: python

You have now successfully installed transmorph. Here is
a small tutorial to ensure everything works properly on
your machine. We will use a small benchmark dataset,
containing two linear cloud points embedded in a
3D space, in two spiraling shapes. Let us first load
these datasets, represented as `numpy` arrays.
::

   import transmorph as tr
   from transmorph.datasets import load_spirals
   query, target = load_spirals()

Once this is done, we need to create the `Transmorph`
object containing the integration method. This object
can take a lot of parameters (see the API), but we will
simply let the defaults on this example. Then, we apply
the method `Transmorph.fit_transform()` onto our two
datasets in order to get our integration result.
::
   
   integration = tr.Transmorph()
   integrated = integration.fit_transform(query, target)

The variable `integrated` now contains a `numpy` array
representing an integrated version of the `query` dataset
onto the `target`.
If everything went well until there, chances are you are done
with this tutorial. Feel free to apply visualization methods
on `query`, `target` and `integrated` datasets to visualize
the result. For a more in-depth features highlight, you can
move to our more comprehensive tutorials.
