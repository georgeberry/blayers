blayers documentation
=====================

Bayesian Layers for NumPyro: Modular building blocks for flexible, adaptive probabilistic modeling.

This library provides a collection of layers and inference tools for building Bayesian models on top of Numpyro and Jax.

Quickstart
----------

.. code-block:: python

   from blayers import layers
   layer = layers.AdaptiveLayer(...)
   ...

Resources
---------

- `GitHub <https://github.com/georgeberry/blayers>`_
- `PyPI <https://pypi.org/project/blayers/>`_


Detailed API
------------

.. toctree::
   :maxdepth: 2

   api/layers
   api/infer
   api/links
   api/sampling