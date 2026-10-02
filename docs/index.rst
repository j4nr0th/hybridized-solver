hybsol
======

A solver for hybridized block systems.

A *block system* is a square matrix that is built out of small dense blocks,
only some of which are stored. :class:`hybsol.BlockSystem` assembles such a
system, factorizes it with a block LU decomposition and replays the recorded
factorization against any number of right-hand sides.

The numerically interesting part lives in a small, dependency-free C core; the
Python package is a thin set of bindings around it.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   quickstart
   python_api
   c_api

.. toctree::
   :maxdepth: 1
   :caption: Examples

   auto_examples/index

Indices and tables
------------------

- :ref:`genindex`
- :ref:`search`
