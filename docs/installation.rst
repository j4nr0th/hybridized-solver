Installation
============

From source
-----------

``hybsol`` builds a small C core and the Python bindings around it, so a C
compiler, CMake_ 3.29 or newer and NumPy are needed. With `uv`_ the whole
thing is a single command::

    uv pip install .

This builds the extension in place and installs the ``hybsol`` package.

Development install
-------------------

For a development setup use an editable install; the ``dev`` dependency group
pulls in pytest, ruff and the other development tools::

    uv sync
    uv pip install -e .

Dependencies
------------

- Python 3.11 or newer
- NumPy 2.0 or newer
- SciPy (only used by the examples and the benchmarks)
- OpenMP, when available. Without it the solver falls back to a serial
  implementation; pass ``-DHYBSOL_ENABLE_OPENMP=OFF`` to build without it
  explicitly.

.. _CMake: https://cmake.org
.. _uv: https://docs.astral.sh/uv/
