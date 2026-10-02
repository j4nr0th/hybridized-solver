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
- OpenCL, for the GPU backend. Optional: with ``HYBSOL_ENABLE_OPENCL`` off --
  which is also what a build falls back to when the OpenCL loader or headers
  are missing -- there is no ``hybsol.opencl``, and nothing else changes.
  Building with it needs the OpenCL headers and an ICD loader
  (``pacman -S opencl-headers ocl-icd`` on Arch, ``apt install
  opencl-headers ocl-icd-opencl-dev`` on Debian), and running it needs a
  vendor ICD -- the one that ships with the GPU driver. :mod:`hybsol.opencl`
  is then importable wherever such a device exists, and
  :func:`hybsol.opencl.devices` lists them.

.. code-block:: sh

    uv pip install . --config-settings=cmake.define.HYBSOL_ENABLE_OPENCL=OFF

.. _CMake: https://cmake.org
.. _uv: https://docs.astral.sh/uv/
