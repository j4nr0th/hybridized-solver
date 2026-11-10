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
- ROCm, for the AMD GPU backend. Optional: with ``HYBSOL_ENABLE_HIP``
  off -- which is also what a build falls back to, with a warning, when
  ROCm is not installed -- there is no ``hybsol.hip``, and nothing else
  changes. Building with it needs a ROCm installation, which the build
  looks for under ``/opt/rocm`` unless told otherwise; the kernels are
  compiled ahead of time for the architectures ``rocminfo`` names on the
  build machine, or for a pinned list via
  ``-DHYBSOL_HIP_ARCHITECTURES=gfx90a,gfx1100``. The kernels run on a
  GPU of one of those architectures. :mod:`hybsol.hip` is then
  importable wherever such a device exists, and :func:`hybsol.hip.devices`
  lists them.

.. code-block:: sh

    uv pip install . --config-settings=cmake.define.HYBSOL_ENABLE_OPENCL=OFF

.. _CMake: https://cmake.org
.. _uv: https://docs.astral.sh/uv/
