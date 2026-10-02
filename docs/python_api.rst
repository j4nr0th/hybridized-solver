Python API
==========

.. currentmodule:: hybsol

Three types cover the life cycle of a block system.
:class:`BlockSystem` owns the assembly: the blocks, their values, and the block
order. :class:`Decomposition` is what a factorization produces — a copy of the
blocks it needs, which it solves any number of times. :class:`Elimination` is
the symbolic walk in front of both, which reports what factorizing would cost
without touching a value. The type of the values stored is chosen at
construction from :class:`Precision`.

The pipeline runs in that order: :meth:`BlockSystem.elimination` stops after
the walk, :meth:`BlockSystem.decompose` runs all four stages and hands back the
decomposition, and :func:`factorize` returns the walk and the decomposition
together so both stay available.

.. autoclass:: BlockSystem
   :members:
   :special-members: __new__
   :undoc-members:

.. autoclass:: Precision
   :members:
   :undoc-members:

.. autoclass:: Decomposition
   :members:
   :undoc-members:

.. autoclass:: Elimination
   :members:
   :undoc-members:

.. autofunction:: factorize

.. autofunction:: refined_solve

The OpenCL backend
------------------

GPU factorization lives in its own module, ``hybsol.opencl``, which is
imported when it is wanted and never otherwise. Importing :mod:`hybsol` alone
loads no OpenCL; an installation built without the backend has no
:mod:`hybsol.opencl` at all, and the import says so.

The walk and the assembly stay on the CPU; the factorization, the solve and
the replay run as device kernels, and the factors never come back to the
host. What comes back is an ordinary :class:`Decomposition`, used exactly as
one from :meth:`BlockSystem.decompose` -- :func:`refined_solve` included --
with ``device`` telling it apart.

.. autofunction:: hybsol.opencl.devices

.. autofunction:: hybsol.opencl.device_info

.. autofunction:: hybsol.opencl.decompose

.. autoclass:: DeviceError
   :members:
   :undoc-members:
