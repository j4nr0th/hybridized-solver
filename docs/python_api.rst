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
