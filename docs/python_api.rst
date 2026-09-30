Python API
==========

.. currentmodule:: hybsol

The package exposes a single type, :class:`BlockSystem`, which owns the whole
life cycle of a block system: assembly, decomposition and solving. The type of
the values it stores is chosen at construction from :class:`Precision`.

.. autoclass:: BlockSystem
   :members:
   :special-members: __new__
   :undoc-members:

.. autoclass:: Precision
   :members:
   :undoc-members:
