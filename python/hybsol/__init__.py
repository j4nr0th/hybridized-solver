"""Hybridized block solver.

The heavy lifting happens in a small, dependency-free C core that is exposed
to Python through the :class:`BlockSystem` extension type.
"""

from hybsol._mod import BlockSystem as BlockSystem

__all__ = ["BlockSystem", "__version__"]

__version__ = "0.0.1a"
