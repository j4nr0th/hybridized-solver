"""Hybridized block solver.

The heavy lifting happens in a small, dependency-free C core, exposed to
Python through three extension types: :class:`BlockSystem` for the assembled
system, :class:`Decomposition` for a factorized one, and :class:`Elimination`
for the symbolic walk that decides what factorizing would cost.
"""

from enum import StrEnum

from hybsol._mod import BlockSystem as BlockSystem
from hybsol._mod import Decomposition as Decomposition
from hybsol._mod import Elimination as Elimination
from hybsol._mod import SingularSystemError as SingularSystemError
from hybsol.pipeline import factorize as factorize
from hybsol.pipeline import refined_solve as refined_solve

__all__ = [
    "BlockSystem",
    "Decomposition",
    "Elimination",
    "Precision",
    "SingularSystemError",
    "__version__",
    "factorize",
    "refined_solve",
]

__version__ = "0.0.1a"


class Precision(StrEnum):
    """The floating-point type a :class:`BlockSystem` stores its blocks in.

    Precision is chosen when a system is created and cannot be changed: it
    decides the type of the blocks, of the factors the decomposition produces
    and of every array :meth:`BlockSystem.block_storage` and
    :meth:`BlockSystem.get_block` hand back.

    Because this is a :class:`enum.StrEnum`, a plain string works wherever a
    member is expected — ``BlockSystem(2, 2, precision="single")`` and
    ``BlockSystem(2, 2, precision=Precision.SINGLE)`` mean the same thing.
    """

    DOUBLE = "double"
    """IEEE-754 binary64. The default."""

    SINGLE = "single"
    """IEEE-754 binary32: half the memory, and a solve good to about
    ``cond * 1e-7`` rather than ``cond * eps``. Recover the difference with
    an iterative refinement step in double precision.
    """
