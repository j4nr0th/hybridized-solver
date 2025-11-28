"""Stub for the C extension module _mod."""

from typing import Self

import numpy as np
from numpy import typing as npt

class BlockInfo:
    """Type describing a block."""

    def __new__(cls, row: int, col: int, val: npt.ArrayLike) -> Self: ...
    @property
    def value(self) -> npt.NDArray[np.double]:
        """The block value."""
        ...
    @property
    def row(self) -> int:
        """The row index of the block."""
        ...
    @property
    def col(self) -> int:
        """The column index of the block."""
        ...
    @property
    def shape(self) -> tuple[int, int]:
        """The shape of the block."""
        ...

class BlockSystem:
    """Block system for hybridized solver."""

    def __new__(cls, *block_sizes: int) -> Self: ...
    def add_block(self, row: int, col: int, val: npt.ArrayLike) -> None:
        """Add a block to the system.

        Parameters
        ----------
        row : int
            Row index of the block.
        col : int
            Column index of the block.
        val : array_like
            Value of the block.
        """
        ...

    def is_valid(self) -> bool:
        """Check if the system has symmetric sparsity and has diagonal blocks."""
        ...

    def as_array(self) -> npt.NDArray[np.double]:
        """Return the system representation as a full matrix."""
        ...

    def get_row_block_indices(self, row: int) -> tuple[int, ...]:
        """Get the column block indices of the specified row.

        Parameters
        ----------
        row : int
            Row index of the block to get the indices for.

        Returns
        -------
        tuple of int
            Indices of columns that appear in the row.
        """
        ...

    def get_block(self, row: int, col: int) -> npt.NDArray[np.double]:
        """Get the block of the matrix."""
        ...
