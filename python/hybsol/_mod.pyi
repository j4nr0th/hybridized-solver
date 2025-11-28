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
        """Get the block of the system.

        Parameters
        ----------
        row : int
            Row index of the block to get.
        col : int
            Column index of the block to get.

        Returns
        -------
        array
            New array with the same value as the block.
        """
        ...

    def multiply_row(self, row: int, val: npt.ArrayLike) -> None:
        """Multiply the row by the matrix.

        Parameters
        ----------
        row : int
            Row index of blocks to multiply.

        val : array_like
            Square matrix with which the row should be multiplied.
        """
        ...

    def get_block_size(self, row: int, col: int) -> tuple[int, int]:
        """Get the size of a system block."""
        ...

    def eliminate_row(self, row_src: int, row_tgt: int, val: npt.ArrayLike) -> None:
        r"""Eliminate a target row using a source row, multiplied by matrix.

        This performs the row elimination operation from the source row on the target
        row. If the source row is represented by :math:`\mathbf{M}_s`, the target row
        by :math:`\mathbf{M}_t`, and the scaling matrix as :math:`\mathbf{S}`, the
        new vale of the target row will be:

        ..math ::
            \mathbf{M}_t^\prime = \mathbf{M}_t - \mathbf{S} \mathbf{M}_s

        This operation will ignore all entres in both rows with column index lower or
        equal to ``row_src``, since those are assumed to have already been eliminated.

        Parameters
        ----------
        row_src : int
            Index of the source row.

        row_tgt : int
            Index of the row to eliminate.

        val : array_like
            Matrix used to scale the target row.
        """
        ...
