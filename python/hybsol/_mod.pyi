"""Stub for the C extension module _mod."""

from typing import Literal, Self

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

    def multiply_row(self, row: int, val: npt.ArrayLike, start: int = 0) -> None:
        """Multiply the row by the matrix.

        Parameters
        ----------
        row : int
            Row index of blocks to multiply.

        val : array_like
            Square matrix with which the row should be multiplied.

        start : int, default: 0
            All blocks before this block index are skipped.
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

    def has_block(self, row: int, col: int) -> bool:
        """Check if the block at row ``row`` and column ``col`` is present."""
        ...

    @property
    def n_blocks(self) -> int:
        """Number of blocks."""
        ...

    @property
    def block_sizes(self) -> npt.NDArray[np.uint64]:
        """Array of sizes of blocks."""
        ...

    def no_lower_connections(self) -> npt.NDArray[np.bool]:
        """Return an array indicating entries to the left of the diagonal."""
        ...

    def first_column(self, row: int) -> int:
        """Return the index of the first index in a non-empy row."""
        ...

    def get_next_column_index(self, row: int, col: int) -> int:
        """Get the column index after the current one."""
        ...

    def decompose_diagonal(self, idx: int) -> None:
        """Decomposes the diagonal block using LU decomposition.

        Performs unpivoted LU decomposition on the block ``(idx, idx)``. This
        is done in preparation to a call to ``solve_diagonal``.

        Parameters
        ----------
        idx : int
            Index of the block to decompose.
        """
        ...

    def solve_diagonal(
        self, idx: int, val: npt.ArrayLike, out: npt.NDArray[np.double] | None = None
    ) -> npt.NDArray[np.double]:
        """Use the previously decomposed diagonal to solve the linear system.

        Parameters
        ----------
        idx : int
            Index of the diagonal to use. For this method to make any sense, a call to
            :meth:`BlockSystem.decompose_diagonal` should have been made for the same
            block ``idx``.

        val : array_like
            Value to use as the right side of the matrix.

        out : array, optional
            Array to write the result to. If not given (or none), a new array
            is created.

        Returns
        -------
        array
            Result, which if ``out`` was ``None`` will be in a new array, otherwise
            another reference to ``out`` is returned.
        """
        ...

    def row_apply_decomposition(self, row: int) -> None:
        """Apply the decomposition of the diagonal block to the rest of the same row.

        Parameters
        ----------
        row : int
            Index of the row to perform this on. This row must have had its diagonal
            block decomposed by a call to :meth:`BlockSystem.decompose_diagonal` with
            ``row`` passed to it before.
        """
        ...

    def decompose(self, n_threads: int = 0) -> None:
        """Decompose the block system.

        Parameters
        ----------
        n_threads : int, default: 0
            Number of OpenMP threads to use. When 0 or 1 is used,
            the decomposition is serial.
        """
        ...

    def operations(self) -> tuple[tuple[int, int] | tuple[int]]:
        """Get operations as tuples of one or two ints."""
        ...

    def solve(
        self, val: npt.ArrayLike, out: npt.NDArray[np.double] | None = None
    ) -> npt.NDArray[np.double]:
        """Solve the system for the given right side.

        Parameters
        ----------
        val : array_like
            Right side "forcing" of the system.

        out : array, optional
            Array to which the result is written to. Can be the same as ``val``.
            If left unspecified, a new array is created.

        Returns
        -------
        array
            Array with the result of the solve operation. If ``out`` was not specified,
            this is a new array. Otherwise, a reference to ``out`` is returned.
        """
        ...

    def copy(self) -> BlockSystem:
        """Create a copy of the system."""
        ...

    def reorder_blocks(self, new_order: npt.ArrayLike, n_threads: int = 0) -> None:
        """Reorders blocks of the system to follow the newly specified ordering.

        Parameters
        ----------
        new_order : array_like
            Array of indices specifying where the old values should be moved to.

        n_threads : int, default: 0
            Number of OpenMP threads to use for reordering. Specifiny 0 or 1 means
            that it is done in series.
        """
        ...

    def compute_reordering(
        self,
        strategy: Literal["first", "greedy", "balanced"] = "first",
        max_colors: int = 0,
    ) -> npt.NDArray[np.uint64]:
        """Find ordering of unknowns in the system based on "coloring".

        The idea behind computing reordering the degrees of freedom is to first sort them
        by group index, such that a degree of freedom has no interaction with degree of
        freedom has no shared interactions (non-zero blocks in the system) with any other
        degree of freedom in that group. This is often called "coloring". After all the
        degrees of freedom are sorted into these groups, they are ordered group by group.

        Parameters
        ----------
        strategy : typing.Literal["first", "greedy", "balanced"], default: "first"
            How the grouping is constructed. "first" tries to group as many degrees of
            freedom into the first available group, while "greedy" and "balanced" will
            use the group with the most or the least others in them respectively.

        max_colors: int, default: 0
            Maximum number of colors allowed for the coloring. If ``0`` is specified,
            each unknown cna have its own color. If the coloring can not be completed
            using this number of colors, an exception will be raised.

        Returns
        -------
        array
            Array, which specifies new indices for old degrees of freedom. If the
            old degree of freedom had index ``i``, then its new index will be in
            this array at the same index.
        """
        ...
