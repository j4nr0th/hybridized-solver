"""Stub for the C extension module _mod."""

from collections.abc import Sequence
from enum import StrEnum
from typing import Literal, Self

import numpy as np
from numpy import typing as npt

OrderingStrategy = Literal["first", "greedy", "balanced"]

class Precision(StrEnum):
    """The floating-point type a :class:`BlockSystem` stores its blocks in.

    Re-exported from :mod:`hybsol`; a plain string of the same value is
    accepted anywhere a member is.
    """

    DOUBLE = "double"
    SINGLE = "single"

class BlockSystem:
    """Block system for hybridized solver.

    The system is an ``n x n`` arrangement of blocks, where block ``(i, j)`` has
    shape ``(block_sizes[i], block_sizes[j])``. Only a subset of the blocks needs
    to be present; :meth:`is_valid` reports whether the structure is one the
    solver can decompose.
    """

    def __new__(
        cls, *block_sizes: int, precision: Precision | str = Precision.DOUBLE
    ) -> Self: ...
    @classmethod
    def from_blocks(
        cls,
        block_sizes: Sequence[int],
        rows: npt.ArrayLike,
        cols: npt.ArrayLike,
        data: npt.ArrayLike,
        precision: Precision | str = Precision.DOUBLE,
    ) -> Self:
        """Build a whole system from a flat COO description in one pass.

        Parameters
        ----------
        block_sizes : sequence of int
            Size of every block on the diagonal.
        rows : array_like
            Row index of every block.
        cols : array_like
            Column index of every block.
        data : array_like
            Concatenated, row-major block values; block ``k`` occupies
            ``block_sizes[rows[k]] * block_sizes[cols[k]]`` entries.

        Returns
        -------
        BlockSystem
            A new system holding exactly the given blocks.
        """
        ...

    @classmethod
    def from_block_list(
        cls,
        blocks: Sequence[tuple[int, int, npt.ArrayLike]],
        block_sizes: Sequence[int] | None = None,
        precision: Precision | str = Precision.DOUBLE,
    ) -> Self:
        """Build a system from a list of ``(row, col, array)`` triples.

        Parameters
        ----------
        blocks : sequence of tuple
            Triples ``(row, col, value)``, where ``value`` has shape
            ``(block_sizes[row], block_sizes[col])``.
        block_sizes : sequence of int, optional
            Size of every block on the diagonal. When omitted, the sizes are
            inferred from the shapes of the given blocks; that is only possible
            if every block index appears in at least one triple.

        Returns
        -------
        BlockSystem
            A new system holding exactly the given blocks.
        """
        ...

    def add_block(self, row: int, col: int, val: npt.ArrayLike) -> None:
        """Add a block to the system.

        If the block is already present the values are summed into it.

        Parameters
        ----------
        row : int
            Row index of the block.
        col : int
            Column index of the block.
        val : array_like
            Value of the block, with shape ``(block_sizes[row], block_sizes[col])``.
        """
        ...

    def add_blocks(
        self, rows: npt.ArrayLike, cols: npt.ArrayLike, data: npt.ArrayLike
    ) -> None:
        """Add many blocks in a single pass.

        This is the fast assembly path: the index arrays are processed once and
        each row allocates its storage a single time. Duplicate ``(row, col)``
        pairs accumulate, exactly as repeated :meth:`add_block` calls would.

        Block ``k`` occupies ``block_sizes[rows[k]] * block_sizes[cols[k]]``
        consecutive entries of ``data``, in the order implied by ``rows`` and
        ``cols``.

        Parameters
        ----------
        rows : array_like
            Row index of every block being added.
        cols : array_like
            Column index of every block being added.
        data : array_like
            Concatenated, row-major block values.
        """
        ...

    def reserve(self, row: int, capacity: int) -> None:
        """Make sure the row can hold ``capacity`` blocks without reallocating.

        Parameters
        ----------
        row : int
            Row index to reserve space for.
        capacity : int
            Number of blocks the row should be able to hold.
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
            Indices of columns that appear in the row, in increasing order.
        """
        ...

    def get_block(self, row: int, col: int) -> npt.NDArray[np.double]:
        """Get a copy of the value of the specified block.

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

    def block_storage(self, row: int, col: int) -> npt.NDArray[np.double]:
        """Get writable storage for a block, creating it on first use.

        The shape of the block is implied by the system, so this hands back
        exactly the right buffer to fill in place instead of building a
        temporary and passing it to :meth:`add_block`. The first call adds the
        block to the sparsity pattern and zeroes it; later calls return the
        same storage unchanged, so nothing written so far is lost.

        The returned array is a view backed by the system, which it keeps
        alive. Operations that only rewrite values in place are visible
        through it, as with any view, but :meth:`eliminate_row`,
        :meth:`reorder_blocks` and :meth:`decompose` refuse to run while any
        such array is alive.

        Parameters
        ----------
        row : int
            Row index of the block.
        col : int
            Column index of the block.

        Returns
        -------
        array
            Writable, C-contiguous view of the block, shape
            ``(block_sizes[row], block_sizes[col])``.

        Raises
        ------
        ValueError
            An index is outside ``[0, n_blocks)``.
        RuntimeError
            The system has already been decomposed, or a previously returned
            view is still alive in front of a method that would move it.
        """
        ...

    def get_block_size(self, row: int, col: int) -> tuple[int, int]:
        """Get the shape ``(rows, cols)`` of a system block.

        Parameters
        ----------
        row : int
            Row index of the block.
        col : int
            Column index of the block.
        """
        ...

    def has_block(self, row: int, col: int) -> bool:
        """Check if the block at row ``row`` and column ``col`` is present."""
        ...

    @property
    def precision(self) -> Precision:
        """The type this system stores its blocks in."""
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
        """Return an array flagging rows with nothing to the left of their diagonal."""
        ...

    def first_column(self, row: int) -> int:
        """Return the index of the first index in a non-empty row.

        Parameters
        ----------
        row : int
            Row index to inspect.
        """
        ...

    def get_next_column_index(self, row: int, col: int) -> int:
        """Get the column index after the current one.

        Parameters
        ----------
        row : int
            Row index to inspect.
        col : int
            Column index to start looking after.
        """
        ...

    def multiply_row(self, row: int, val: npt.ArrayLike, start: int = 0) -> None:
        """Multiply the row by the matrix.

        Every stored block in the row at or after column ``start`` is replaced
        by ``val @ block``; blocks before ``start`` are left alone.

        Parameters
        ----------
        row : int
            Row index of blocks to multiply.
        val : array_like
            Square matrix with which the row should be multiplied.
        start : int, default: 0
            First block column (inclusive) to transform.
        """
        ...

    def eliminate_row(self, row_src: int, row_tgt: int, val: npt.ArrayLike) -> None:
        r"""Eliminate a target row using a source row, multiplied by matrix.

        This performs the row elimination operation from the source row on the target
        row. If the source row is represented by :math:`\mathbf{M}_s`, the target row
        by :math:`\mathbf{M}_t`, and the scaling matrix as :math:`\mathbf{S}`, the
        new value of the target row will be:

        .. math ::
            \mathbf{M}_t^\prime = \mathbf{M}_t - \mathbf{S} \mathbf{M}_s

        This operation will ignore all entries in both rows with column index lower or
        equal to ``row_src``, since those are assumed to have already been eliminated.

        Parameters
        ----------
        row_src : int
            Index of the source row.
        row_tgt : int
            Index of the row to eliminate.
        val : array_like
            Matrix used to scale the source row.
        """
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
            :meth:`decompose_diagonal` should have been made for the same block ``idx``.
        val : array_like
            Value to use as the right side of the matrix.
        out : array, optional
            Array to write the result to. If not given (or ``None``), a new array
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
            block decomposed by a call to :meth:`decompose_diagonal` with ``row``
            passed to it before.
        """
        ...

    def decompose(
        self,
        n_threads: int = 0,
        workspace: npt.NDArray[np.uint8] | None = None,
    ) -> None:
        """Decompose the block system.

        The system must be valid (see :meth:`is_valid`). After a successful
        decomposition the system is frozen: no further blocks may be added,
        eliminated or reordered.

        Parameters
        ----------
        n_threads : int, default: 0
            Number of OpenMP threads to use. ``0`` selects the OpenMP default
            (usually every core) and ``1`` runs the decomposition serially.
        workspace : numpy.typing.NDArray[numpy.uint8], optional
            A writable 1-D array of at least :meth:`workspace_bytes` bytes.
            Passing one keeps the scratch out of the library's allocator, so a
            buffer can be reused across systems. Its contents are overwritten.
            Omit it and the scratch is allocated internally.
        """
        ...

    def workspace_bytes(self, n_threads: int = 0) -> int:
        """Bytes of scratch :meth:`decompose` needs at this thread count.

        Sizes the ``workspace`` array :meth:`decompose` accepts, so one buffer
        can be allocated once and reused. It depends only on the block sizes,
        the precision and the thread count, so it may be asked for before any
        blocks are added.

        Parameters
        ----------
        n_threads : int, default: 0
            The thread count that will be passed to :meth:`decompose`.
        """
        ...

    def operations(self) -> tuple[tuple[int, ...], ...]:
        """Get the recorded operations as tuples of one or two ints.

        A one-element tuple ``(idx_row,)`` solves with the LU factors of the
        diagonal block; a two-element tuple ``(idx_row, idx_col)`` eliminates
        block ``(idx_row, idx_col)`` using block row ``idx_col``.
        """
        ...

    def solve(
        self, val: npt.ArrayLike, out: npt.NDArray[np.double] | None = None
    ) -> npt.NDArray[np.double]:
        """Solve the system for the given right side.

        Parameters
        ----------
        val : array_like
            Right-hand side of length ``sum(block_sizes)``.
        out : array, optional
            Array to write the solution to. If not given (or ``None``), a new array
            is created. It may be the same array as ``val``, in which case the
            solution overwrites the right-hand side in place.

        Returns
        -------
        array
            Solution of the linear system.
        """
        ...

    def copy(self) -> BlockSystem:
        """Create a deep copy of the system, including its decomposition."""
        ...

    def reorder_blocks(self, new_order: npt.ArrayLike, n_threads: int = 0) -> None:
        """Reorders blocks of the system to follow the newly specified ordering.

        Parameters
        ----------
        new_order : array_like
            Array of indices specifying where the old values should be moved to.
        n_threads : int, default: 0
            Number of OpenMP threads to use for reordering. ``0`` selects the
            OpenMP default and ``1`` reorders serially.
        """
        ...

    def compute_reordering(
        self,
        strategy: OrderingStrategy = "first",
        max_colors: int = 0,
    ) -> npt.NDArray[np.uint64]:
        """Find ordering of unknowns in the system based on "coloring".

        The idea behind computing the reordering of the degrees of freedom is to first
        sort them by group index, such that a degree of freedom shares no non-zero
        block with any other degree of freedom in that group. This is often called
        "coloring". After all the degrees of freedom are sorted into these groups,
        they are ordered group by group.

        Parameters
        ----------
        strategy : typing.Literal["first", "greedy", "balanced"], default: "first"
            How the grouping is constructed. "first" tries to group as many degrees of
            freedom into the first available group, while "greedy" and "balanced" will
            use the group with the most or the least others in them respectively.
        max_colors : int, default: 0
            Maximum number of colors allowed for the coloring. If ``0`` is specified,
            each unknown may have its own color. If the coloring can not be completed
            using this number of colors, an exception will be raised.

        Returns
        -------
        array
            Array which specifies new indices for old degrees of freedom. If the
            old degree of freedom had index ``i``, then its new index will be in
            this array at the same index.
        """
        ...

    def reorder_vector(
        self,
        new_order: npt.ArrayLike,
        vector: npt.ArrayLike,
        out: npt.NDArray[np.double] | None = None,
    ) -> npt.NDArray[np.double]:
        """Reorder the vector based on new block ordering.

        The block that was at slot ``i`` is moved to slot ``new_order[i]``, so the
        result is the vector of the system as :meth:`reorder_blocks` left it. The
        method is meant to be called on the reordered system.

        Parameters
        ----------
        new_order : array_like
            New order of blocks, as returned by :meth:`compute_reordering`.
        vector : array_like
            Vector in the *old* ordering.
        out : array, optional
            Output array to receive the reordered vector contents. Must not be the same
            array as ``vector``.

        Returns
        -------
        array
            Reordered contents of ``vector``. If ``out`` was specified, this is just a
            reference to it, otherwise a new array is created.
        """
        ...

    def unorder_vector(
        self,
        new_order: npt.ArrayLike,
        vector: npt.ArrayLike,
        out: npt.NDArray[np.double] | None = None,
    ) -> npt.NDArray[np.double]:
        """Undo reordering of the vector based on new block ordering.

        The block that sits at slot ``new_order[i]`` is moved back to slot ``i``,
        restoring the vector of the system as it was before
        :meth:`reorder_blocks`. The method is meant to be called on the
        reordered system.

        Parameters
        ----------
        new_order : array_like
            New order of blocks, as returned by :meth:`compute_reordering`.
        vector : array_like
            Vector in the *new* ordering.
        out : array, optional
            Output array to receive the resulting vector contents. Must not be the same
            array as ``vector``.

        Returns
        -------
        array
            Un-re-ordered contents of ``vector``. If ``out`` was specified, this is just a
            reference to it, otherwise a new array is created.
        """
        ...
