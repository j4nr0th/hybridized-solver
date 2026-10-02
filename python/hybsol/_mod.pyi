"""Stub for the C extension module _mod."""

from collections.abc import Sequence
from typing import Literal, Self

import numpy as np
from numpy import typing as npt

from hybsol import Precision as Precision

OrderingStrategy = Literal["first", "greedy", "balanced"]

class SingularSystemError(ValueError):
    """A diagonal block could not be factorized.

    Raised when the LU factorization of a diagonal block hits a zero pivot
    and nothing repairs it. The failure is a property of the system rather
    than of the call, so the caller can retry at a different block
    granularity. The message names the block. Subclasses :class:`ValueError`,
    so existing handlers still catch it.

    Note that a *structurally* zero diagonal block -- one that holds only
    zeros -- is not reported this way: no block order can rescue it, so
    :meth:`BlockSystem.decompose` and :meth:`BlockSystem.elimination` raise
    :class:`ValueError` for it instead.
    """

class DeviceError(RuntimeError):
    """A backend device could not be reached, or could not do what was asked.

    Raised for a device index that names nothing, a device that lacks a
    capability the operation needs (double precision, say), and for failures
    of the device runtime itself, including a kernel build that did not
    compile. Subclasses :class:`RuntimeError`: the machine is at fault, not
    the call, so a caller that only wants to catch bad input still does not
    see it.
    """

class BlockSystem:
    """Block system for hybridized solver.

    The system is an ``n x n`` arrangement of blocks, where block ``(i, j)`` has
    shape ``(block_sizes[i], block_sizes[j])``. Not every block needs to be
    stored: the fill-in supplies the rest. Every diagonal block and every mirror
    of a block below the diagonal *must* be stored; :meth:`is_valid` reports
    whether the structure is decomposable.

    ``precision`` fixes the type blocks are stored in and computed in for the
    life of the system.
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

            A block and its transpose share a row-major ravel *only* when one of
            them has a single column.
        precision : hybsol.Precision, default: Precision.DOUBLE
            Type the blocks are stored in. Values are converted on the way in,
            so ``data`` may hold any floating-point type.

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
        precision : hybsol.Precision, default: Precision.DOUBLE
            Type the blocks are stored in. Unlike :meth:`add_block`, the values
            are *not* converted: each ``value`` must already have this type,
            and a ``float64`` array is refused for a single-precision system.

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
            Concatenated, row-major block values; block ``k`` occupies
            ``block_sizes[rows[k]] * block_sizes[cols[k]]`` entries.

            A block and its transpose share a row-major ravel *only* when one of
            them has a single column.
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

    def as_array(self) -> npt.NDArray[np.double] | npt.NDArray[np.float32]:
        """Return the system representation as a full matrix.

        Blocks the system does not store come out zero, and the result has the
        system's own :attr:`precision` -- ``float32`` for a single-precision
        system, ``float64`` otherwise.
        """
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

    def get_block(
        self, row: int, col: int
    ) -> npt.NDArray[np.double] | npt.NDArray[np.float32]:
        """Get a copy of the value of the specified block.

        The copy has the system's own :attr:`precision`, so it is ``float32``
        for a single-precision system and ``float64`` otherwise.

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

        Raises
        ------
        ValueError
            An index is outside ``[0, n_blocks)``, or the system does not
            store that block.
        """
        ...

    def block_storage(
        self, row: int, col: int
    ) -> npt.NDArray[np.double] | npt.NDArray[np.float32]:
        """Get writable storage for a block, creating it on first use.

        The shape of the block is implied by the system, so this hands back
        exactly the right buffer to fill in place instead of building a
        temporary and passing it to :meth:`add_block`. The first call adds the
        block to the sparsity pattern and zeroes it; later calls return the
        same storage unchanged, so nothing written so far is lost.

        The buffer has the system's own :attr:`precision`, so it is ``float32``
        for a single-precision system and ``float64`` otherwise.

        The returned array is a view backed by the system, which it keeps
        alive. Operations that only rewrite values in place are visible
        through it, as with any view, but :meth:`eliminate_row` and
        :meth:`reorder_blocks` refuse to run while any such array is alive.
        :meth:`decompose` does not: the decomposition copies what it needs and
        leaves the system, and its storage, alone.

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
            A previously returned view is still alive in front of
            :meth:`eliminate_row` or :meth:`reorder_blocks`.
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

    def no_lower_connections(self) -> npt.NDArray[np.bool_]:
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

    def matvec(
        self,
        x: npt.ArrayLike,
        out: npt.NDArray[np.double] | None = None,
        n_threads: int = 0,
    ) -> npt.NDArray[np.double]:
        """Apply the system to a vector, ``y = A x``.

        This is the operator the elimination solves against, and the one worth
        having fast: it walks the stored blocks rather than a dense form, so a
        sparse system costs the sum of its blocks rather than the square of its
        dimension. It is what :func:`hybsol.refined_solve` measures its
        residual with.

        ``out`` is written, not accumulated into. The work is split across block
        rows, each of which writes only its own slice, so the system is only read
        and stays usable throughout.

        Parameters
        ----------
        x : array_like
            Right-hand side of length ``sum(block_sizes)``. Narrower floating
            point is widened as it is read.
        out : array, optional
            Array to write the result to. If not given (or ``None``), a new array
            is created.
        n_threads : int, default: 0
            Number of OpenMP threads; ``0`` selects the OpenMP default and ``1``
            runs it serially.

        Returns
        -------
        array
            The result, always in double precision.
        """
        ...

    def matmat(
        self,
        x: npt.ArrayLike,
        out: npt.NDArray[np.double] | None = None,
        n_threads: int = 0,
    ) -> npt.NDArray[np.double]:
        """Apply the system to several right-hand sides, ``y = A x``.

        The same walk as :meth:`matvec`, done for every column of ``x`` at once,
        so the pattern is traversed once rather than once per column. With one
        column the two are identical.

        Parameters
        ----------
        x : array_like
            Two-dimensional input of ``sum(block_sizes)`` rows. Narrower floating
            point is widened as it is read.
        out : array, optional
            Array to write the result to, with the same shape as ``x``. If not
            given (or ``None``), a new array is created.
        n_threads : int, default: 0
            Number of OpenMP threads; ``0`` selects the OpenMP default and ``1``
            runs it serially.

        Returns
        -------
        array
            The result, always in double precision.
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

    def decompose(
        self,
        n_threads: int = 0,
        workspace: npt.NDArray[np.uint8] | None = None,
        precision: Precision | None = None,
    ) -> Decomposition:
        """Factorize the system and return the result.

        The system is left exactly as it was assembled, so it can be
        decomposed again -- under a different block order, or with a
        different thread count -- and the decompositions are independent.
        The factorization runs in a destination that owns a copy of every
        block it needs, including the fill-in.

        Parameters
        ----------
        n_threads : int, default: 0
            Number of OpenMP threads to use. ``0`` selects the OpenMP default
            (usually every core) and ``1`` runs the factorization serially.
        workspace : numpy.typing.NDArray[numpy.uint8], optional
            A writable 1-D array of at least :meth:`workspace_bytes` bytes.
            Passing one keeps the scratch out of the library's allocator, so a
            buffer can be reused across systems. Its contents are overwritten.
            Omit it and the scratch is allocated internally.

        precision : hybsol.Precision, optional
            The precision of the factors, which need not match the system's:
            a double system can produce single factors and a single system
            double ones. Defaults to the system's own precision. Values
            convert on the way in, so the factorization is exact for the
            precision it runs in, but the values themselves are still the
            system's.

        Returns
        -------
        Decomposition
            The factorized system.

        Raises
        ------
        ValueError
            The system does not satisfy the solver's structural assumptions;
            a diagonal block holds nothing but zeros, which no block order
            can rescue; ``n_threads`` is negative; or ``workspace`` is too
            small or read-only.
        hybsol.SingularSystemError
            A diagonal block is singular, so the LU factorization hit a zero
            pivot. The message names the block.
        """
        ...

    def workspace_bytes(
        self, n_threads: int = 0, precision: Precision | None = None
    ) -> int:
        """Bytes of scratch :meth:`decompose` needs at this thread count.

        Sizes the ``workspace`` array :meth:`decompose` accepts, so one buffer
        can be allocated once and reused. It depends only on the block sizes,
        the precision and the thread count, so it may be asked for before any
        blocks are added.

        Parameters
        ----------
        n_threads : int, default: 0
            The thread count that will be passed to :meth:`decompose`.
        precision : hybsol.Precision, optional
            The precision :meth:`decompose` will factor in. Defaults to the
            system's own.
        """
        ...

    def elimination(self) -> Elimination:
        """Walk the elimination graph without computing any values.

        Reports the pattern the fill-in will produce, the passes a
        factorization runs in and what it will cost. Cheap enough to ask for
        while deciding on a block structure, and it touches nothing, so the
        system may still be assembled.

        Returns
        -------
        Elimination
            The symbolic result for this system.

        Raises
        ------
        ValueError
            The system does not satisfy the solver's structural assumptions,
            or a diagonal block holds nothing but zeros, which no block order
            can rescue. The message names the offending block.
        """
        ...

    def copy(self) -> BlockSystem:
        """Create a deep copy of the system."""
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

        Raises
        ------
        ValueError
            ``new_order`` is not a permutation of ``[0, n_blocks)``;
            ``n_threads`` is negative; the system does not satisfy the
            solver's structural assumptions; or a diagonal block holds
            nothing but zeros, which no block order can rescue.
        RuntimeError
            An array returned by :meth:`block_storage` is still alive.
        """
        ...

    def compute_reordering(
        self,
        strategy: OrderingStrategy = "first",
        max_colors: int = 0,
    ) -> npt.NDArray[np.uint64]:
        """Find a *coloring* of the blocks.

        Grouped so that no two blocks in a group share a non-zero off-diagonal
        block.

        This is **not** a factorization ordering. A coloring makes no promise
        about which blocks come first, so :meth:`decompose` may fill in far
        more than it would have to. Choose the permutation yourself, with every
        block ahead of the blocks it couples to, and pass *that* to
        :meth:`reorder_blocks`.

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

class Decomposition:
    """A factorized block system.

    Produced by :meth:`BlockSystem.decompose`. It owns a copy of every
    block the elimination needs, so the system it came from is unchanged
    and can be decomposed again under a different block order.
    """

    @property
    def n_blocks(self) -> int:
        """Number of blocks."""
        ...

    @property
    def total_size(self) -> int:
        """Rows of the matrix the decomposition solves."""
        ...

    @property
    def n_operations(self) -> int:
        """Number of operations the factorization records."""
        ...

    def solve(
        self,
        val: npt.ArrayLike,
        out: npt.NDArray[np.double] | None = None,
        n_threads: int = 0,
    ) -> npt.NDArray[np.double]:
        """Solve the system for the given right side.

        The forward substitution runs one elimination pass at a time,
        parallel across the block rows of a pass; the back substitution that
        follows is serial. The answer does not depend on the thread count,
        and the decomposition may be solved any number of times.

        Parameters
        ----------
        val : array_like
            Right-hand side of length ``sum(block_sizes)``.
        out : array, optional
            Array to write the solution to. If not given (or ``None``), a new array
            is created. It may be the same array as ``val``, in which case the
            solution overwrites the right-hand side in place.
        n_threads : int, default: 0
            Number of OpenMP threads for the forward substitution; ``0``
            selects the OpenMP default and ``1`` runs it serially.

        Returns
        -------
        array
            Solution of the linear system.

        Raises
        ------
        ValueError
            ``val`` does not hold exactly ``total_size`` values;
            ``n_threads`` is negative; or ``out`` is not a writable
            ``float64`` array of that length.
        """
        ...

    def operations(self) -> tuple[tuple[int, ...], ...]:
        """Get the recorded operations as tuples of one or two ints.

        A one-element tuple ``(idx_row,)`` solves with the LU factors of the
        diagonal block; a two-element tuple ``(idx_row, idx_col)`` eliminates
        block ``(idx_row, idx_col)`` using block row ``idx_col``.

        The list is rebuilt from the decomposition's own copy of the schedule
        rather than stored, so it costs nothing until it is asked for and
        materializing it in Python is ``O(ops)`` in objects.
        """
        ...

    @property
    def device(self) -> int | None:
        """Index of the backend device the factors live on, or None.

        Decompositions from :meth:`BlockSystem.decompose` keep their factors
        in host memory and report None. A backend module -- ``hybsol.opencl``,
        say -- produces decompositions whose factors stay on the device and
        report its index, out of the range its own ``devices()`` lists.
        """
        ...

class Elimination:
    """The symbolic result of eliminating a system.

    Produced by :meth:`BlockSystem.elimination`. It computes no values, so
    it can be asked for before a factorization is committed.

    A successful walk does not promise the factorization will succeed:
    :meth:`BlockSystem.decompose` can still hit a singular diagonal block.
    """

    @property
    def n_blocks(self) -> int:
        """Number of blocks."""
        ...

    @property
    def n_columns(self) -> int:
        """Blocks the final pattern holds, fill-in included."""
        ...

    @property
    def n_operations(self) -> int:
        """Operations a factorization of this system records."""
        ...

    @property
    def n_levels(self) -> int:
        """Passes the elimination takes."""
        ...

    @property
    def value_bytes(self) -> int:
        """Bytes of block storage a decomposition needs, fill-in included."""
        ...

    @property
    def total_bytes(self) -> int:
        """Bytes this graph itself occupies.

        The block storage it describes is counted separately in
        ``value_bytes``, so this may be the smaller of the two.
        """

    @property
    def precision(self) -> Precision:
        """The precision the analyzed system stores."""
        ...

    @property
    def failing_block(self) -> int | None:
        """Block whose diagonal is identically zero, if any.

        Always ``None`` in practice: :meth:`BlockSystem.elimination` raises
        :class:`ValueError` -- naming the block -- instead of returning a
        graph for a system that has one.
        """
        ...

    def row_columns(self, row: int) -> tuple[int, ...]:
        """Column indices ``row`` holds once the fill-in is complete.

        Includes every block the system already stores and every block the
        elimination adds, in increasing order.

        Parameters
        ----------
        row : int
            Block row index, in ``[0, n_blocks)``.

        Raises
        ------
        ValueError
            ``row`` is outside ``[0, n_blocks)``.
        """
        ...

    def level_rows(self, level: int) -> tuple[int, ...]:
        """Block rows the given elimination pass processes.

        A row appears in every pass from its first to its last, so a row
        whose source is not ready yet simply sits a pass out. Within a pass
        the rows are in ascending index order, which is the order the recorded
        operations come out in.

        Parameters
        ----------
        level : int
            Pass index, in ``[0, n_levels)``.

        Raises
        ------
        ValueError
            ``level`` is outside ``[0, n_levels)``.
        """
        ...

    def row_length(self, row: int) -> int:
        """Blocks ``row`` holds once the fill-in is complete.

        Raises
        ------
        ValueError
            ``row`` is outside ``[0, n_blocks)``.
        """
        ...

    def row_n_eliminations(self, row: int) -> int:
        """How many eliminations ``row`` performs.

        Raises
        ------
        ValueError
            ``row`` is outside ``[0, n_blocks)``.
        """
        ...

    def row_first_level(self, row: int) -> int:
        """First pass ``row`` is processed in.

        Raises
        ------
        ValueError
            ``row`` is outside ``[0, n_blocks)``.
        """
        ...

    def row_level(self, row: int) -> int:
        """Last pass ``row`` is processed in.

        Raises
        ------
        ValueError
            ``row`` is outside ``[0, n_blocks)``.
        """
        ...
