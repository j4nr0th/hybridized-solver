"""Tests that precondition violations raise instead of aborting the interpreter.

The C core checks preconditions with ``HYBSOL_ASSERT``, which aborts the
process. Every one of them therefore has to be caught by the bindings and
turned into an exception first. Each test here calls a method with an argument
that violates a precondition; if a guard is missing the interpreter dies and
the whole run is lost, which is exactly the failure these tests exist to catch.
"""

import numpy as np
import pytest
from hybsol import BlockSystem, Precision


def full_system(n: int = 2) -> BlockSystem:
    """Build a complete, decomposable system of ``n`` blocks of size 2.

    Diagonally dominant with a nonzero off-diagonal, so it is structurally
    valid *and* factorizable: every structural precondition holds, and only the
    argument under test is at fault.

    Returns
    -------
    BlockSystem
        A symmetric system with every block present and a full-rank LU.
    """
    sys = BlockSystem(2, n)
    for i in range(n):
        for j in range(n):
            block = np.ones((2, 2)) if i != j else np.eye(2) * 4.0
            sys.add_block(i, j, block)
    return sys


class TestIndexPreconditions:
    """Every block index must be inside ``[0, n_blocks)``."""

    @pytest.mark.parametrize("index", [-1, 2, 99])
    def test_get_block_rejects_bad_indices(self, index: int) -> None:
        """An index outside ``[0, n_blocks)`` is a ValueError, not an abort."""
        sys = full_system()
        with pytest.raises(ValueError, match="range"):
            sys.get_block(index, 0)
        with pytest.raises(ValueError, match="range"):
            sys.get_block(0, index)

    @pytest.mark.parametrize("index", [-1, 2, 99])
    def test_block_storage_rejects_bad_indices(self, index: int) -> None:
        """Storage is not allocated for an out-of-range index."""
        sys = full_system()
        with pytest.raises(ValueError, match="range"):
            sys.block_storage(index, 0)

    @pytest.mark.parametrize("index", [-1, 2, 99])
    def test_single_index_methods_reject_bad_indices(self, index: int) -> None:
        """Every method taking a bare row index validates it first."""
        sys = full_system()
        with pytest.raises(ValueError, match="range"):
            sys.reserve(index, 4)
        with pytest.raises(ValueError, match="range"):
            sys.first_column(index)
        with pytest.raises(ValueError, match="range"):
            sys.get_next_column_index(index, 0)
        with pytest.raises(ValueError, match="range"):
            sys.multiply_row(index, np.eye(2))

    def test_solve_accepts_a_valid_vector(self) -> None:
        """An in-range call still works, so the guards are not blanket-rejecting."""
        decomposition = full_system().decompose()
        assert decomposition.solve(np.ones(4)).shape == (4,)


class TestShapePreconditions:
    """Block and multiplier shapes must match what the system implies."""

    def test_add_block_rejects_wrong_shape(self) -> None:
        """A block must have the size the system gives it."""
        sys = full_system()
        with pytest.raises(ValueError):
            sys.add_block(0, 0, np.ones((3, 2)))
        with pytest.raises(ValueError):
            sys.add_block(0, 0, np.ones((2, 5)))

    def test_multiply_row_rejects_wrong_size(self) -> None:
        """The multiplier must be square and match the block row."""
        sys = full_system()
        with pytest.raises(ValueError):
            sys.multiply_row(0, np.eye(5))
        with pytest.raises(ValueError):
            sys.multiply_row(0, np.ones((2, 3)))

    def test_solve_rejects_wrong_shapes(self) -> None:
        """Right-hand side and destination must match the system's size."""
        decomposition = full_system().decompose()
        with pytest.raises(ValueError, match="val"):
            decomposition.solve(np.zeros(5))
        with pytest.raises(ValueError):
            decomposition.solve(np.zeros(4), out=np.zeros(7))

    def test_add_blocks_rejects_out_of_range_index(self) -> None:
        """Bulk assembly validates every index in the arrays."""
        sys = full_system()
        with pytest.raises(ValueError):
            sys.add_blocks(np.array([0, 99]), np.array([0, 0]), np.ones(8))

    def test_block_size_of_a_valid_system_still_works(self) -> None:
        """A correctly shaped call must succeed, proving the guards are narrow."""
        sys = full_system()
        sys.multiply_row(0, np.eye(2))
        assert sys.get_block(0, 0).shape == (2, 2)


class TestOrderingPreconditions:
    """A reordering must be a genuine permutation of ``[0, n_blocks)``."""

    @pytest.mark.parametrize("order", [[0, 0], [1, 1], [0, 5], [0, 3]])
    def test_reorder_blocks_rejects_non_permutations(self, order: list[int]) -> None:
        """A repeated or out-of-range index is not a permutation."""
        sys = full_system()
        with pytest.raises(ValueError):
            sys.reorder_blocks(np.array(order))

    def test_reorder_blocks_rejects_negative_index(self) -> None:
        """Negative indices are rejected before the unsigned conversion."""
        sys = full_system()
        with pytest.raises(ValueError):
            sys.reorder_blocks(np.array([-1, 1]))

    def test_reorder_blocks_rejects_wrong_length(self) -> None:
        """The permutation must have exactly one entry per block."""
        sys = full_system()
        with pytest.raises(ValueError):
            sys.reorder_blocks(np.array([0]))

    def test_reorder_vector_rejects_wrong_length(self) -> None:
        """The vector must match the system's total size."""
        sys = full_system()
        with pytest.raises(ValueError):
            sys.reorder_vector(np.array([0, 1]), np.zeros(3))

    def test_a_real_permutation_is_still_accepted(self) -> None:
        """A genuine permutation still reorders, so the check is not too strict."""
        sys = full_system()
        sys.reorder_blocks(np.array([1, 0]))
        assert sys.n_blocks == 2


class TestStructuralPreconditions:
    """Rows must have their diagonal block where one is required."""

    def test_decompose_rejects_a_system_missing_a_diagonal(self) -> None:
        """An invalid pattern is a returned error, not an assert."""
        sys = BlockSystem(2, 2)
        sys.add_block(0, 0, np.eye(2))
        sys.add_block(0, 1, np.eye(2))
        with pytest.raises(ValueError):
            sys.decompose()

    @pytest.mark.parametrize("entry", ["elimination", "decompose"])
    def test_pipeline_rejects_a_row_without_a_diagonal(self, entry: str) -> None:
        """A missing diagonal is refused by both entry points, as a ValueError."""
        sys = BlockSystem(2, 2)
        sys.add_block(0, 0, np.eye(2))
        with pytest.raises(ValueError, match="not valid"):
            getattr(sys, entry)()

    def test_eliminate_row_requires_the_source_column(self) -> None:
        """The target row must actually hold the column being eliminated."""
        sys = BlockSystem(2, 2)
        sys.add_block(0, 0, np.eye(2))
        sys.add_block(1, 1, np.eye(2))
        with pytest.raises(ValueError):
            sys.eliminate_row(1, 0, np.eye(2))

    def test_get_block_rejects_an_absent_block(self) -> None:
        """Looking up a block the system does not store raises."""
        sys = BlockSystem(2, 2)
        sys.add_block(0, 0, np.eye(2))
        with pytest.raises(ValueError, match="does not contain"):
            sys.get_block(0, 1)


class TestCreationPreconditions:
    """Block sizes must be positive and finite."""

    @pytest.mark.parametrize("sizes", [(2, 0), (2, -1), (0,), (3, 0, 2)])
    def test_create_rejects_non_positive_sizes(self, sizes: tuple[int, ...]) -> None:
        """Zero and negative block sizes are refused at construction."""
        with pytest.raises(ValueError):
            BlockSystem(*sizes)


class TestPrecisionIsHandledNotAborted:
    """Precision is converted at the boundary, so no spelling mismatch escapes.

    A ``float64`` array handed to a single-precision system is converted, which
    means the core's precision precondition is unreachable from Python. This
    pins that behaviour down rather than leaving it to chance.
    """

    def test_double_values_are_converted_for_a_single_system(self) -> None:
        """float64 input to a single-precision system is converted, not rejected."""
        sys = BlockSystem(2, 2, precision=Precision.SINGLE)
        sys.add_block(0, 0, np.eye(2))  # float64 input

        block = sys.get_block(0, 0)
        assert block.dtype == np.float32
        assert np.allclose(block, np.eye(2))

    def test_single_values_are_converted_for_a_double_system(self) -> None:
        """float32 input to a double system is converted, not rejected."""
        sys = BlockSystem(2, 2)
        sys.add_block(0, 0, np.eye(2, dtype=np.float32))

        block = sys.get_block(0, 0)
        assert block.dtype == np.float64
        assert np.allclose(block, np.eye(2))

    def test_a_single_system_still_solves(self) -> None:
        """The converted path must remain correct, not merely non-crashing."""
        sys = BlockSystem(2, 2, precision=Precision.SINGLE)
        for i in range(2):
            for j in range(2):
                sys.add_block(i, j, np.eye(2) * (4.0 if i == j else 1.0))
        dec = sys.decompose()

        # The dense operator is [[4, 1], [1, 4]], so a right-hand side of
        # ones has the constant solution 1 / 5 in every component.
        solution = dec.solve(np.ones(4))
        assert np.allclose(solution, np.full(4, 0.2), atol=1e-5)


class TestDecompositionIsGuarded:
    """The decomposition and the graph check their arguments before the core.

    The core asserts on these, so reaching it unchecked would abort the
    interpreter rather than raise.
    """

    @staticmethod
    def full_system() -> BlockSystem:
        """Build a decomposable 2 x 2 system with blocks of size 3."""
        sys = BlockSystem(3, 3)
        sys.add_block(0, 0, np.eye(3) * 4.0)
        sys.add_block(0, 1, np.ones((3, 3)))
        sys.add_block(1, 0, np.ones((3, 3)))
        sys.add_block(1, 1, np.eye(3) * 4.0)
        return sys

    def test_decompose_returns_a_solvable_decomposition(self) -> None:
        """What :meth:`BlockSystem.decompose` hands back is ready to solve."""
        sys = self.full_system()
        dec = sys.decompose()

        assert dec.n_blocks == 2
        assert dec.n_operations > 0

        rhs = np.ones(6)
        assert pytest.approx(dec.solve(rhs), abs=1e-12) == np.linalg.solve(
            sys.as_array(), rhs
        )

    def test_solve_accepts_a_valid_vector(self) -> None:
        """The guard is narrow: a factorized decomposition solves normally."""
        dec = self.full_system().decompose()
        solution = dec.solve(np.ones(6))
        assert solution.shape == (6,)

    def test_elimination_rejects_an_invalid_system(self) -> None:
        """A system the solver cannot use is refused, not asserted on."""
        sys = BlockSystem(2, 2)
        sys.add_block(0, 1, np.eye(2))
        sys.add_block(1, 0, np.eye(2))

        with pytest.raises(ValueError, match="not valid"):
            sys.elimination()
        with pytest.raises(ValueError, match="not valid"):
            sys.decompose()

    @pytest.mark.parametrize("bad", (-1, 2, 99))
    def test_graph_queries_check_their_index(self, bad: int) -> None:
        """Row and pass indices are range-checked before the core is called."""
        graph = self.full_system().elimination()

        with pytest.raises(ValueError, match="range"):
            graph.row_columns(bad)
        with pytest.raises(ValueError, match="range"):
            graph.row_level(bad)
        with pytest.raises(ValueError, match="range"):
            graph.row_length(bad)
        with pytest.raises(ValueError, match="range"):
            graph.level_rows(bad)

    def test_decompose_rejects_a_negative_thread_count(self) -> None:
        """A negative thread count is a caller mistake, not an OpenMP default."""
        sys = self.full_system()
        with pytest.raises(ValueError, match="non-negative"):
            sys.decompose(-1)
        with pytest.raises(ValueError, match="non-negative"):
            sys.workspace_bytes(-1)

    def test_solve_rejects_a_negative_thread_count(self) -> None:
        """The solve takes a thread count too, and checks it the same way."""
        dec = self.full_system().decompose()
        with pytest.raises(ValueError, match="non-negative"):
            dec.solve(np.ones(6), n_threads=-1)
