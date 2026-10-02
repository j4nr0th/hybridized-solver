"""Check the elimination graph: the pattern it predicts, and what it costs."""

import numpy as np
import pytest
from hybsol import BlockSystem, Elimination, Precision


def fill_in_system() -> tuple[BlockSystem, np.ndarray]:
    """Four 1x1 blocks whose fill-in adds a column to two of the rows.

    The stored pattern is ``{0,1}``, ``{0,1,2,3}``, ``{1,2}``, ``{1,3}``;
    eliminating rows 2 and 3 against each other's block grows both to
    ``{1,2,3}``.
    """
    mat = np.zeros((4, 4))
    for row, col in ((1, 0), (2, 1), (3, 1)):
        mat[row, col] = mat[col, row] = 1.0 + 0.5 * (row + col)
    np.fill_diagonal(mat, np.arange(4.0) + 4.0)

    # Only the stored blocks, not the zeros the fill-in will create.
    stored = np.nonzero(mat)
    return BlockSystem.from_blocks([1] * 4, stored[0], stored[1], mat[stored]), mat


def test_graph_of_the_fill_in_example() -> None:
    """The graph names the final pattern, the passes and the cost."""
    sys, _ = fill_in_system()
    graph = sys.elimination()

    assert isinstance(graph, Elimination)
    assert graph.n_blocks == 4
    assert graph.precision == Precision.DOUBLE
    assert graph.failing_block is None

    # Rows 2 and 3 each gain the column the other brought in.
    assert graph.n_columns == 12
    assert [graph.row_columns(i) for i in range(4)] == [
        (0, 1),
        (0, 1, 2, 3),
        (1, 2, 3),
        (1, 2, 3),
    ]
    assert [graph.row_length(i) for i in range(4)] == [2, 4, 3, 3]

    # Four diagonal solves and four eliminations.
    assert graph.n_operations == 8
    assert graph.n_levels == 4
    assert [graph.level_rows(p) for p in range(graph.n_levels)] == [
        (0,),
        (1,),
        (2, 3),
        (3,),
    ]

    assert [graph.row_n_eliminations(i) for i in range(4)] == [0, 1, 1, 2]
    assert [graph.row_first_level(i) for i in range(4)] == [0, 1, 2, 2]
    assert [graph.row_level(i) for i in range(4)] == [0, 1, 2, 3]

    # Every block is a 1x1 double, rounded up to the library's 64-byte granularity.
    assert graph.value_bytes == 12 * 64
    assert graph.total_bytes > 0


def test_graph_is_reproducible() -> None:
    """Two walks of the same system agree exactly."""
    sys, _ = fill_in_system()
    first = sys.elimination()
    second = sys.elimination()

    assert first.total_bytes == second.total_bytes
    assert first.value_bytes == second.value_bytes
    assert first.n_operations == second.n_operations
    for row in range(sys.n_blocks):
        assert first.row_columns(row) == second.row_columns(row)


def test_graph_matches_the_decomposition() -> None:
    """What the walk predicts is what the factorization records."""
    sys, mat = fill_in_system()
    graph = sys.elimination()
    dec = sys.decompose()

    assert graph.n_operations == dec.n_operations == len(dec.operations())
    assert graph.n_blocks == dec.n_blocks == sys.n_blocks
    assert dec.total_size == sys.as_array().shape[0]
    assert dec.is_factorized

    # Solving through the decomposition reproduces the system it came from.
    rhs = mat @ np.ones(4)
    assert pytest.approx(dec.solve(rhs)) == np.ones(4)


def test_out_of_range_queries_are_rejected() -> None:
    """Indices the core would only accept under an assertion are checked here."""
    sys, _ = fill_in_system()
    graph = sys.elimination()

    with pytest.raises(ValueError, match="range"):
        graph.row_columns(4)
    with pytest.raises(ValueError, match="range"):
        graph.row_columns(-1)
    with pytest.raises(ValueError, match="range"):
        graph.level_rows(graph.n_levels)
    with pytest.raises(ValueError, match="range"):
        graph.row_level(-1)
    with pytest.raises(ValueError, match="range"):
        graph.row_first_level(9)


def test_invalid_systems_are_rejected() -> None:
    """A walk refuses what a factorization would refuse, before touching values."""
    rng = np.random.default_rng(7)

    # No diagonal block.
    sys = BlockSystem(2, 2)
    sys.add_block(0, 1, rng.random((2, 2)))
    sys.add_block(1, 0, rng.random((2, 2)))
    with pytest.raises(ValueError, match="not valid"):
        sys.elimination()

    # A block below the diagonal with no mirror above it.
    sys = BlockSystem(2, 2, 2)
    sys.add_block(0, 0, rng.random((2, 2)) + np.eye(2))
    sys.add_block(1, 1, rng.random((2, 2)) + np.eye(2))
    sys.add_block(1, 0, rng.random((2, 2)))
    with pytest.raises(ValueError, match="not valid"):
        sys.elimination()


def test_unusable_order_is_named() -> None:
    """A structurally zero diagonal admits no pivot, and the block is reported."""
    sys = BlockSystem(2)
    sys.add_block(0, 0, np.zeros((2, 2)))

    assert sys.is_valid()
    with pytest.raises(ValueError, match=r"no factorization.*\(block 0\)"):
        sys.elimination()


def test_dense_system_reaches_the_operation_maximum() -> None:
    """A fully coupled system records one operation per (target, source) pair."""
    rng = np.random.default_rng(5)
    n, size = 3, 2
    mat = rng.random((n * size, n * size)) + np.eye(n * size) * 2.0
    rows = np.repeat(np.arange(n), n)
    cols = np.tile(np.arange(n), n)
    sys = BlockSystem.from_blocks([size] * n, rows, cols, mat.ravel())

    graph = sys.elimination()
    assert graph.n_operations == n * (n + 1) / 2
    assert graph.n_levels == n
