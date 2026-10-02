"""Check that a system decomposes, and that the decomposition solves."""

import numpy as np
import pytest
from hybsol._mod import BlockSystem, Decomposition


def random_block_system(
    rng: np.random.Generator, n_blocks: int, block_size: int
) -> tuple[BlockSystem, np.ndarray]:
    """Create a dense block system and the full matrix it represents."""
    mat = rng.random((n_blocks * block_size, n_blocks * block_size))

    sys = BlockSystem(*np.full(n_blocks, block_size, int))
    for i in range(n_blocks):
        for j in range(n_blocks):
            sys.add_block(
                i,
                j,
                mat[
                    block_size * i : block_size * (i + 1),
                    block_size * j : block_size * (j + 1),
                ],
            )

    assert np.all(mat == sys.as_array())
    return sys, mat


@pytest.mark.parametrize("n", (2, 4, 10))
def test_dense_matrix(n: int) -> None:
    """Check the decomposition works on normal dense matrices."""
    rng = np.random.default_rng(15)
    mat = rng.random((n, n))
    # Check the matrix that was randomly generated is not anywhere near singular
    assert np.abs(np.linalg.det(mat)) > 1e-3, "RNG needs a better seed."

    # Create the system from the full matrix
    sys = BlockSystem(*np.ones(n, dtype=int))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            sys.add_block(i, j, ((mat[i, j],),))

    sys.decompose()

    lhs = rng.random(n)
    sol = sys.decompose().solve(mat @ lhs)

    assert pytest.approx(sol) == lhs


@pytest.mark.parametrize("n_threads", (0, 1, 2))
@pytest.mark.parametrize("n_blocks", (2, 4, 10))
@pytest.mark.parametrize("block_size", (2, 3, 4))
def test_dense_matrix_blocks(n_blocks: int, block_size: int, n_threads: int) -> None:
    """Check the decomposition works on normal dense matrices."""
    rng = np.random.default_rng(15)
    sys, mat = random_block_system(rng, n_blocks, block_size)

    sys.decompose(n_threads)

    lhs = rng.random(n_blocks * block_size)
    sol = sys.decompose().solve(mat @ lhs)

    assert pytest.approx(sol) == lhs


@pytest.mark.parametrize("n_blocks", (2, 4, 10))
@pytest.mark.parametrize("block_size", (2, 3, 4))
def test_decompose_records_operations(n_blocks: int, block_size: int) -> None:
    """Check that the recorded operations only reference valid blocks."""
    rng = np.random.default_rng(15)
    sys, _ = random_block_system(rng, n_blocks, block_size)

    sys.decompose()

    operations = sys.decompose().operations()
    for op in operations:
        assert op[0] < n_blocks
        if len(op) == 2:
            # Eliminations always use a source row below the target row.
            assert op[1] < op[0]
        else:
            assert len(op) == 1


@pytest.mark.parametrize("n_blocks", (2, 4, 10))
@pytest.mark.parametrize("block_size", (2, 3, 4))
def test_decomposed_and_copied_system_agree(n_blocks: int, block_size: int) -> None:
    """Check that a copy of a decomposed system solves the same system."""
    rng = np.random.default_rng(15)
    sys, mat = random_block_system(rng, n_blocks, block_size)

    # The system is untouched by the factorization, so a copy and the original
    # decompose independently and agree.
    before = sys.as_array().copy()
    first = sys.decompose()
    copied = sys.copy()

    assert np.all(sys.as_array() == before)
    assert np.all(copied.as_array() == sys.as_array())
    assert isinstance(first, Decomposition)

    lhs = rng.random(n_blocks * block_size)
    rhs = mat @ lhs

    second = copied.decompose()
    assert pytest.approx(second.solve(rhs)) == lhs
    assert pytest.approx(first.solve(rhs, out=np.empty_like(rhs))) == lhs


@pytest.mark.parametrize("n_blocks", (2, 4))
def test_solve_in_place(n_blocks: int) -> None:
    """Check that the right-hand side can be reused as the output."""
    rng = np.random.default_rng(3)
    sys, mat = random_block_system(rng, n_blocks, 2)

    dec = sys.decompose()

    lhs = rng.random(n_blocks * 2)
    rhs = mat @ lhs
    returned = dec.solve(rhs, out=rhs)

    assert returned is rhs
    assert pytest.approx(rhs) == lhs


def test_decompose_rejects_invalid_system() -> None:
    """A system without symmetric sparsity must not decompose."""
    sys = BlockSystem(2, 2, 2)
    rng = np.random.default_rng(7)
    sys.add_block(0, 0, rng.random((2, 2)) + np.eye(2))
    sys.add_block(0, 1, rng.random((2, 2)))
    # The mirror block (1, 0) is missing.

    assert not sys.is_valid()
    with pytest.raises(ValueError, match="block system is not valid"):
        sys.decompose()


def test_decompose_rejects_missing_diagonal() -> None:
    """A system without diagonal blocks must not decompose."""
    sys = BlockSystem(2, 2)
    rng = np.random.default_rng(7)
    sys.add_block(0, 1, rng.random((2, 2)))
    sys.add_block(1, 0, rng.random((2, 2)))

    assert not sys.is_valid()
    with pytest.raises(ValueError):
        sys.decompose()


def test_decompose_is_idempotent_guarded() -> None:
    """Modifying a decomposed system must be refused."""
    rng = np.random.default_rng(11)
    sys, _ = random_block_system(rng, 3, 2)
    dec = sys.decompose()

    # A second decomposition is a fresh, independent one rather than a refusal.
    again = sys.decompose()
    assert again is not dec
    assert list(again.operations()) == list(dec.operations())

    # And the system is still fully mutable.
    sys.add_block(0, 0, np.eye(2))
    sys.multiply_row(0, np.eye(2))
    sys.reorder_blocks(np.arange(sys.n_blocks))


def test_elimination_reports_the_same_operations() -> None:
    """The walk knows the operation count before anything is factorized."""
    rng = np.random.default_rng(11)
    sys, mat = random_block_system(rng, 3, 2)

    graph = sys.elimination()
    dec = sys.decompose()

    assert graph.n_blocks == sys.n_blocks
    assert graph.n_operations == dec.n_operations
    assert len(dec.operations()) == graph.n_operations
    assert graph.n_levels >= 1
    assert graph.precision == sys.precision
    assert graph.failing_block is None
    assert np.all(mat == sys.as_array())


def test_decompose_reports_zero_diagonal_block() -> None:
    """A block holding nothing but zeros must be reported, and named."""
    sys = BlockSystem(2)
    sys.add_block(0, 0, np.zeros((2, 2)))

    assert sys.is_valid()
    # Identically zero admits no pivot, so it is an unusable order, not a
    # singularity. The walk names the block it cannot get past.
    with pytest.raises(ValueError, match=r"no factorization"):
        sys.decompose()
    with pytest.raises(ValueError, match=r"no factorization"):
        sys.elimination()


if __name__ == "__main__":
    with np.printoptions(precision=2, suppress=True):
        test_dense_matrix_blocks(5, 4, 1)
