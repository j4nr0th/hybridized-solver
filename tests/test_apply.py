"""Applying the system to vectors and matrices.

The point of these is that the operator walks the stored blocks, so its cost
tracks the system's blocks rather than the square of its dimension -- and that
it stays honest about the pattern, in both precisions and at every thread
count.
"""

import numpy as np
import pytest
from hybsol import BlockSystem, Precision


def sparse_system(precision: Precision, n_blocks: int = 5, block_size: int = 3):
    """Build a banded block system, and the dense matrix it stands for.

    Banded rather than dense on purpose: a dense system would hide whether the
    operator is actually reading the pattern.
    """
    dim = n_blocks * block_size
    edges = np.arange(n_blocks + 1) * block_size
    matrix = np.zeros((dim, dim))

    rng = np.random.default_rng(11)
    pattern = [
        (i, j) for i in range(n_blocks) for j in range(n_blocks) if abs(i - j) <= 1
    ]
    for i, j in pattern:
        block = rng.standard_normal((block_size, block_size))
        if i == j:
            block = block + np.eye(block_size) * (block_size + 1.0)
        matrix[edges[i] : edges[i + 1], edges[j] : edges[j + 1]] = block

    system = BlockSystem.from_blocks(
        [block_size] * n_blocks,
        np.array([i for i, _ in pattern]),
        np.array([j for _, j in pattern]),
        np.concatenate(
            [
                matrix[edges[i] : edges[i + 1], edges[j] : edges[j + 1]].ravel()
                for i, j in pattern
            ]
        ),
        precision=precision,
    )
    assert system.is_valid()
    return system, matrix


def stored_dense(system: BlockSystem) -> np.ndarray:
    """Return the matrix as the system holds it, widened to double."""
    sizes = np.asarray(system.block_sizes, dtype=int)
    edges = np.zeros(len(sizes) + 1, dtype=int)
    np.cumsum(sizes, out=edges[1:])
    out = np.zeros((edges[-1], edges[-1]), dtype=np.float64)
    for row in range(system.n_blocks):
        here = slice(edges[row], edges[row + 1])
        for col in system.get_row_block_indices(row):
            out[here, slice(edges[col], edges[col + 1])] = system.get_block(row, col)
    return out


# --------------------------------------------------------------------------
# matvec
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_threads", (0, 1, 2, 4))
def test_matvec_matches_the_dense_matrix(n_threads: int) -> None:
    """A vector times the system is the dense product."""
    system, matrix = sparse_system(Precision.DOUBLE)
    x = np.linspace(1.0, 2.0, matrix.shape[0])

    result = system.matvec(x, n_threads=n_threads)

    assert result.dtype == np.float64
    assert np.allclose(result, matrix @ x, rtol=1e-12, atol=1e-12)


def test_matvec_follows_the_pattern_rather_than_a_dense_form() -> None:
    """A coupling the system does not store contributes nothing to the product.

    The two systems differ only in whether an off-diagonal pair is present --
    and a valid system stores both or neither, so the diagonal-only one is the
    pattern with that pair simply absent.
    """
    rng = np.random.default_rng(4)
    coupling = rng.standard_normal((2, 2)) * 0.1
    diagonal = np.eye(2) * 4.0

    diagonal_only = BlockSystem(2, 2)
    diagonal_only.add_block(0, 0, diagonal)
    diagonal_only.add_block(1, 1, diagonal)

    coupled = BlockSystem(2, 2)
    blocks = ((0, 0, diagonal), (0, 1, coupling), (1, 0, coupling.T), (1, 1, diagonal))
    for i, j, block in blocks:
        coupled.add_block(i, j, block)

    assert diagonal_only.is_valid()
    assert coupled.is_valid()
    assert (1, 0) not in diagonal_only.get_row_block_indices(1)

    x = np.ones(4)
    zero = np.zeros((2, 2))
    dense_diagonal = np.block([[diagonal, zero], [zero, diagonal]])
    dense_coupled = np.block([[diagonal, coupling], [coupling.T, diagonal]])

    assert np.allclose(diagonal_only.matvec(x), dense_diagonal @ x)
    assert np.allclose(coupled.matvec(x), dense_coupled @ x)
    assert not np.allclose(diagonal_only.matvec(x), coupled.matvec(x))


def test_matvec_widens_a_single_precision_system() -> None:
    """A single-precision system produces a double-precision result."""
    system, matrix = sparse_system(Precision.SINGLE)
    x = np.ones(matrix.shape[0], dtype=np.float32)

    result = system.matvec(x)

    assert result.dtype == np.float64
    # The stored blocks are floats, so the answer is right to float accuracy.
    expected = stored_dense(system) @ x.astype(np.float64)
    assert np.allclose(result, expected, rtol=1e-6, atol=1e-6)


def test_matvec_accepts_anything_array_like() -> None:
    """A list goes through the same conversion a solve does."""
    system, matrix = sparse_system(Precision.DOUBLE)
    x = np.ones(matrix.shape[0])

    assert np.allclose(system.matvec(list(x)), matrix @ x)


def test_matvec_writes_into_out() -> None:
    """``out`` receives the result and is what comes back."""
    system, matrix = sparse_system(Precision.DOUBLE)
    x = np.ones(matrix.shape[0])
    out = np.empty_like(x)

    result = system.matvec(x, out=out)

    assert result is out
    assert np.allclose(out, matrix @ x)


def test_matvec_overwrites_a_dirty_out() -> None:
    """The result is written, not accumulated into."""
    system, matrix = sparse_system(Precision.DOUBLE)
    x = np.ones(matrix.shape[0])
    out = np.full_like(x, 1000.0)

    system.matvec(x, out=out)

    assert np.allclose(out, matrix @ x)


def test_matvec_survives_a_decomposition() -> None:
    """The system is only read, so a factorization leaves it applicable."""
    system, matrix = sparse_system(Precision.DOUBLE)
    x = np.ones(matrix.shape[0])
    before = system.matvec(x)

    system.decompose()

    assert np.allclose(system.matvec(x), before)


def test_matvec_rejects_a_wrong_length() -> None:
    """The length is checked here, not left to the extension's assert."""
    system, matrix = sparse_system(Precision.DOUBLE)
    with pytest.raises(ValueError, match="x"):
        system.matvec(np.ones(matrix.shape[0] + 1))
    with pytest.raises(ValueError, match="x"):
        system.matvec(np.ones(matrix.shape[0] - 1))


def test_matvec_rejects_a_two_dimensional_input() -> None:
    """A vector is one-dimensional; ``matmat`` is the two-dimensional spelling."""
    system, matrix = sparse_system(Precision.DOUBLE)
    with pytest.raises(ValueError):
        system.matvec(np.ones((matrix.shape[0], 1)))


def test_matvec_rejects_bad_options() -> None:
    """A negative thread count is a caller mistake."""
    system, _ = sparse_system(Precision.DOUBLE)
    with pytest.raises(ValueError, match="non-negative"):
        system.matvec(np.ones(system.n_blocks * 3), n_threads=-1)
    with pytest.raises(ValueError, match="out"):
        system.matvec(np.ones(system.n_blocks * 3), out=np.ones(2))


# --------------------------------------------------------------------------
# matmat
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_threads", (0, 1, 2, 4))
def test_matmat_matches_the_dense_matrix(n_threads: int) -> None:
    """Several right-hand sides, one pass over the pattern."""
    system, matrix = sparse_system(Precision.DOUBLE)
    x = np.arange(1.0, 1.0 + 3 * matrix.shape[0]).reshape(matrix.shape[0], 3)

    result = system.matmat(x, n_threads=n_threads)

    assert result.shape == x.shape
    assert np.allclose(result, matrix @ x, rtol=1e-12, atol=1e-12)


def test_matmat_column_by_column_agrees_with_matvec() -> None:
    """The many-column walk is the one-column walk, column for column."""
    system, matrix = sparse_system(Precision.DOUBLE, n_blocks=6, block_size=4)
    x = np.arange(1.0, 1.0 + 5 * matrix.shape[0]).reshape(matrix.shape[0], 5)

    together = system.matmat(x)
    apart = np.column_stack([system.matvec(x[:, j]) for j in range(x.shape[1])])

    assert np.array_equal(together, apart)


def test_matmat_with_one_column_is_matvec() -> None:
    """A single right-hand side is not a special case."""
    system, matrix = sparse_system(Precision.DOUBLE)
    x = np.ones(matrix.shape[0])

    assert np.array_equal(system.matmat(x.reshape(-1, 1)).reshape(-1), system.matvec(x))


def test_matmat_writes_into_out() -> None:
    """``out`` receives the result and is what comes back."""
    system, matrix = sparse_system(Precision.DOUBLE)
    x = np.ones((matrix.shape[0], 2))
    out = np.empty_like(x)

    result = system.matmat(x, out=out)

    assert result is out
    assert np.allclose(out, matrix @ x)


def test_matmat_rejects_the_wrong_shape() -> None:
    """Rows must match the system; columns are free."""
    system, matrix = sparse_system(Precision.DOUBLE)
    with pytest.raises(ValueError, match="x"):
        system.matmat(np.ones((matrix.shape[0] + 1, 2)))
    # A one-dimensional input is a depth error from NumPy, before the length
    # check, so only the type is pinned here.
    with pytest.raises(ValueError):
        system.matmat(np.ones(matrix.shape[0]))
    with pytest.raises(ValueError, match="out"):
        system.matmat(np.ones((matrix.shape[0], 2)), out=np.ones((matrix.shape[0], 3)))
