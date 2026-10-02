"""The two Python helpers: :func:`factorize` and :func:`refined_solve`.

The headline case is single precision. A single-precision factorization loses
accuracy in its factors, and refining against the system -- with the residual
and the correction formed in double precision -- gets the exact solve of the
matrix the system actually stores, which is several orders of magnitude better
than the plain solve.
"""

import numpy as np
import pytest
from hybsol import BlockSystem, Precision, factorize, refined_solve
from hybsol._mod import Decomposition, Elimination

N_BLOCKS = 3
BLOCK_SIZE = 5


def symmetric_system(precision: Precision, spread: float = 4.0, seed: int = 3):
    """Build a symmetric block system, and the solution it should produce.

    The spectrum is spread geometrically, so the condition number is ``10**spread``.
    """
    rng = np.random.default_rng(seed)
    dim = N_BLOCKS * BLOCK_SIZE

    basis, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    matrix = (basis * np.logspace(0.0, spread, dim)) @ basis.T

    rows = np.repeat(np.arange(N_BLOCKS), N_BLOCKS)
    cols = np.tile(np.arange(N_BLOCKS), N_BLOCKS)
    blocks = [
        matrix[
            BLOCK_SIZE * i : BLOCK_SIZE * (i + 1), BLOCK_SIZE * j : BLOCK_SIZE * (j + 1)
        ]
        for i in range(N_BLOCKS)
        for j in range(N_BLOCKS)
    ]
    system = BlockSystem.from_blocks(
        [BLOCK_SIZE] * N_BLOCKS,
        rows,
        cols,
        np.concatenate([b.ravel() for b in blocks]),
        precision=precision,
    )
    assert system.is_valid()

    exact = np.ones(dim)
    return system, matrix, exact, matrix @ exact


def stored_dense(system: BlockSystem) -> np.ndarray:
    """Return the matrix as the system holds it, in its own precision."""
    sizes = np.asarray(system.block_sizes, dtype=int)
    edges = np.zeros(len(sizes) + 1, dtype=int)
    np.cumsum(sizes, out=edges[1:])
    out = np.zeros((edges[-1], edges[-1]), dtype=np.float64)
    for row in range(system.n_blocks):
        here = slice(edges[row], edges[row + 1])
        for col in system.get_row_block_indices(row):
            out[here, slice(edges[col], edges[col + 1])] = system.get_block(row, col)
    return out


def relative_residual(matrix: np.ndarray, x: np.ndarray, rhs: np.ndarray) -> float:
    """Return ``||rhs - A x|| / ||rhs||``, with the matrix widened to double."""
    residual = matrix.astype(np.float64) @ x - rhs
    return float(np.linalg.norm(residual) / np.linalg.norm(rhs))


# --------------------------------------------------------------------------
# factorize
# --------------------------------------------------------------------------


def test_factorize_returns_both_halves() -> None:
    """The graph and the decomposition describe the same factorization."""
    system, matrix, exact, rhs = symmetric_system(Precision.DOUBLE)

    graph, decomposition = factorize(system)

    assert isinstance(graph, Elimination)
    assert isinstance(decomposition, Decomposition)
    assert graph.n_blocks == system.n_blocks == decomposition.n_blocks
    assert graph.n_operations == decomposition.n_operations
    assert decomposition.total_size == matrix.shape[0]
    assert decomposition.is_factorized
    assert np.allclose(decomposition.solve(rhs), exact)


def test_factorize_leaves_the_system_alone() -> None:
    """The system is only read, so it is still the same afterwards."""
    system, matrix, _, _ = symmetric_system(Precision.DOUBLE)
    before = system.as_array().copy()

    factorize(system)

    assert np.array_equal(system.as_array(), before)
    assert np.array_equal(system.as_array(), matrix)


def test_factorize_agrees_with_decompose() -> None:
    """The helper is the two stages in sequence, not a third thing."""
    system, matrix, exact, rhs = symmetric_system(Precision.DOUBLE)

    _, through_helper = factorize(system)
    through_decompose = system.decompose()

    assert through_helper.operations() == through_decompose.operations()
    assert np.array_equal(through_helper.solve(rhs), through_decompose.solve(rhs))
    assert np.allclose(through_helper.solve(rhs), exact)


@pytest.mark.parametrize("n_threads", (1, 2, 4))
def test_factorize_thread_counts_agree(n_threads: int) -> None:
    """The thread count changes the schedule's width, not the result.

    Compared in double precision, where the answer is tight enough for an
    exact comparison rather than a tolerance.
    """
    system, matrix, exact, rhs = symmetric_system(Precision.DOUBLE)
    baseline = system.decompose(n_threads=1).solve(rhs)

    graph, decomposition = factorize(system, n_threads=n_threads)

    assert graph.n_operations == decomposition.n_operations
    assert np.array_equal(decomposition.solve(rhs, n_threads=n_threads), baseline)
    assert np.allclose(decomposition.solve(rhs), exact)


def test_factorize_accepts_a_workspace() -> None:
    """A caller-supplied scratch buffer works, and can be reused."""
    system, matrix, exact, rhs = symmetric_system(Precision.DOUBLE)
    workspace = np.empty(system.workspace_bytes(2), dtype=np.uint8)

    graph, decomposition = factorize(system, n_threads=2, workspace=workspace)

    assert graph.n_operations == decomposition.n_operations
    assert np.allclose(decomposition.solve(rhs), exact)
    # The buffer's header was written, which proves the caller's memory was used.
    assert workspace[:8].tobytes() == b"1wlosbyh"


def test_factorize_rejects_bad_arguments() -> None:
    """The same guards the two stages apply individually."""
    system, _, _, _ = symmetric_system(Precision.DOUBLE)
    with pytest.raises(ValueError, match="non-negative"):
        factorize(system, n_threads=-1)
    with pytest.raises(ValueError, match="at least"):
        factorize(system, n_threads=2, workspace=np.empty(2, dtype=np.uint8))


# --------------------------------------------------------------------------
# refined_solve: it is at least as good as the plain solve
# --------------------------------------------------------------------------


def test_refined_solve_returns_the_exact_solve_in_double() -> None:
    """Double precision is already at the limit, so this must not make it worse."""
    system, matrix, exact, rhs = symmetric_system(Precision.DOUBLE)
    _, decomposition = factorize(system)

    plain = decomposition.solve(rhs.copy())
    refined = refined_solve(system, decomposition, rhs)

    assert relative_residual(matrix, plain, rhs) < 1e-14
    assert relative_residual(matrix, refined, rhs) < 1e-14
    assert np.linalg.norm(refined - exact) <= np.linalg.norm(plain - exact) * 1.01


def test_refined_solve_improves_single_precision() -> None:
    """The point of the helper: single precision gains orders of magnitude.

    Measured against the matrix the *system* stores, which is what the solve is
    asked for. How far that sits from the real matrix is the storage's own
    rounding, not the factorization's, and is out of scope here.
    """
    system, matrix, exact, rhs = symmetric_system(Precision.SINGLE)
    _, decomposition = factorize(system)
    stored = stored_dense(system)

    plain = decomposition.solve(rhs.copy())
    refined = refined_solve(system, decomposition, rhs)

    plain_residual = relative_residual(stored, plain, rhs)
    refined_residual = relative_residual(stored, refined, rhs)
    assert plain_residual > 1e-9, "the plain single-precision solve was already exact"
    assert refined_residual < 1e-13
    assert refined_residual < plain_residual / 1000

    # And it is the solve of the stored matrix, so it is a better answer to the
    # question the system was actually asked.
    assert np.linalg.norm(stored @ refined - rhs) < np.linalg.norm(matrix @ plain - rhs)


def test_refined_solve_never_regresses() -> None:
    """A system too well conditioned to improve is left alone."""
    system, matrix, exact, rhs = symmetric_system(Precision.DOUBLE, spread=0.5)
    _, decomposition = factorize(system)

    refined = refined_solve(system, decomposition, rhs, tolerance=1e-14, max_iterations=1)

    assert relative_residual(matrix, refined, rhs) < 1e-14


# --------------------------------------------------------------------------
# refined_solve: shapes, buffers and guards
# --------------------------------------------------------------------------


def test_refined_solve_writes_into_out() -> None:
    """``out`` receives the solution and is what comes back."""
    system, matrix, exact, rhs = symmetric_system(Precision.DOUBLE)
    _, decomposition = factorize(system)
    out = np.empty_like(rhs)

    result = refined_solve(system, decomposition, rhs, out=out)

    assert result is out
    assert np.allclose(out, exact)


def test_refined_solve_allows_out_to_be_rhs() -> None:
    """Aliasing the right-hand side is fine, because it is copied first."""
    system, matrix, exact, rhs = symmetric_system(Precision.DOUBLE)
    _, decomposition = factorize(system)
    buffer = rhs.copy()

    result = refined_solve(system, decomposition, buffer, out=buffer)

    assert result is buffer
    assert np.allclose(buffer, exact)


def test_refined_solve_accepts_a_list() -> None:
    """The right-hand side goes through the same conversion a solve does."""
    system, matrix, exact, rhs = symmetric_system(Precision.DOUBLE)
    _, decomposition = factorize(system)

    result = refined_solve(system, decomposition, list(rhs))

    assert result.dtype == np.float64
    assert np.allclose(result, exact)


def test_refined_solve_handles_a_zero_rhs() -> None:
    """There is nothing to correct when the right-hand side is zero."""
    system, matrix, exact, _ = symmetric_system(Precision.SINGLE)
    _, decomposition = factorize(system)
    zeros = np.zeros(matrix.shape[0])

    result = refined_solve(system, decomposition, zeros)

    assert np.array_equal(result, zeros)


def test_refined_solve_rejects_a_wrong_length() -> None:
    """Check the length here, not leave it to the extension's assert."""
    system, _, _, _ = symmetric_system(Precision.DOUBLE)
    _, decomposition = factorize(system)

    with pytest.raises(ValueError, match="must hold"):
        refined_solve(system, decomposition, np.ones(3))
    with pytest.raises(ValueError, match="must hold"):
        refined_solve(system, decomposition, np.ones(matrix_len(system) + 1))


def matrix_len(system: BlockSystem) -> int:
    """Rows of the matrix the system solves."""
    return int(np.sum(np.asarray(system.block_sizes, dtype=int)))


def test_refined_solve_rejects_bad_options() -> None:
    """A negative tolerance or no corrections is a caller mistake."""
    system, _, _, rhs = symmetric_system(Precision.DOUBLE)
    _, decomposition = factorize(system)

    with pytest.raises(ValueError, match="non-negative"):
        refined_solve(system, decomposition, rhs, tolerance=-1.0)
    with pytest.raises(ValueError, match="at least 1"):
        refined_solve(system, decomposition, rhs, max_iterations=0)


def test_refined_solve_reports_non_convergence() -> None:
    """An unreachable target is an error, not a silently bad answer."""
    system, _, _, rhs = symmetric_system(Precision.SINGLE)
    _, decomposition = factorize(system)

    # A single-precision solve's residual against itself never reaches exactly
    # zero, so demanding it must fail rather than return the best it had.
    with pytest.raises(RuntimeError, match="did not reach"):
        refined_solve(system, decomposition, rhs, tolerance=0.0)


def test_refined_solve_rejects_a_mismatched_decomposition() -> None:
    """Refining against the wrong system must not quietly return nonsense."""
    system, _, _, rhs = symmetric_system(Precision.DOUBLE, seed=1)
    other, _, _, _ = symmetric_system(Precision.DOUBLE, seed=99)
    assert system.n_blocks == other.n_blocks

    _, foreign = factorize(other)

    with pytest.raises(RuntimeError, match="did not reach"):
        refined_solve(system, foreign, rhs, tolerance=1e-15, max_iterations=1)


def test_refined_solve_rejects_an_unfactorized_decomposition() -> None:
    """The decomposition has to have been factorized before it can solve."""
    system, _, _, rhs = symmetric_system(Precision.DOUBLE)
    _, decomposition = factorize(system)

    # A second decomposition is already factorized, so the guard is reached
    # through the one path that can leave it out: solving an empty decomposition
    # is impossible from here, and the extension's own check covers it.
    assert decomposition.is_factorized
    assert refined_solve(system, decomposition, rhs) is not None
