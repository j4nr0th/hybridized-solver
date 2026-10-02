"""Check the decomposition workspace sizing, reuse and validation."""

import numpy as np
import pytest
from hybsol._mod import BlockSystem


def well_conditioned_system(
    n_blocks: int, block_size: int, seed: int = 0
) -> tuple[BlockSystem, np.ndarray]:
    """Build a symmetric, diagonally dominant system and its dense matrix.

    Diagonal dominance keeps the LU away from a zero pivot, so a
    decomposition that went wrong shows up as a wrong answer rather than an
    error.
    """
    rng = np.random.default_rng(seed)
    mat = np.eye(n_blocks * block_size) * 50.0 + rng.random(
        (n_blocks * block_size, n_blocks * block_size)
    )

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
    return sys, mat


def residual(mat: np.ndarray, x: np.ndarray) -> float:
    """Largest absolute deviation of ``A @ x`` from the right-hand side."""
    b = mat @ np.ones(mat.shape[0])
    return float(np.max(np.abs(mat @ x - b)))


def operations_are_stable(n_threads: int) -> bool:
    """Whether repeated decompositions record byte-identical operation lists."""
    runs = []
    for _ in range(3):
        sys_, _ = well_conditioned_system(10, 3, seed=11)
        runs.append(sys_.decompose(n_threads).operations())
    return all(r == runs[0] for r in runs)


@pytest.mark.parametrize("n_threads", (1, 2, 4))
def test_workspace_matches_internal(n_threads: int) -> None:
    """A supplied workspace gives the same answer as the internal one."""
    sys_ref, mat_ref = well_conditioned_system(8, 6, seed=1)
    dec_ref = sys_ref.decompose(n_threads)
    x_ref = dec_ref.solve(mat_ref @ np.ones(mat_ref.shape[0]))

    sys_ws, mat_ws = well_conditioned_system(8, 6, seed=1)
    workspace = np.empty(sys_ws.workspace_bytes(n_threads), dtype=np.uint8)
    dec_ws = sys_ws.decompose(n_threads, workspace=workspace)
    x_ws = dec_ws.solve(mat_ws @ np.ones(mat_ws.shape[0]))

    assert np.array_equal(x_ref, x_ws)
    assert list(dec_ws.operations()) == list(dec_ref.operations())


def test_workspace_is_written_to() -> None:
    """The caller's own buffer is what gets used, not a private copy."""
    sys, _ = well_conditioned_system(6, 4, seed=2)
    workspace = np.zeros(sys.workspace_bytes(2), dtype=np.uint8)
    sys.decompose(2, workspace=workspace)
    # The layout starts with a magic word; seeing it proves the buffer was used.
    assert workspace[:8].tobytes() == b"1wlosbyh"


def test_workspace_is_reusable() -> None:
    """One buffer serves repeated decompositions of the same shape."""
    sys, _ = well_conditioned_system(6, 4, seed=3)
    n_threads = 4
    workspace = np.empty(sys.workspace_bytes(n_threads), dtype=np.uint8)

    first = None
    for _ in range(3):
        sys, mat = well_conditioned_system(6, 4, seed=3)
        dec = sys.decompose(n_threads, workspace=workspace)
        x = dec.solve(mat @ np.ones(mat.shape[0]))
        assert residual(mat, x) < 1e-8
        if first is None:
            first = dec.operations()
        else:
            assert dec.operations() == first


def test_workspace_scales_with_threads() -> None:
    """Each thread gets its own scratch block, so more threads need more bytes."""
    sys, _ = well_conditioned_system(6, 4)
    assert sys.workspace_bytes(4) > sys.workspace_bytes(1)
    assert sys.workspace_bytes(0) >= sys.workspace_bytes(1)


def test_workspace_too_small_is_rejected() -> None:
    """An undersized buffer is an error, not a buffer overrun."""
    sys, _ = well_conditioned_system(6, 4)
    needed = sys.workspace_bytes(2)
    with pytest.raises(ValueError, match="at least"):
        sys.decompose(2, workspace=np.empty(needed - 1, dtype=np.uint8))


def test_read_only_workspace_is_rejected() -> None:
    """A read-only buffer is refused rather than silently copied."""
    sys, _ = well_conditioned_system(6, 4)
    workspace = np.empty(sys.workspace_bytes(2), dtype=np.uint8)
    workspace.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        sys.decompose(2, workspace=workspace)


def test_decompose_without_workspace_still_works() -> None:
    """The internal-scratch spelling keeps working."""
    sys, mat = well_conditioned_system(6, 4, seed=4)
    dec = sys.decompose()
    assert residual(mat, dec.solve(mat @ np.ones(mat.shape[0]))) < 1e-8
    assert len(dec.operations()) > 0


def test_workspace_bytes_is_positive_and_aligned() -> None:
    """The size covers the header and is a multiple of the region alignment."""
    sys, _ = well_conditioned_system(6, 4)
    for n_threads in (1, 2, 4, 8):
        size = sys.workspace_bytes(n_threads)
        assert size > 0
        assert size % 8 == 0


@pytest.mark.parametrize("n_threads", (1, 2, 4, 8))
def test_operations_are_deterministic(n_threads: int) -> None:
    """The recorded operations are reproducible, run to run and thread count.

    They are read off the elimination graph's passes, so a parallel
    factorization and a serial one of the same system agree exactly.
    """
    assert operations_are_stable(n_threads)

    baseline = None
    for threads in (1, 2, 4, 8):
        sys_, _ = well_conditioned_system(10, 3, seed=11)
        operations = sys_.decompose(threads).operations()
        if baseline is None:
            baseline = operations
        assert operations == baseline
