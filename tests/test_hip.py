"""Tests of the HIP backend: :mod:`hybsol.hip`."""

from __future__ import annotations

import hybsol
import numpy as np
import pytest
from hybsol import BlockSystem, DeviceError, Precision

hip = pytest.importorskip("hybsol.hip", reason="built without the HIP backend")

SIZES = (2, 3, 2)
_ALL = hip.devices()
DEVICES = [info for info in _ALL if info["double"]]

requires_device = pytest.mark.skipif(
    not DEVICES, reason="no HIP device that can do double arithmetic"
)


def reference_matrix(sizes: tuple[int, ...], seed: int = 11) -> np.ndarray:
    """Build a symmetric, diagonally dominant dense matrix of the given blocks."""
    rng = np.random.default_rng(seed)
    n = sum(sizes)
    matrix = rng.standard_normal((n, n))
    matrix = matrix + matrix.T
    matrix[np.diag_indices(n)] += n
    return matrix


def build(
    precision: Precision = Precision.DOUBLE,
    sizes: tuple[int, ...] = SIZES,
) -> tuple[BlockSystem, np.ndarray]:
    """Assemble the reference matrix block by block in the given precision."""
    matrix = reference_matrix(sizes)
    dtype = np.float32 if precision is Precision.SINGLE else np.float64
    system = BlockSystem(*sizes, precision=precision)
    offsets = np.cumsum((0, *sizes))
    for i in range(len(sizes)):
        for j in range(len(sizes)):
            block = matrix[offsets[i] : offsets[i + 1], offsets[j] : offsets[j + 1]]
            system.add_block(i, j, block.astype(dtype))
    return system, matrix.astype(dtype)


def first_device() -> int:
    """Index of the first device that can take a decomposition."""
    for info in _ALL:
        if info["double"]:
            return int(info["index"])
    raise pytest.skip("no HIP device that can do double arithmetic")


def test_importing_hybsol_does_not_load_the_backend() -> None:
    """The opt-in is real: importing the package loads no ROCm."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, hybsol; "
            "print(any(m.startswith('hybsol._mod_hip') for m in sys.modules), "
            "'hybsol.hip' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False False"


def test_devices_are_listed_with_their_facts() -> None:
    """Each device reports a name, a kind, a uuid and whether it can do doubles."""
    for info in hip.devices():
        assert set(info) >= {
            "index",
            "name",
            "vendor",
            "uuid",
            "kind",
            "double",
            "memory",
            "compute_units",
        }
        assert info["name"]
        assert info["kind"] in {"gpu", "integrated", "unknown"}
        assert isinstance(info["double"], bool)
        assert info["uuid"]


def test_a_discrete_gpu_comes_first() -> None:
    """Wherever the machine has a discrete GPU, it sits at index 0."""
    found = hip.devices()
    if any(info["kind"] == "gpu" for info in found):
        assert found[0]["kind"] == "gpu"


def test_device_index_past_the_end_is_an_error() -> None:
    """An index nothing has is an error, not an empty answer."""
    with pytest.raises(DeviceError):
        hip.device_info(len(hip.devices()) + 100)


@requires_device
def test_device_decomposition_solves_like_the_cpu() -> None:
    """A device solve agrees with the CPU's to its own precision."""
    system, matrix = build()
    rhs = np.ones(sum(SIZES))

    cpu = system.decompose().solve(rhs)
    gpu = hip.decompose(system, device=first_device()).solve(rhs)

    assert gpu.dtype == np.float64
    np.testing.assert_allclose(gpu, cpu, rtol=1e-9, atol=1e-12)


@requires_device
@pytest.mark.parametrize("precision", [Precision.DOUBLE, Precision.SINGLE])
def test_device_factorization_honours_the_precision(precision: Precision) -> None:
    """Both factor precisions work on a device, each to its own accuracy."""
    system, matrix = build()
    rhs = np.ones(sum(SIZES))

    dec = hip.decompose(system, device=first_device(), precision=precision)
    solution = dec.solve(rhs)
    residual = np.abs(matrix.astype(np.float64) @ solution - 1.0).max()

    eps = (
        np.finfo(np.float32).eps
        if precision is Precision.SINGLE
        else np.finfo(np.float64).eps
    )
    assert residual < max(np.linalg.cond(matrix.astype(np.float64)) * eps, 1e-6)


@requires_device
def test_a_double_system_can_produce_single_factors_on_a_device() -> None:
    """The mixed-precision path runs on the device too."""
    system, matrix = build()
    rhs = np.ones(sum(SIZES))

    dec = hip.decompose(system, device=first_device(), precision=Precision.SINGLE)
    solution = dec.solve(rhs)
    residual = np.abs(matrix @ solution - 1.0).max()

    bound = np.linalg.cond(matrix) * np.finfo(np.float32).eps
    assert residual < max(bound, 1e-5)


@requires_device
def test_device_decomposition_reports_its_device() -> None:
    """A device decomposition names its device; a CPU one names none."""
    system, _ = build()
    device = first_device()

    assert system.decompose().device is None
    assert hip.decompose(system, device=device).device == device


@requires_device
def test_device_decomposition_matches_the_cpu_operation_list() -> None:
    """The replay path works on a device: same operations, same schedule."""
    system, _ = build()
    gpu = hip.decompose(system, device=first_device())
    cpu = system.decompose()

    assert gpu.n_operations == cpu.n_operations
    assert gpu.operations() == cpu.operations()


@requires_device
def test_refined_solve_works_on_a_device_decomposition() -> None:
    """refined_solve measures against the system, so a device path serves it."""
    system, matrix = build()
    rhs = np.ones(sum(SIZES))
    dec = hip.decompose(system, device=first_device())

    solution = hybsol.refined_solve(system, dec, rhs)
    exact = np.linalg.solve(matrix, rhs)
    np.testing.assert_allclose(solution, exact, rtol=1e-8, atol=1e-12)


@requires_device
def test_arith_native_is_an_explicit_name_for_the_default() -> None:
    """``arith="native"`` asks for the mode ``auto`` resolves to today."""
    system, matrix = build()
    rhs = np.ones(sum(SIZES))

    dec = hip.decompose(system, device=first_device(), arith="native")
    solution = dec.solve(rhs)
    residual = np.abs(matrix.astype(np.float64) @ solution - 1.0).max()

    eps = np.finfo(np.float64).eps
    assert residual < max(np.linalg.cond(matrix.astype(np.float64)) * eps, 1e-6)


@requires_device
def test_split_arith_solves_at_native_accuracy() -> None:
    """``arith="split"`` carries the double factors in double-single pairs.

    Upgraded from the refusal that guarded this mode before the double-single
    kernels landed: the same seeded system now solves inside the
    ``cond * 1e-10`` gate, beats the single-precision factors of that system
    by 1e3, and agrees with the native solve to native tolerance.
    """
    system, matrix = build()
    rhs = np.ones(sum(SIZES))
    device = first_device()

    native = hip.decompose(system, device=device, arith="native").solve(rhs)
    split = hip.decompose(system, device=device, arith="split").solve(rhs)
    single = hip.decompose(system, device=device, precision=Precision.SINGLE).solve(rhs)

    split_residual = np.abs(matrix @ split - 1.0).max()
    single_residual = np.abs(matrix @ single - 1.0).max()

    assert split_residual < max(np.linalg.cond(matrix) * 1e-10, 1e-8)
    assert split_residual * 1e3 < single_residual
    np.testing.assert_allclose(split, native, rtol=1e-9, atol=1e-12)


@requires_device
def test_ozaki_arith_solves_at_native_accuracy() -> None:
    """``arith="ozaki"`` runs the block GEMMs on the WMMA matrix cores.

    Sized so the tiles apply (a 16-wide block at the gate), gated exactly
    like the split parity case: residual inside ``cond * 1e-10``, and the
    solve agrees with the native one to native tolerance.
    """
    sizes = (16, 16, 8)
    system, matrix = build(precision=Precision.DOUBLE, sizes=sizes)
    rhs = np.ones(sum(sizes))
    device = first_device()

    try:
        ozaki = hip.decompose(system, device=device, arith="ozaki").solve(rhs)
    except DeviceError:
        pytest.skip("no gfx12 device for the Ozaki modes")

    native = hip.decompose(system, device=device, arith="native").solve(rhs)
    ozaki_residual = np.abs(matrix @ ozaki - 1.0).max()
    assert ozaki_residual < max(np.linalg.cond(matrix) * 1e-10, 1e-8)
    np.testing.assert_allclose(ozaki, native, rtol=1e-9, atol=1e-12)


@requires_device
def test_ozaki2_arith_runs_as_a_refinement_partner() -> None:
    """``arith="ozaki2"`` runs and solves with no accuracy gate on purpose.

    Asymmetric slicing keeps only ~11 bits on the unsliced side, so the
    raw solve is refinement-partner grade; the contract is that
    :func:`hybsol.refined_solve` on top of it lands the answer.
    """
    sizes = (16, 16, 8)
    system, matrix = build(precision=Precision.DOUBLE, sizes=sizes)
    rhs = np.ones(sum(sizes))
    device = first_device()

    try:
        dec = hip.decompose(system, device=device, arith="ozaki2")
        solution = dec.solve(rhs)
    except DeviceError:
        pytest.skip("no gfx12 device for the Ozaki modes")

    assert solution.shape == rhs.shape
    assert np.all(np.isfinite(solution))

    refined = hybsol.refined_solve(system, dec, rhs, tolerance=1e-8)
    exact = np.linalg.solve(matrix, rhs)
    np.testing.assert_allclose(refined, exact, rtol=1e-6, atol=1e-9)


@requires_device
def test_an_unknown_arith_name_is_rejected() -> None:
    """A misspelled mode is a mistake in the call, listed with the valid names."""
    system, _ = build()
    with pytest.raises(ValueError, match='must be one of "auto"'):
        hip.decompose(system, device=first_device(), arith="fast")  # type: ignore[arg-type]


@requires_device
def test_a_singular_diagonal_is_reported_the_same_way() -> None:
    """A device factorization names the singular block it failed on."""
    system = BlockSystem(2)
    system.add_block(0, 0, [[1.0, 2.0], [2.0, 4.0]])  # Rank one: no second pivot.

    with pytest.raises(hybsol.SingularSystemError, match="block 0"):
        hip.decompose(system, device=first_device())


@requires_device
def test_a_device_index_past_the_end_raises() -> None:
    """Asking for a device that is not there is a DeviceError."""
    system, _ = build()
    with pytest.raises(DeviceError):
        hip.decompose(system, device=len(hip.devices()) + 100)


@requires_device
def test_a_negative_device_index_is_a_value_error() -> None:
    """A negative index is a mistake in the call, not a missing device."""
    system, _ = build()
    with pytest.raises(ValueError, match="non-negative"):
        hip.decompose(system, device=-1)


@requires_device
def test_an_invalid_system_is_refused_before_the_device() -> None:
    """A system the walk would reject never reaches a kernel."""
    system = BlockSystem(2, 2)
    system.add_block(0, 0, [[2.0, 0.0], [0.0, 2.0]])
    system.add_block(1, 1, [[2.0, 0.0], [0.0, 2.0]])
    system.add_block(1, 0, [[1.0, 0.0], [0.0, 1.0]])  # No mirror above it.
    with pytest.raises(ValueError):
        hip.decompose(system, device=first_device())


def test_decompose_needs_a_block_system() -> None:
    """Anything else is a TypeError, before any device work."""
    with pytest.raises(TypeError, match="BlockSystem"):
        hip.decompose("not a system")  # type: ignore[arg-type]


@requires_device
def test_an_unknown_precision_is_rejected() -> None:
    """The precision argument is checked the same way the core checks it."""
    system, _ = build()
    with pytest.raises((ValueError, TypeError), match="precision must be"):
        hip.decompose(system, precision="quad")  # type: ignore[arg-type]


@requires_device
def test_two_device_decompositions_coexist() -> None:
    """Each decomposition owns its own device memory."""
    system, matrix = build()
    rhs = np.ones(sum(SIZES))

    first = hip.decompose(system, device=first_device())
    second = hip.decompose(system, device=first_device())
    np.testing.assert_allclose(
        first.solve(rhs), second.solve(rhs), rtol=1e-12, atol=1e-14
    )

    exact = np.linalg.solve(matrix, rhs)
    np.testing.assert_allclose(first.solve(rhs), exact, rtol=1e-8, atol=1e-12)


# ---------------------------------------------------------------------------
# The shapes and magnitudes the fast paths are particular about
# ---------------------------------------------------------------------------

_ARITH = ("auto", "native", "split", "ozaki", "ozaki2")


def _from_blocks(
    sizes: tuple[int, ...], entries: dict[tuple[int, int], np.ndarray]
) -> tuple[BlockSystem, np.ndarray]:
    """Store dense blocks in a system of ``sizes``; return it and its matrix."""
    system = BlockSystem(*sizes)
    offsets = np.cumsum((0, *sizes))
    matrix = np.zeros((int(offsets[-1]), int(offsets[-1])))
    for (row, col), block in entries.items():
        array = np.asarray(block, dtype=np.float64)
        system.add_block(row, col, array)
        matrix[offsets[row] : offsets[row + 1], offsets[col] : offsets[col + 1]] = array
    return system, matrix


def _dominated(size: int, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    """Build a diagonally dominant block with every entry scaled by ``scale``."""
    block = rng.standard_normal((size, size)) * scale
    block[np.diag_indices(size)] += size * scale
    return block


def _decompose_or_skip(system: BlockSystem, device: int, arith: str, **kwargs: object):
    """Decompose on ``device``; a mode the device lacks skips the test."""
    try:
        return hip.decompose(system, device=device, arith=arith, **kwargs)
    except DeviceError as exc:
        if "capability" in str(exc):
            pytest.skip(f"{arith} needs a device this machine has not got: {exc}")
        raise


def _relative_residual(
    matrix: np.ndarray, solution: np.ndarray, rhs: np.ndarray
) -> float:
    """How far the solve is from reproducing its right-hand side."""
    return float(np.abs(matrix @ solution - rhs).max() / np.abs(rhs).max())


@requires_device
@pytest.mark.parametrize("arith", _ARITH)
def test_stored_zero_blocks_are_walked_like_any_other(arith: str) -> None:
    """A coupling stored as zeros is still a stored block the walk visits.

    A system whose only off-diagonal blocks are exact zeros asks for no
    elimination at all -- and the scaling pass still runs over those blocks.
    It once did so past the end of its scratch, and with no scratch at all,
    which the GPU answered with a fault rather than an answer.
    """
    rng = np.random.default_rng(11)
    zero = np.zeros((16, 16))
    system, matrix = _from_blocks(
        (16, 16),
        {
            (0, 0): _dominated(16, rng),
            (1, 1): _dominated(16, rng),
            (0, 1): zero,
            (1, 0): zero,
        },
    )
    rhs = np.ones(32)

    solution = _decompose_or_skip(system, first_device(), arith).solve(rhs)
    np.testing.assert_allclose(
        solution, np.linalg.solve(matrix, rhs), rtol=1e-9, atol=1e-12
    )


@requires_device
@pytest.mark.parametrize("spokes", [2, 6])
@pytest.mark.parametrize("arith", ("auto", "split", "ozaki"))
def test_a_hub_of_stored_blocks_solves_like_the_cpu(spokes: int, arith: str) -> None:
    """One row feeding many: scratch is sized for a row's sources, not its blocks.

    The hub row owns ``spokes`` couplings, so a work item per (row, source)
    needs that many scratch slots; sizing them from the block count alone
    made the later spokes write over -- and past -- the earlier ones.
    """
    rng = np.random.default_rng(7)
    sizes = (16,) * (spokes + 1)
    entries: dict[tuple[int, int], np.ndarray] = {
        (i, i): _dominated(16, rng) for i in range(spokes + 1)
    }
    for spoke in range(1, spokes + 1):
        link = rng.standard_normal((16, 16))
        entries[(0, spoke)] = link
        entries[(spoke, 0)] = link.T
    system, matrix = _from_blocks(sizes, entries)
    rhs = np.ones(sum(sizes))

    solution = _decompose_or_skip(system, first_device(), arith).solve(rhs)
    np.testing.assert_allclose(
        solution, np.linalg.solve(matrix, rhs), rtol=1e-9, atol=1e-12
    )


@requires_device
@pytest.mark.parametrize("arith", ("auto", "split", "ozaki"))
def test_a_multi_pass_singular_block_is_named_by_the_device(arith: str) -> None:
    """The block the device blames is the block the CPU blames.

    The rank-one block sits in the middle, so its second pivot fails partway
    through a pass that covers several blocks: the pass reports the failing
    column by where it sits in memory, not by which block it came from, and
    blamed the row before the truth.
    """
    entries: dict[tuple[int, int], np.ndarray] = {
        (0, 0): np.eye(2) * 3.0,
        (1, 1): np.array([[1.0, 2.0], [2.0, 4.0]]),  # Rank one: no second pivot.
        (2, 2): np.eye(2) * 3.0,
        (3, 3): np.eye(2) * 3.0,
    }
    for row, col in ((0, 1), (1, 0), (2, 3), (3, 2)):
        entries[(row, col)] = np.zeros((2, 2))
    system, _ = _from_blocks((2, 2, 2, 2), entries)

    with pytest.raises(hybsol.SingularSystemError) as cpu:
        system.decompose()
    with pytest.raises(hybsol.SingularSystemError) as device:
        _decompose_or_skip(system, first_device(), arith)

    assert str(device.value) == str(cpu.value)
    assert str(cpu.value).endswith("(block 1)")


@requires_device
@pytest.mark.parametrize("arith", ("ozaki", "ozaki2"))
def test_a_mode_the_device_cannot_do_names_the_capability_it_lacks(arith: str) -> None:
    """Where the matrix cores are not there, the refusal says which ones are."""
    rng = np.random.default_rng(3)
    system, _ = _from_blocks(
        (16, 16),
        {
            (0, 0): _dominated(16, rng),
            (1, 1): _dominated(16, rng),
            (0, 1): rng.standard_normal((16, 16)),
            (1, 0): rng.standard_normal((16, 16)),
        },
    )

    refusals = 0
    for info in _ALL:
        if not info["double"]:
            continue
        try:
            hip.decompose(system, device=info["index"], arith=arith)
        except DeviceError as exc:
            refusals += 1
            assert "WMMA" in str(exc), str(exc)
            assert "gfx12" in str(exc), str(exc)
            assert str(info["index"]) in str(exc), str(exc)
    if refusals == len(DEVICES):
        pytest.skip("no device here has the WMMA matrix cores")


@requires_device
@pytest.mark.parametrize("arith", _ARITH)
def test_single_factors_ignore_the_arith_on_every_device(arith: str) -> None:
    """``arith`` chooses between the double implementations.

    Single factors are already the narrow ones -- the matrix cores hold no
    double -- so the mode answers with the same kernels on every device
    rather than a capability refusal. That is a choice, and this is the test
    that keeps it one: a refusal here would be a device-dependent answer for
    the same call.
    """
    sizes = (16, 16)
    system, matrix = build(precision=Precision.SINGLE, sizes=sizes)
    rhs = np.ones(sum(sizes))
    eps = np.finfo(np.float32).eps

    for info in _ALL:
        if not info["double"]:
            continue
        solution = hip.decompose(
            system, device=info["index"], arith=arith, precision=Precision.SINGLE
        ).solve(rhs)
        assert np.all(np.isfinite(solution))
        residual = _relative_residual(matrix, solution, rhs)
        assert residual < max(np.linalg.cond(matrix) * eps, 1e-6)


@requires_device
def test_a_block_under_the_tile_width_falls_back_quietly() -> None:
    """Narrow blocks never reach the matrix cores, and lose nothing by that.

    Asking for ``ozaki`` on a system of 8-wide blocks is not an error and not
    a slower answer: the tiles are 16 wide, so the plain walk serves every
    block, at the accuracy the plain walk gives.
    """
    sizes = (8, 8, 8)
    system, _ = build(precision=Precision.DOUBLE, sizes=sizes)
    rhs = np.ones(sum(sizes))

    ozaki = _decompose_or_skip(system, first_device(), "ozaki").solve(rhs)
    native = hip.decompose(system, device=first_device(), arith="native").solve(rhs)
    np.testing.assert_allclose(ozaki, native, rtol=1e-9, atol=1e-12)


@requires_device
def test_a_block_past_the_tile_width_falls_back_quietly() -> None:
    """Fall back at the other end of the width range, just as quietly.

    A 33-wide block is wider than the tile walk spans, so it gets the plain
    walk too -- which is why asking for the matrix cores is safe whether or
    not the system is shaped for them.
    """
    sizes = (33, 33)
    system, matrix = build(precision=Precision.DOUBLE, sizes=sizes)
    rhs = np.ones(sum(sizes))

    ozaki = _decompose_or_skip(system, first_device(), "ozaki").solve(rhs)
    np.testing.assert_allclose(ozaki, np.linalg.solve(matrix, rhs), rtol=1e-9, atol=1e-12)


@requires_device
@pytest.mark.parametrize("scale", [1e-300, 1e-40, 1e-30, 1e20, 1e300])
@pytest.mark.parametrize("arith", ("split", "ozaki"))
def test_a_system_outside_a_fast_path_window_still_solves(
    scale: float, arith: str
) -> None:
    """Leaving a fast path is a switch of arithmetic, never a wrong answer.

    The double-single pairs and the Ozaki slices carry magnitudes over a
    range narrower than a double's, so both modes ask per block whether the
    block fits and hand the rest to the plain walk. A 20-wide system of
    1e-300 entries -- wide enough to be walked as 16-wide panels, which is
    where the slicing scale used to be taken from the wrong half of the
    block -- answers as well as an order-one one.
    """
    rng = np.random.default_rng(5)
    sizes = (20, 20)
    system, matrix = _from_blocks(
        sizes,
        {
            (0, 0): _dominated(20, rng, scale),
            (1, 1): _dominated(20, rng, scale),
            (0, 1): rng.standard_normal((20, 20)) * scale,
            (1, 0): rng.standard_normal((20, 20)) * scale,
        },
    )
    rhs = np.ones(sum(sizes))

    solution = _decompose_or_skip(system, first_device(), arith).solve(rhs)
    assert np.all(np.isfinite(solution))
    bound = max(np.linalg.cond(matrix) * 1e-10, 1e-8)
    assert _relative_residual(matrix, solution, rhs) < bound


@requires_device
@pytest.mark.parametrize("arith", ("split", "ozaki"))
def test_a_block_outside_the_window_moves_only_itself(arith: str) -> None:
    """One huge block among order-one blocks: the guards answer per block.

    The wide coupling is 1e31, the rest of the system is ordinary. The mode
    has to leave that block to the plain walk and keep the fast arithmetic
    for the others, rather than degrade the whole factorization -- or lose
    the one block it decided to carry in the narrow form.
    """
    rng = np.random.default_rng(11)
    sizes = (8, 8, 8)
    wide = rng.standard_normal((8, 8)) * 1e31
    middle = rng.standard_normal((8, 8)) * 1e5
    system, matrix = _from_blocks(
        sizes,
        {
            (0, 0): _dominated(8, rng),
            (1, 1): _dominated(8, rng),
            (2, 2): _dominated(8, rng),
            (0, 1): wide,
            (1, 0): wide.T,
            (1, 2): middle,
            (2, 1): middle.T,
        },
    )
    rhs = np.ones(sum(sizes))

    solution = _decompose_or_skip(system, first_device(), arith).solve(rhs)
    exact = np.linalg.solve(matrix, rhs)
    scale = float(np.abs(exact).max())
    assert np.abs(solution - exact).max() / scale < 1e-9


@requires_device
@pytest.mark.parametrize("arith", _ARITH)
def test_a_live_writable_view_does_not_stop_a_decomposition(arith: str) -> None:
    """A view the caller still holds does not block a copy.

    Only the operations that move the system's own storage -- a reorder, an
    elimination -- refuse while a :meth:`BlockSystem.block_storage` view is
    alive. A device decomposition copies the blocks, so the view stays what
    it was and the factorization goes ahead; that is the documented
    behaviour, and this keeps it documented behaviour.
    """
    sizes = (16, 16)
    system, matrix = build(precision=Precision.DOUBLE, sizes=sizes)
    view = system.block_storage(0, 1)
    rhs = np.ones(sum(sizes))

    dec = _decompose_or_skip(system, first_device(), arith)
    solution = dec.solve(rhs)
    assert np.all(np.isfinite(solution))
    if arith == "ozaki2":
        # Eleven-bit grade by design: the mode is a refinement partner, and
        # this test is about the view, not about the arithmetic.
        assert _relative_residual(matrix, solution, rhs) < 1e-2
    else:
        np.testing.assert_allclose(
            solution, np.linalg.solve(matrix, rhs), rtol=1e-9, atol=1e-12
        )
    assert view[0, 0] == matrix[0, 16]


@requires_device
def test_the_device_copy_does_not_follow_later_edits() -> None:
    """Writing through a view after the copy leaves the factors alone."""
    sizes = (8, 8)
    system, matrix = build(precision=Precision.DOUBLE, sizes=sizes)
    rhs = np.ones(sum(sizes))

    dec = hip.decompose(system, device=first_device())
    before = dec.solve(rhs)
    view = system.block_storage(0, 0)
    view[np.diag_indices(8)] += 100.0

    np.testing.assert_allclose(dec.solve(rhs), before, rtol=1e-12, atol=1e-14)
    assert _relative_residual(matrix, dec.solve(rhs), rhs) < 1e-9


@requires_device
@pytest.mark.parametrize("value", [1.5, "0", None, 2**63])
def test_a_device_that_is_not_an_index_is_a_type_error(value: object) -> None:
    """A float, a string or nothing at all is the caller's mistake, early."""
    system, _ = build()
    with pytest.raises(TypeError):
        hip.decompose(system, device=value)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        hip.device_info(value)  # type: ignore[arg-type]


@requires_device
def test_a_device_index_far_out_of_range_says_so_itself() -> None:
    """The detail a DeviceError carries belongs to the call that raised.

    A refusal with something to explain -- an Ozaki mode on a device without
    the matrix cores -- leaves that explanation behind, and the very next
    failure used to quote it: asking for a device nobody has was answered
    with a complaint about the device from the earlier call.
    """
    rng = np.random.default_rng(3)
    system, _ = _from_blocks(
        (16, 16),
        {
            (0, 0): _dominated(16, rng),
            (1, 1): _dominated(16, rng),
            (0, 1): rng.standard_normal((16, 16)),
            (1, 0): rng.standard_normal((16, 16)),
        },
    )

    earlier = None
    for info in _ALL:
        if not info["double"]:
            continue
        try:
            hip.decompose(system, device=info["index"], arith="ozaki")
        except DeviceError as exc:
            earlier = str(exc)
            break
    if earlier is None:
        pytest.skip("every device here has the WMMA matrix cores")

    with pytest.raises(DeviceError) as missing:
        hip.decompose(system, device=10**12)
    message = str(missing.value)
    assert earlier not in message
    assert "1000000000000" in message


def test_the_device_list_is_its_own_indices_and_stable() -> None:
    """Asking twice answers the same, and index ``i`` is position ``i``."""
    again = hip.devices()
    assert [info["index"] for info in again] == list(range(len(again)))
    assert again == _ALL
    if _ALL:
        kinds = [info["kind"] for info in again]
        if "gpu" in kinds:
            assert kinds[0] == "gpu"


@requires_device
def test_arith_none_names_the_default_mode() -> None:
    """``arith=None`` is the default, not a fifth mode."""
    system, _ = build(sizes=(8, 8))
    rhs = np.ones(16)
    default = hip.decompose(system, device=first_device(), arith=None)
    auto = hip.decompose(system, device=first_device(), arith="auto")
    np.testing.assert_allclose(
        default.solve(rhs), auto.solve(rhs), rtol=1e-12, atol=1e-14
    )
