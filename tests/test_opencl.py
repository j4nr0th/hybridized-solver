"""Tests of the OpenCL backend: :mod:`hybsol.opencl`."""

from __future__ import annotations

import hybsol
import numpy as np
import pytest
from hybsol import BlockSystem, DeviceError, Precision

opencl = pytest.importorskip("hybsol.opencl", reason="built without the OpenCL backend")

SIZES = (2, 3, 2)
_ALL = opencl.devices()
DEVICES = [info for info in _ALL if info["double"]]

requires_device = pytest.mark.skipif(
    not DEVICES, reason="no OpenCL device that can do double arithmetic"
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
    raise pytest.skip("no OpenCL device that can do double arithmetic")


def test_importing_hybsol_does_not_load_the_backend() -> None:
    """The opt-in is real: importing the package loads no OpenCL."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, hybsol; "
            "print(any(m.startswith('hybsol._mod_opencl') for m in sys.modules), "
            "'hybsol.opencl' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False False"


def test_devices_are_listed_with_their_facts() -> None:
    """Each device reports a name, a kind and whether it can do doubles."""
    for info in opencl.devices():
        assert set(info) >= {
            "index",
            "name",
            "vendor",
            "kind",
            "double",
            "memory",
            "compute_units",
        }
        assert info["name"]
        assert info["kind"] in {"gpu", "cpu", "accelerator", "unknown"}
        assert isinstance(info["double"], bool)


def test_device_index_past_the_end_is_an_error() -> None:
    """An index nothing has is an error, not an empty answer."""
    with pytest.raises(DeviceError):
        opencl.device_info(len(opencl.devices()) + 100)


@requires_device
def test_device_decomposition_solves_like_the_cpu() -> None:
    """A device solve agrees with the CPU's to its own precision."""
    system, matrix = build()
    rhs = np.ones(sum(SIZES))

    cpu = system.decompose().solve(rhs)
    gpu = opencl.decompose(system, device=first_device()).solve(rhs)

    assert gpu.dtype == np.float64
    np.testing.assert_allclose(gpu, cpu, rtol=1e-9, atol=1e-12)


@requires_device
@pytest.mark.parametrize("precision", [Precision.DOUBLE, Precision.SINGLE])
def test_device_factorization_honours_the_precision(precision: Precision) -> None:
    """Both factor precisions work on a device, each to its own accuracy."""
    system, matrix = build()
    rhs = np.ones(sum(SIZES))

    dec = opencl.decompose(system, device=first_device(), precision=precision)
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

    dec = opencl.decompose(system, device=first_device(), precision=Precision.SINGLE)
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
    assert opencl.decompose(system, device=device).device == device


@requires_device
def test_device_decomposition_matches_the_cpu_operation_list() -> None:
    """The replay path works on a device: same operations, same schedule."""
    system, _ = build()
    gpu = opencl.decompose(system, device=first_device())
    cpu = system.decompose()

    assert gpu.n_operations == cpu.n_operations
    assert gpu.operations() == cpu.operations()


@requires_device
def test_refined_solve_works_on_a_device_decomposition() -> None:
    """refined_solve measures against the system, so a device path serves it."""
    system, matrix = build()
    rhs = np.ones(sum(SIZES))
    dec = opencl.decompose(system, device=first_device())

    solution = hybsol.refined_solve(system, dec, rhs)
    exact = np.linalg.solve(matrix, rhs)
    np.testing.assert_allclose(solution, exact, rtol=1e-8, atol=1e-12)


@requires_device
def test_a_singular_diagonal_is_reported_the_same_way() -> None:
    """A device factorization names the singular block it failed on."""
    system = BlockSystem(2)
    system.add_block(0, 0, [[1.0, 2.0], [2.0, 4.0]])  # Rank one: no second pivot.

    with pytest.raises(hybsol.SingularSystemError, match="block 0"):
        opencl.decompose(system, device=first_device())


@requires_device
def test_a_device_index_past_the_end_raises() -> None:
    """Asking for a device that is not there is a DeviceError."""
    system, _ = build()
    with pytest.raises(DeviceError):
        opencl.decompose(system, device=len(opencl.devices()) + 100)


@requires_device
def test_a_negative_device_index_is_a_value_error() -> None:
    """A negative index is a mistake in the call, not a missing device."""
    system, _ = build()
    with pytest.raises(ValueError, match="non-negative"):
        opencl.decompose(system, device=-1)


@requires_device
def test_an_invalid_system_is_refused_before_the_device() -> None:
    """A system the walk would reject never reaches a kernel."""
    system = BlockSystem(2, 2)
    system.add_block(0, 0, [[2.0, 0.0], [0.0, 2.0]])
    system.add_block(1, 1, [[2.0, 0.0], [0.0, 2.0]])
    system.add_block(1, 0, [[1.0, 0.0], [0.0, 1.0]])  # No mirror above it.
    with pytest.raises(ValueError):
        opencl.decompose(system, device=first_device())


def test_decompose_needs_a_block_system() -> None:
    """Anything else is a TypeError, before any device work."""
    with pytest.raises(TypeError, match="BlockSystem"):
        opencl.decompose("not a system")  # type: ignore[arg-type]


@requires_device
def test_an_unknown_precision_is_rejected() -> None:
    """The precision argument is checked the same way the core checks it."""
    system, _ = build()
    with pytest.raises((ValueError, TypeError), match="precision must be"):
        opencl.decompose(system, precision="quad")  # type: ignore[arg-type]


@requires_device
def test_two_device_decompositions_coexist() -> None:
    """Each decomposition owns its own device memory."""
    system, matrix = build()
    rhs = np.ones(sum(SIZES))

    first = opencl.decompose(system, device=first_device())
    second = opencl.decompose(system, device=first_device())
    np.testing.assert_allclose(
        first.solve(rhs), second.solve(rhs), rtol=1e-12, atol=1e-14
    )

    exact = np.linalg.solve(matrix, rhs)
    np.testing.assert_allclose(first.solve(rhs), exact, rtol=1e-8, atol=1e-12)
