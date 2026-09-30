"""Tests of mixed-precision: :class:`hybsol.Precision` and :class:`hybsol.BlockSystem`."""

import numpy as np
import pytest
from hybsol import BlockSystem, Precision

SIZES = (2, 3, 2)


def reference_matrix(sizes: tuple[int, ...], seed: int = 7) -> np.ndarray:
    """Build a symmetric, diagonally dominant dense matrix of the given blocks.

    Parameters
    ----------
    sizes : tuple of int
        Sizes of the diagonal blocks.
    seed : int
        Seed for the random part, so tests are reproducible.

    Returns
    -------
    numpy.ndarray
        A dense ``sum(sizes)`` square matrix that is not singular.
    """
    rng = np.random.default_rng(seed)
    n = sum(sizes)
    matrix = rng.random((n, n))
    return matrix @ matrix.T + n * np.eye(n)


def build(
    precision: Precision, sizes: tuple[int, ...] = SIZES, seed: int = 7
) -> tuple[BlockSystem, np.ndarray]:
    """Assemble ``matrix`` block by block in the given precision.

    Returns
    -------
    tuple of (BlockSystem, numpy.ndarray)
        The system and the dense matrix it represents, in the system's dtype.
    """
    matrix = reference_matrix(sizes, seed)
    offsets = np.pad(np.cumsum(sizes), (1, 0))
    dtype = np.float32 if precision is Precision.SINGLE else np.float64

    system = BlockSystem(*sizes, precision=precision)
    for i in range(len(sizes)):
        for j in range(len(sizes)):
            system.add_block(
                i,
                j,
                matrix[offsets[i] : offsets[i + 1], offsets[j] : offsets[j + 1]].astype(
                    dtype
                ),
            )
    return system, matrix.astype(dtype)


def test_precision_is_a_strenum() -> None:
    """Members are strings, so a plain literal keeps working."""
    assert Precision.DOUBLE == "double"
    assert Precision.SINGLE == "single"
    assert isinstance(Precision.SINGLE, str)


def test_double_is_the_default() -> None:
    """A system built without an argument stores doubles."""
    system = BlockSystem(*SIZES)

    assert system.precision is Precision.DOUBLE
    assert system.as_array().dtype == np.float64


def test_precision_round_trip() -> None:
    """The constructor takes a member or a literal and reports it back."""
    assert BlockSystem(*SIZES, precision=Precision.SINGLE).precision is Precision.SINGLE
    assert BlockSystem(*SIZES, precision="single").precision is Precision.SINGLE
    assert BlockSystem(*SIZES, precision=Precision.DOUBLE).precision is Precision.DOUBLE


@pytest.mark.parametrize("value", ["quad", "", "SINGLE", "float32"])
def test_unknown_precision_is_rejected(value: str) -> None:
    """A typo cannot quietly build a double system."""
    with pytest.raises(ValueError, match="precision must be"):
        BlockSystem(*SIZES, precision=value)


@pytest.mark.parametrize("value", [1, 1.5, None, np.float32(2)])
def test_non_string_precision_is_rejected(value: object) -> None:
    """Anything that is not a string is a TypeError, not a conversion."""
    with pytest.raises(TypeError, match="precision must be"):
        BlockSystem(*SIZES, precision=value)  # type: ignore[arg-type]


def test_arrays_follow_the_system_precision() -> None:
    """Every array handed back is in the system's dtype, whatever it holds."""
    single, matrix = build(Precision.SINGLE)
    double, _ = build(Precision.DOUBLE)

    assert single.as_array().dtype == np.float32
    assert np.array_equal(single.as_array(), matrix)
    assert double.as_array().dtype == np.float64

    single.add_block(0, 0, np.eye(SIZES[0], dtype=np.float32))
    assert single.get_block(0, 0).dtype == np.float32
    assert double.get_block(0, 0).dtype == np.float64


def test_block_storage_view_is_single_for_a_single_system() -> None:
    """In-place storage hands out a writable view in the system's dtype."""
    system = BlockSystem(*SIZES, precision=Precision.SINGLE)
    view = system.block_storage(1, 2)

    assert view.dtype == np.float32
    assert view.shape == (SIZES[1], SIZES[2])
    assert view.flags.writeable
    assert np.all(view == 0)

    view[:] = np.arange(view.size, dtype=np.float32).reshape(view.shape)
    assert np.array_equal(system.get_block(1, 2), view)

    with pytest.raises(RuntimeError, match="block_storage"):
        system.decompose()


def test_values_are_converted_on_the_way_in() -> None:
    """Array-likes are cast to the system's precision when assembled."""
    system = BlockSystem(*SIZES, precision=Precision.SINGLE)
    system.add_block(0, 0, np.eye(SIZES[0]))

    stored = system.get_block(0, 0)
    assert stored.dtype == np.float32
    assert np.array_equal(stored, np.eye(SIZES[0], dtype=np.float32))


def test_classmethods_take_precision() -> None:
    """Both bulk assembly paths honour the argument."""
    matrix = reference_matrix(SIZES)
    offsets = np.pad(np.cumsum(SIZES), (1, 0))
    rows, cols, data = [], [], []
    for i in range(len(SIZES)):
        for j in range(len(SIZES)):
            rows.append(i)
            cols.append(j)
            data.append(
                matrix[offsets[i] : offsets[i + 1], offsets[j] : offsets[j + 1]].ravel()
            )

    from_blocks = BlockSystem.from_blocks(
        SIZES, rows, cols, np.concatenate(data), precision=Precision.SINGLE
    )
    from_list = BlockSystem.from_block_list(
        [
            (
                i,
                j,
                matrix[offsets[i] : offsets[i + 1], offsets[j] : offsets[j + 1]].astype(
                    np.float32
                ),
            )
            for i in range(len(SIZES))
            for j in range(len(SIZES))
        ],
        precision=Precision.SINGLE,
    )

    assert from_blocks.precision is Precision.SINGLE
    assert from_list.precision is Precision.SINGLE
    assert from_blocks.as_array().dtype == np.float32
    assert from_list.as_array().dtype == np.float32


def test_single_solve_is_accurate_to_float() -> None:
    """A single-precision solve lands where ``cond * 1e-7`` says it must."""
    system, matrix = build(Precision.SINGLE)
    system.decompose()

    solution = system.solve(np.ones(sum(SIZES)))
    residual = np.abs(matrix.astype(np.float64) @ solution - 1.0).max()

    assert solution.dtype == np.float64
    bound = np.linalg.cond(matrix.astype(np.float64)) * np.finfo(np.float32).eps
    assert residual < max(bound, 1e-6)


def test_single_and_double_agree_to_float_accuracy() -> None:
    """Both precisions solve the same system to their own accuracy."""
    single, _ = build(Precision.SINGLE)
    double, _ = build(Precision.DOUBLE)
    exact = np.linalg.solve(
        reference_matrix(SIZES).astype(np.float64), np.ones(sum(SIZES))
    )

    single.decompose()
    double.decompose()

    single_error = np.abs(single.solve(np.ones(sum(SIZES))) - exact).max()
    double_error = np.abs(double.solve(np.ones(sum(SIZES))) - exact).max()

    assert double_error < 1e-10
    assert single_error < max(
        np.linalg.cond(reference_matrix(SIZES)) * np.finfo(np.float32).eps, 1e-5
    )


def test_copy_keeps_the_precision() -> None:
    """A copy of a single-precision system is single precision too."""
    system, _ = build(Precision.SINGLE)
    duplicate = system.copy()

    assert duplicate.precision is Precision.SINGLE
    assert duplicate.as_array().dtype == np.float32
