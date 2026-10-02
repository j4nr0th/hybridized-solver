"""Tests for the in-place block storage view returned by ``block_storage()``."""

import gc

import numpy as np
import pytest
from hybsol import BlockSystem


def two_block_system() -> BlockSystem:
    """Build a decomposable 2 x 2 block system with blocks of size 3.

    Returns
    -------
    BlockSystem
        A system with both diagonal blocks and both off-diagonal blocks.
    """
    sys = BlockSystem(3, 3)
    sys.add_block(0, 0, np.eye(3) * 4.0)
    sys.add_block(0, 1, np.ones((3, 3)))
    sys.add_block(1, 0, np.ones((3, 3)))
    sys.add_block(1, 1, np.eye(3) * 4.0)
    return sys


def test_block_storage_is_a_zero_filled_view() -> None:
    """The first call creates a zeroed, writable, C-contiguous view."""
    sys = BlockSystem(2, 3, 2)

    assert not sys.has_block(0, 1)
    view = sys.block_storage(0, 1)

    assert view.shape == (2, 3)
    assert view.dtype == np.float64
    assert view.flags.writeable
    assert view.flags.c_contiguous
    assert np.all(view == 0.0)
    assert sys.has_block(0, 1)


def test_block_storage_writes_land_in_the_system() -> None:
    """Values written through the view are what ``as_array`` reports."""
    sys = BlockSystem(2, 2)
    view = sys.block_storage(0, 1)
    expected = np.arange(4.0).reshape(2, 2)

    view[:] = expected

    assert np.array_equal(sys.get_block(0, 1), expected)
    assert np.all(sys.as_array()[:2, 2:] == expected)


def test_block_storage_returns_the_same_buffer() -> None:
    """Re-fetching gives fresh array objects over untouched storage."""
    sys = BlockSystem(2, 2)
    first = sys.block_storage(0, 1)
    first[:] = 5.0

    second = sys.block_storage(0, 1)

    assert second is not first
    assert np.shares_memory(second, first)
    assert np.all(second == 5.0)
    assert np.all(first == 5.0)


def test_block_storage_survives_later_additions() -> None:
    """Adding blocks must not relocate storage handed out earlier."""
    sys = BlockSystem(2, 2, 2)
    pinned = sys.block_storage(0, 0)
    aliased = sys.block_storage(0, 2)
    addr = pinned.ctypes.data

    sys.add_block(0, 1, np.ones((2, 2)))
    sys.add_block(1, 0, np.ones((2, 2)))

    assert pinned.ctypes.data == addr
    assert np.shares_memory(aliased, sys.block_storage(0, 2))
    assert np.all(pinned == 0.0)


def test_block_storage_fills_a_block_that_was_left_sparse() -> None:
    """A block created in place is a normal member of the pattern."""
    sys = BlockSystem(2, 2)
    sys.add_block(0, 0, np.eye(2))
    sys.add_block(1, 0, np.eye(2))
    sys.add_block(1, 1, np.eye(2))

    sys.block_storage(0, 1)[:] = np.array([[7.0, 8.0], [9.0, 10.0]])

    assert sys.is_valid()
    expected = np.zeros((4, 4))
    expected[:2, :2] = np.eye(2)
    expected[:2, 2:] = [[7.0, 8.0], [9.0, 10.0]]
    expected[2:, :2] = np.eye(2)
    expected[2:, 2:] = np.eye(2)
    assert np.array_equal(sys.as_array(), expected)


@pytest.mark.parametrize(
    ("row", "col", "error"),
    [
        (-1, 0, ValueError),
        (0, -1, ValueError),
        (2, 0, ValueError),
        (0, 5, ValueError),
    ],
)
def test_block_storage_rejects_bad_indices(
    row: int, col: int, error: type[Exception]
) -> None:
    """Out-of-range indices are reported before anything is allocated."""
    sys = BlockSystem(2, 3)

    with pytest.raises(error):
        sys.block_storage(row, col)


def test_block_storage_survives_a_decomposition() -> None:
    """A factorization writes elsewhere, so the system keeps serving storage."""
    sys = two_block_system()
    sys.decompose()

    view = sys.block_storage(0, 1)
    view[:] = 2.5
    assert np.all(sys.get_block(0, 1) == 2.5)


def test_decomposition_does_not_observe_a_live_view() -> None:
    """The factorization copies what it needs, so a held view is not in the way."""
    sys = two_block_system()
    held = sys.block_storage(0, 1)
    before = sys.as_array().copy()

    dec = sys.decompose()

    assert np.all(sys.as_array() == before)
    assert np.all(held == 1.0)
    assert dec.total_size == before.shape[0]


def test_live_view_blocks_restructuring() -> None:
    """Methods that rebuild a row refuse while a view may still be written to."""
    sys = two_block_system()
    held = sys.block_storage(0, 1)

    with pytest.raises(RuntimeError, match="block_storage"):
        sys.reorder_blocks(np.array([1, 0], dtype=np.uint64))
    with pytest.raises(RuntimeError, match="block_storage"):
        sys.eliminate_row(1, 0, np.eye(3))

    del held
    sys.reorder_blocks(np.array([1, 0], dtype=np.uint64))


def test_in_place_updates_are_not_blocked() -> None:
    """Operations that only rewrite values behave as ordinary view writes."""
    sys = BlockSystem(2, 2)
    held = sys.block_storage(0, 1)

    sys.add_block(0, 1, np.ones((2, 2)))
    sys.add_block(0, 0, np.eye(2) * 3.0)
    sys.add_block(1, 0, np.eye(2))
    sys.add_block(1, 1, np.eye(2) * 3.0)
    sys.reserve(1, 4)

    assert np.all(held == 1.0)


def test_releasing_the_view_lifts_the_guard() -> None:
    """The count drops as soon as the last array over the storage dies."""
    sys = two_block_system()
    keep = sys.block_storage(0, 0)
    drop = sys.block_storage(1, 1)
    flip = np.array([1, 0], dtype=np.uint64)

    with pytest.raises(RuntimeError, match="2 array"):
        sys.reorder_blocks(flip)

    del drop
    gc.collect()

    with pytest.raises(RuntimeError, match="1 array"):
        sys.reorder_blocks(flip)

    del keep
    gc.collect()
    sys.reorder_blocks(flip)


def test_view_keeps_the_system_alive() -> None:
    """The system outlives every reference the caller drops."""
    view = BlockSystem(4, 2).block_storage(0, 0)
    gc.collect()

    view[:] = 7.0

    assert np.all(view == 7.0)
    assert view.base is not None
