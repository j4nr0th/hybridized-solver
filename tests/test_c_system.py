"""Check that the C implementation of the system behaves as the Python one does."""

import numpy as np
import pytest
from hybsol._mod import BlockSystem


@pytest.mark.parametrize("n_blocks", (1, 2, 4, 10))
@pytest.mark.parametrize("max_size", (1, 2, 4, 10))
def test_c_system_as_array(n_blocks: int, max_size: int) -> None:
    """Check that C system can be converted to and from an array."""
    rng = np.random.default_rng(67)
    block_sizes = (
        rng.integers(1, max_size, n_blocks)
        if max_size != 1
        else np.ones(n_blocks, dtype=np.uint32)
    )
    block_offsets = np.pad(np.cumsum(block_sizes), (1, 0))
    full_array = rng.random((block_sizes.sum(), block_sizes.sum()))

    sys = BlockSystem(*block_sizes)
    # Zero for now
    assert np.all(sys.as_array() == np.zeros_like(full_array))

    # Fill it up
    for i_row in range(n_blocks):
        for i_col in rng.permutation(n_blocks):
            val = full_array[
                block_offsets[i_row] : block_offsets[i_row + 1],
                block_offsets[i_col] : block_offsets[i_col + 1],
            ]
            sys.add_block(i_row, i_col, val)
            # print(f"Added block {(i_row, i_col)}, {val=}")
            # print(sys.as_array())

    assert np.all(sys.as_array() == full_array)

    second_array = rng.random((block_sizes.sum(), block_sizes.sum()))

    # Add the second
    for i_row in range(n_blocks):
        for i_col in rng.permutation(n_blocks):
            val = second_array[
                block_offsets[i_row] : block_offsets[i_row + 1],
                block_offsets[i_col] : block_offsets[i_col + 1],
            ]
            sys.add_block(i_row, i_col, val)
            # print(f"Added block {(i_row, i_col)}, {val=}")
            # print(sys.as_array())

    # This should be the same as taking the sum
    assert np.all(sys.as_array() == full_array + second_array)


@pytest.mark.parametrize("n_blocks", (1, 2, 4, 10))
@pytest.mark.parametrize("max_size", (1, 2, 4, 10))
def test_c_system_get_block(n_blocks: int, max_size: int) -> None:
    """Check that C system can get blocks back."""
    rng = np.random.default_rng(67)
    block_sizes = (
        rng.integers(1, max_size, n_blocks)
        if max_size != 1
        else np.ones(n_blocks, dtype=np.uint32)
    )
    block_offsets = np.pad(np.cumsum(block_sizes), (1, 0))
    full_array = rng.random((block_sizes.sum(), block_sizes.sum()))

    sys = BlockSystem(*block_sizes)

    # Fill it up
    for i_row in range(n_blocks):
        for i_col in rng.permutation(n_blocks):
            val = full_array[
                block_offsets[i_row] : block_offsets[i_row + 1],
                block_offsets[i_col] : block_offsets[i_col + 1],
            ]
            sys.add_block(i_row, i_col, val)
            # print(f"Added block {(i_row, i_col)}, {val=}")
            # print(sys.as_array())

    assert np.all(sys.as_array() == full_array)

    for i_row in range(n_blocks):
        for i_col in rng.permutation(n_blocks):
            val = full_array[
                block_offsets[i_row] : block_offsets[i_row + 1],
                block_offsets[i_col] : block_offsets[i_col + 1],
            ]
            # print(f"Getting block {(i_row, i_col)}")
            val_2 = sys.get_block(i_row, i_col)
            shape = sys.get_block_size(i_row, i_col)
            assert shape == val.shape
            assert np.all(val == val_2)


@pytest.mark.parametrize("n_blocks", (1, 2, 4, 10))
@pytest.mark.parametrize("max_size", (1, 2, 4, 10))
def test_c_system_multiply_row(n_blocks: int, max_size: int) -> None:
    """Check that C system can get blocks back."""
    rng = np.random.default_rng(21)
    block_sizes = (
        rng.integers(1, max_size, n_blocks)
        if max_size != 1
        else np.ones(n_blocks, dtype=np.uint32)
    )
    block_offsets = np.pad(np.cumsum(block_sizes), (1, 0))
    full_array = rng.random((block_sizes.sum(), block_sizes.sum()))

    sys = BlockSystem(*block_sizes)

    # Fill it up
    for i_row in range(n_blocks):
        for i_col in rng.permutation(n_blocks):
            val = full_array[
                block_offsets[i_row] : block_offsets[i_row + 1],
                block_offsets[i_col] : block_offsets[i_col + 1],
            ]
            sys.add_block(i_row, i_col, val)
            # print(f"Added block {(i_row, i_col)}, {val=}")
            # print(sys.as_array())

    assert np.all(sys.as_array() == full_array)

    for i_row in range(n_blocks):
        diagonal_block = sys.get_block(i_row, i_row)
        inv = np.linalg.inv(diagonal_block)
        sys.multiply_row(i_row, inv)
        for i_col in rng.permutation(n_blocks):
            val = full_array[
                block_offsets[i_row] : block_offsets[i_row + 1],
                block_offsets[i_col] : block_offsets[i_col + 1],
            ]
            # print(f"Getting block {(i_row, i_col)}")
            val_2 = sys.get_block(i_row, i_col)
            shape = sys.get_block_size(i_row, i_col)
            assert shape == val.shape
            assert inv @ val == pytest.approx(val_2)


# TODO: add a sparse version of this test!
@pytest.mark.parametrize("n_blocks", (1, 2, 4, 10))
@pytest.mark.parametrize("max_size", (1, 2, 4, 10))
def test_c_system_eliminate_row_dense(n_blocks: int, max_size: int) -> None:
    """Check that C system can get blocks back with a dense matrix."""
    rng = np.random.default_rng(21)
    block_sizes = (
        rng.integers(1, max_size, n_blocks)
        if max_size != 1
        else np.ones(n_blocks, dtype=np.uint32)
    )
    block_offsets = np.pad(np.cumsum(block_sizes), (1, 0))
    full_array = rng.random((block_sizes.sum(), block_sizes.sum()))

    sys = BlockSystem(*block_sizes)

    # Fill it up
    for i_row in range(n_blocks):
        for i_col in rng.permutation(n_blocks):
            val = full_array[
                block_offsets[i_row] : block_offsets[i_row + 1],
                block_offsets[i_col] : block_offsets[i_col + 1],
            ]
            sys.add_block(i_row, i_col, val)
            # print(f"Added block {(i_row, i_col)}, {val=}")
            # print(sys.as_array())

    assert np.all(sys.as_array() == full_array)

    # Select random rows for elimination
    permuted = rng.permutation(n_blocks)
    sources = permuted[0::2]
    targets = permuted[1::2]
    for i_src, i_tgt in zip(sources, targets):
        mat = sys.get_block(i_tgt, i_src)
        # print(sys.as_array(), i_src, i_tgt)
        # print(sys.get_row_block_indices(i_tgt))
        sys.eliminate_row(i_src, i_tgt, mat)
        # print(sys.as_array())
        # print(sys.get_row_block_indices(i_tgt))

        for i in range(i_src + 1, n_blocks):
            computed = sys.get_block(i_tgt, i)
            expected = full_array[
                block_offsets[i_tgt] : block_offsets[i_tgt + 1],
                block_offsets[i] : block_offsets[i + 1],
            ] - mat @ sys.get_block(i_src, i)
            assert pytest.approx(computed) == expected


@pytest.mark.parametrize("n_blocks", (10, 20, 25))
@pytest.mark.parametrize("max_size", (1, 2, 4, 10))
@pytest.mark.parametrize("density", (0.1, 0.3, 0.5, 0.9))
def test_c_system_eliminate_row_sparse(
    n_blocks: int, max_size: int, density: float
) -> None:
    """Check that C system can get blocks back with a sparse matrix."""
    rng = np.random.default_rng(21)
    block_sizes = (
        rng.integers(1, max_size, n_blocks)
        if max_size != 1
        else np.ones(n_blocks, dtype=np.uint32)
    )
    block_offsets = np.pad(np.cumsum(block_sizes), (1, 0))
    full_array = rng.random((block_sizes.sum(), block_sizes.sum()))

    sys = BlockSystem(*block_sizes)

    # Fill it up
    for i_row in range(n_blocks):
        for i_col in rng.permutation(n_blocks):
            should_add = False
            if i_row == i_col or rng.random(1) < density:
                should_add = True

            if should_add:
                val = full_array[
                    block_offsets[i_row] : block_offsets[i_row + 1],
                    block_offsets[i_col] : block_offsets[i_col + 1],
                ]
                sys.add_block(i_row, i_col, val)
            else:
                full_array[
                    block_offsets[i_row] : block_offsets[i_row + 1],
                    block_offsets[i_col] : block_offsets[i_col + 1],
                ] = 0
            # print(f"Added block {(i_row, i_col)}, {val=}")
            # print(sys.as_array())

    assert np.all(sys.as_array() == full_array)

    # Select random rows for elimination
    permuted = rng.permutation(n_blocks)
    sources = permuted[0::2]
    targets = permuted[1::2]
    for i_src, i_tgt in zip(sources, targets):
        if not sys.has_block(i_tgt, i_src):
            continue
        # print("Not skipped")
        mat = sys.get_block(i_tgt, i_src)
        # print(sys.as_array())
        # print(i_src, i_tgt, mat)
        # print(sys.get_row_block_indices(i_tgt))
        sys.eliminate_row(i_src, i_tgt, mat)
        # print(sys.as_array())
        # print(sys.get_row_block_indices(i_tgt))

        for i in range(i_src + 1, n_blocks):
            if not sys.has_block(i_tgt, i):
                continue

            computed = sys.get_block(i_tgt, i)
            expected = full_array[
                block_offsets[i_tgt] : block_offsets[i_tgt + 1],
                block_offsets[i] : block_offsets[i + 1],
            ]
            if i in sys.get_row_block_indices(i_src):
                expected -= mat @ sys.get_block(i_src, i)
            assert pytest.approx(computed) == expected


@pytest.mark.parametrize(("n", "m"), ((2, 2), (3, 3), (10, 15)))
def test_c_system_inverse(n: int, m: int) -> None:
    """Check that inverse matrix function works correctly."""
    rng = np.random.default_rng(n * m)
    mat = rng.random((n, n))
    lhs = rng.random((n, m))
    rhs = mat @ lhs
    out = np.empty_like(rhs)

    sys = BlockSystem(n)
    sys.add_block(0, 0, mat)
    sys.decompose_diagonal(idx=0)
    sys.solve_diagonal(out=out, idx=0, val=rhs)
    assert pytest.approx(out) == lhs


@pytest.mark.parametrize("n_blocks", (10, 25))
@pytest.mark.parametrize("max_size", (1, 2, 10))
@pytest.mark.parametrize("density", (0.1, 0.5, 0.9))
def test_c_system_copy(n_blocks: int, max_size: int, density: float) -> None:
    """Check that C system can get blocks back with a sparse matrix."""
    rng = np.random.default_rng(21)
    block_sizes = (
        rng.integers(1, max_size, n_blocks)
        if max_size != 1
        else np.ones(n_blocks, dtype=np.uint32)
    )
    block_offsets = np.pad(np.cumsum(block_sizes), (1, 0))
    full_array = rng.random((block_sizes.sum(), block_sizes.sum()))

    sys = BlockSystem(*block_sizes)

    # Fill it up
    for i_row in range(n_blocks):
        for i_col in rng.permutation(n_blocks):
            should_add = False
            if i_row == i_col or rng.random(1) < density:
                should_add = True

            if should_add:
                val = full_array[
                    block_offsets[i_row] : block_offsets[i_row + 1],
                    block_offsets[i_col] : block_offsets[i_col + 1],
                ]
                sys.add_block(i_row, i_col, val)
            else:
                full_array[
                    block_offsets[i_row] : block_offsets[i_row + 1],
                    block_offsets[i_col] : block_offsets[i_col + 1],
                ] = 0
            # print(f"Added block {(i_row, i_col)}, {val=}")
            # print(sys.as_array())

    assert np.all(sys.as_array() == full_array)

    # Make a copy
    cpy = sys.copy()

    assert np.all(sys.as_array() == cpy.as_array())


if __name__ == "__main__":
    with np.printoptions(precision=2, suppress=True):
        # test_c_system_as_array(11, 2)
        # test_c_system_get_block(3, 3)
        # test_c_system_multiply_row(10, 3)
        # test_c_system_eliminate_row_dense(10, 4)
        # test_c_system_eliminate_row_sparse(20, 4, 0.1)
        test_c_system_inverse(4, 4)
