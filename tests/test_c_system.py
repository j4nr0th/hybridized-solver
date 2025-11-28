"""Check that the C implementation of the system behaves as the Python one does."""

import numpy as np
import pytest
from hybsol._mod import BlockSystem as CBlockSystem

# from hybsol.system import BlockSystem as PyBlockSystem


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

    sys = CBlockSystem(*block_sizes)
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


if __name__ == "__main__":
    with np.printoptions(precision=2, suppress=True):
        test_c_system_as_array(11, 2)
