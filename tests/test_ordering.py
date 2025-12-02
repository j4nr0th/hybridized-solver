"""Check reordering functions work correctly."""

from typing import Literal

import numpy as np
import pytest
from hybsol._mod import BlockSystem


def random_sparse_system(
    rng: np.random.Generator, n_blocks: int, max_block_size: int, sparsity: float
) -> BlockSystem:
    """Create a random sparse system with symmetric sparsity."""
    block_sizes = rng.integers(1, max_block_size, n_blocks)
    sys = BlockSystem(*block_sizes)
    for ir in range(n_blocks):
        sizes = block_sizes[ir:]
        sys.add_block(ir, ir, rng.random((sizes[0], sizes[0])) + 0.1)
        for ic, sz in zip(range(ir + 1, n_blocks), sizes[1:], strict=True):
            if sparsity > rng.random(1):
                continue
            sys.add_block(ir, ic, rng.random((sizes[0], sz)))
            sys.add_block(ic, ir, rng.random((sz, sizes[0])))

    return sys


@pytest.mark.parametrize("strat", ("first", "greedy", "balanced"))
@pytest.mark.parametrize(("nb", "bs", "sp"), ((10, 5, 0.3), (15, 5, 0.9), (20, 2, 0.8)))
def test_strategies(
    strat: Literal["first", "greedy", "balanced"], nb: int, bs: int, sp: float
) -> None:
    """Check that the strategy produces a valid ordering."""
    rng = np.random.default_rng(1359)
    sys = random_sparse_system(rng, nb, bs, sp)
    orderings = sys.compute_reordering(strat)
    print(orderings)
    # Each index should appear exactly once
    assert np.all(np.unique(orderings) == tuple(range(nb)))


if __name__ == "__main__":
    for strat in ("first", "greedy", "balanced"):
        for nb, bs, sp in ((10, 5, 0.3), (15, 5, 0.9), (20, 2, 0.8)):
            test_strategies(strat, nb, bs, sp)


# def test_greedy_coloring(n_blocks: int, max_size: int, sparsity: float) -> None:
#     """Check how the greedy coloring algorithm works."""

# if __name__ == "__main__":
#     from matplotlib import pyplot as plt

#     sys = random_sparse_system(np.random.default_rng(30), 100, 4, 0.9)

#     fig, ax = plt.subplots()

#     ax.spy(sys.as_array())
#     plt.show()

#     new_ordering = reorder_blocks(sys, strategy="balanced")
#     print(new_ordering)
#     assert len(np.unique(new_ordering)) == len(new_ordering)

#     old_sys = sys.copy()
#     old_sizes = sys.block_sizes
#     sys.reorder_blocks(new_ordering)
#     new_sizes = sys.block_sizes
#     assert np.all(old_sizes == new_sizes[new_ordering])

#     fig, ax = plt.subplots()

#     ax.spy(sys.as_array())
#     plt.show()
