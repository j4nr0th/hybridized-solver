"""Assemble a block system and solve it.

This example walks through the full life cycle of a
:class:`hybsol.BlockSystem`: filling it with blocks, checking that the
structure is one the solver can factorize, decomposing it and solving for a
right-hand side.
"""

import numpy as np
from hybsol import BlockSystem

# --------------------------------------------------------------------------
# A random 8-block system with block sizes between 1 and 4.
# --------------------------------------------------------------------------
rng = np.random.default_rng(7)
n_blocks = 8
block_sizes = rng.integers(1, 4, n_blocks)
dim = int(block_sizes.sum())


def build() -> BlockSystem:
    """Build the same random system from scratch."""
    local_rng = np.random.default_rng(7)
    sizes = local_rng.integers(1, 4, n_blocks)
    system = BlockSystem(*sizes)
    for i_row in range(n_blocks):
        size = int(sizes[i_row])
        system.add_block(i_row, i_row, local_rng.random((size, size)) + 2.0 * n_blocks)
        for i_col in range(i_row + 1, n_blocks):
            if local_rng.random() < 0.6:
                continue
            # The pattern has to be symmetric, so both blocks are stored.
            other = int(sizes[i_col])
            system.add_block(i_row, i_col, local_rng.random((size, other)))
            system.add_block(i_col, i_row, local_rng.random((other, size)))
    return system


system = build()
print(f"assembled a {dim} x {dim} system from {system.n_blocks} blocks")
print("structurally valid:", system.is_valid())

# --------------------------------------------------------------------------
# Decompose once, then solve. The decomposition owns a copy of the blocks, so
# the system is left as it was assembled and could be decomposed again.
# --------------------------------------------------------------------------
matrix = system.as_array()
lhs = rng.random(dim)
rhs = matrix @ lhs

decomposition = system.decompose()
print(f"decomposition recorded {len(decomposition.operations())} operations")

solution = decomposition.solve(rhs)
print("solve error:", np.abs(solution - lhs).max())

# --------------------------------------------------------------------------
# A reordered system solves the same problem with fewer eliminations.
# --------------------------------------------------------------------------
ordered = build()
ordering = ordered.compute_reordering("greedy")
ordered.reorder_blocks(ordering)
ordered_decomposition = ordered.decompose()

reordered_lhs = ordered.unorder_vector(
    ordering,
    ordered_decomposition.solve(ordered.reorder_vector(ordering, rhs)),
)
print("reordered solve error:", np.abs(reordered_lhs - lhs).max())
