"""Compare the block solver against a sparse LU from SciPy.

The block decomposition is most attractive for systems that are made of many
loosely coupled dense blocks. This example times both solvers on such a system
and checks that they agree.
"""

import numpy as np
from hybsol import BlockSystem
from scipy import sparse as sp
from scipy.sparse import linalg as sla

rng = np.random.default_rng(23095)
n_blocks, max_block_size, sparsity = 300, 6, 0.95

# --------------------------------------------------------------------------
# Build the same matrix twice: once as a block system, once as sparse CSC.
# --------------------------------------------------------------------------
block_sizes = rng.integers(1, max_block_size, n_blocks)
offsets = np.pad(np.cumsum(block_sizes), (1, 0))
dim = int(offsets[-1])

system = BlockSystem(*block_sizes)
blocks: list[list[object]] = [[None for _ in range(n_blocks)] for _ in range(n_blocks)]
for i_row in range(n_blocks):
    size = block_sizes[i_row]
    diagonal = rng.random((size, size)) + n_blocks * 0.01
    system.add_block(i_row, i_row, diagonal)
    blocks[i_row][i_row] = diagonal

    for i_col in range(i_row + 1, n_blocks):
        if rng.random() < sparsity:
            continue
        upper = rng.random((block_sizes[i_row], block_sizes[i_col]))
        lower = rng.random((block_sizes[i_col], block_sizes[i_row]))
        system.add_block(i_row, i_col, upper)
        system.add_block(i_col, i_row, lower)
        blocks[i_row][i_col] = upper
        blocks[i_col][i_row] = lower

csc = sp.block_array(blocks, format="csc")
print(f"built a {dim} x {dim} matrix with {csc.nnz} non-zeros")

lhs = rng.random(dim)
rhs = csc @ lhs

# --------------------------------------------------------------------------
# Solve it both ways.
# --------------------------------------------------------------------------
lhs_scipy = sla.splu(csc).solve(rhs)

ordered = system.copy()
ordering = ordered.compute_reordering("greedy")
ordered.reorder_blocks(ordering)
ordered_decomposition = ordered.decompose()
lhs_hybsol = ordered.unorder_vector(
    ordering, ordered_decomposition.solve(ordered.reorder_vector(ordering, rhs))
)

print(f"SciPy max error : {np.abs(lhs_scipy - lhs).max():.3e}")
print(f"hybsol max error: {np.abs(lhs_hybsol - lhs).max():.3e}")
assert np.allclose(lhs_scipy, lhs)
assert np.allclose(lhs_hybsol, lhs)
