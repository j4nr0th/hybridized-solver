"""Recover the digits the factorization gives away.

The block decomposition only pivots to repair an exactly zero pivot, so a solve
comes back a few digits short of what a fully pivoted reference gets. Iterative
refinement buys them back without touching the factorization: form the residual,
solve for a correction, add it.

Every step reuses the factors :meth:`hybsol.BlockSystem.decompose` produced, so
a step costs one extra solve and one matrix-vector product on top of work that
was done anyway. The residual below is formed in ordinary double precision, and
one step is enough to bring the solve down to the rounding limit.

How many steps to take is left to the caller. The loop runs two refinements to
show where the improvement stops; a program would watch the residual and stop
when it stops falling.

The one thing the recipe needs is the *original* matrix.
:meth:`hybsol.BlockSystem.decompose` factorizes in place, so once it has run
the system holds factors rather than ``A``. Keep the matrix the system was
assembled from. This example reads it back with
:meth:`hybsol.BlockSystem.as_array` so that it stays self-contained; in a real
program it is usually cheaper to hold on to the sparse matrix that was
assembled in the first place, and any object that supports ``@`` works here.
"""

import numpy as np
from hybsol import BlockSystem

rng = np.random.default_rng(23095)
n_blocks, max_block_size, sparsity = 300, 6, 0.95

# --------------------------------------------------------------------------
# Assemble the system, and keep a copy of the matrix before it is factored.
# --------------------------------------------------------------------------
block_sizes = rng.integers(1, max_block_size, n_blocks)
system = BlockSystem(*block_sizes)
for i_row in range(n_blocks):
    size = block_sizes[i_row]
    system.add_block(i_row, i_row, rng.random((size, size)) + 0.01)
    for i_col in range(i_row + 1, n_blocks):
        if rng.random() < sparsity:
            continue
        upper = rng.random((block_sizes[i_row], block_sizes[i_col]))
        lower = rng.random((block_sizes[i_col], block_sizes[i_row]))
        system.add_block(i_row, i_col, upper)
        system.add_block(i_col, i_row, lower)

matrix = system.as_array()
dim = int(matrix.shape[0])
print(f"built a {dim} x {dim} matrix from {system.n_blocks} blocks")

lhs = rng.random(dim)
rhs = matrix @ lhs

# --------------------------------------------------------------------------
# Factorize once, in the coloring-based reordering.
# --------------------------------------------------------------------------
ordering = system.compute_reordering("greedy")
system.reorder_blocks(ordering)
system.decompose()


def solve(b: np.ndarray) -> np.ndarray:
    """Solve ``A x = b`` for a right-hand side in the original ordering.

    The system solves in its own reordered basis, so both the right-hand side
    and the answer travel through ``reorder_vector`` and ``unorder_vector``.

    Parameters
    ----------
    b : numpy.ndarray
        Right-hand side, in the ordering the matrix was assembled in.

    Returns
    -------
    numpy.ndarray
        Solution, in the same ordering as ``b``.
    """
    reordered = system.reorder_vector(ordering, b)
    return system.unorder_vector(ordering, system.solve(reordered))


def measure(x: np.ndarray) -> tuple[float, float]:
    """Return the residual and the true error of ``x`` as infinity norms.

    The residual is the one that can be formed without knowing the answer, so
    it is what a program would actually watch.

    Parameters
    ----------
    x : numpy.ndarray
        Candidate solution.

    Returns
    -------
    tuple of float
        ``max |b - A x|`` and ``max |x - lhs|``.
    """
    residual = float(np.abs(rhs - matrix @ x).max())
    error = float(np.abs(x - lhs).max())
    return residual, error


# --------------------------------------------------------------------------
# Refine: r = b - A x, solve for the correction, add it to the solution.
# --------------------------------------------------------------------------
x = solve(rhs)
residual, error = measure(x)
initial_error = error

print(" step   residual     error")
print(f"{0:5d}   {residual:.3e}   {error:.3e}")
for step in range(1, 3):
    x += solve(rhs - matrix @ x)
    residual, error = measure(x)
    print(f"{step:5d}   {residual:.3e}   {error:.3e}")

assert error < initial_error / 100, f"refinement did not help: {initial_error}"
assert np.allclose(x, lhs)
