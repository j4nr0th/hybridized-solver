r"""Why a saddle-point system with singular blocks is out of reach.

A 1-D Poisson problem split over two elements gives two element blocks that are
singular on their own: the Neumann element matrix

.. math::

   K = \begin{pmatrix} 1 & -1 \\ -1 & 1 \end{pmatrix}

has its rows summing to zero, so it describes the element only up to a constant.
Continuity across the shared interface and the two Dirichlet ends supply the
missing information, and keeping those constraints as extra unknowns gives

.. math::

   \begin{pmatrix} A & N^\top \\ N & 0 \end{pmatrix}
   \begin{pmatrix} u \\ \lambda \end{pmatrix}
   = \begin{pmatrix} f \\ g \end{pmatrix}.

The system is perfectly well posed — the constraints are what make it solvable —
and this solver still cannot factor it. The reason is not the block structure
but the factorization: there is no pivoting, so every leading principal minor of
every diagonal block must be nonzero, and the element blocks are singular.

No ordering helps. Whichever block reaches the front of its row first has to be
factorized, and here that is always either a singular element block or an
identically zero multiplier block.
"""

import itertools

import numpy as np
from hybsol import BlockSystem

# --------------------------------------------------------------------------
# Assemble [[A, N^T], [N, 0]] on two singular Neumann elements.
# --------------------------------------------------------------------------
ELEMENT = np.array([[1.0, -1.0], [-1.0, 1.0]])
BLOCK_SIZES = [2, 2, 1, 1, 1]
OFFSETS = np.concatenate([[0], np.cumsum(BLOCK_SIZES)])


def assemble(constraint_mass: np.ndarray) -> np.ndarray:
    """Build the 7x7 operator, with ``constraint_mass`` in the (2,2) corner."""
    operator = np.zeros((7, 7))
    for element in (0, 1):
        operator[2 * element, 2 * element] = 1.0
        operator[2 * element, 2 * element + 1] = -1.0
        operator[2 * element + 1, 2 * element] = -1.0
        operator[2 * element + 1, 2 * element + 1] = 1.0

    operator[4, 1] = 1.0  # continuity, u2 = u3
    operator[4, 2] = -1.0
    operator[5, 0] = 1.0  # Dirichlet at the left end
    operator[6, 3] = 1.0  # Dirichlet at the right end
    for col in range(4, 7):
        for row in range(4):
            operator[row, col] = operator[col, row]
    operator[4:7, 4:7] = constraint_mass
    return operator


def build(operator):
    """Lay the operator out as blocks."""
    blocks = {}
    for row in range(len(BLOCK_SIZES)):
        for col in range(len(BLOCK_SIZES)):
            values = operator[
                OFFSETS[row] : OFFSETS[row + 1], OFFSETS[col] : OFFSETS[col + 1]
            ].copy()
            # A zero block still has to be stored, or the pattern looks incomplete.
            if values.any() or row == col:
                blocks[(row, col)] = values
    return BlockSystem.from_block_list(
        [(row, col, values) for (row, col), values in blocks.items()], BLOCK_SIZES
    )


operator = assemble(np.zeros((3, 3)))

print("element block determinant:", np.linalg.det(operator[0:2, 0:2]))
print("multiplier block is zero :", not operator[4:7, 4:7].any())
print("full system rank         :", np.linalg.matrix_rank(operator), "of 7")
print("full system determinant  :", np.linalg.det(operator))

# --------------------------------------------------------------------------
# No ordering works.
# --------------------------------------------------------------------------
failures = successes = 0
for placement in itertools.permutations(range(5)):
    system = build(operator)
    order = [0] * 5
    for old, new in enumerate(placement):
        order[old] = new
    rebuilt = system
    if rebuilt.has_block(0, 0):
        try:
            rebuilt.reorder_blocks(np.array(order))
            rebuilt.decompose(n_threads=1)
            successes += 1
        except Exception:
            failures += 1
    else:
        failures += 1

print(
    f"\nall {failures + successes} block orderings tried: "
    f"{failures} failed, {successes} succeeded"
)

system = build(operator)
try:
    system.decompose()
except ValueError as error:
    print("plain decompose ->", error)

# --------------------------------------------------------------------------
# What the factorization actually requires
# --------------------------------------------------------------------------
# Not a nonsingular block, but nonzero leading principal minors. A block can
# clear that bar and still be rejected, and a well-behaved block can be
# mis-factored if its minor merely vanishes numerically.
print("\nleading principal minors of the element block:")
for k in (1, 2):
    print(f"  order {k}: {np.linalg.det(operator[0:k, 0:k]):+.3e}")

near = np.array(
    [
        [2.0, -1.0, -1.0, 0.0, 1.0],
        [-1.0, 2.0, 0.0, -1.0, 0.0],
        [-1.0, 0.0, 2.0, -1.0, 0.0],
        [0.0, -1.0, -1.0, 2.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
    ]
)
print("\na nonsingular block with a vanishing leading minor:")
print("  determinant        :", np.linalg.det(near))
print("  leading entry      :", near[0, 0])
print("  leading 4x4 minor  :", np.linalg.det(near[:4, :4]))
probe = BlockSystem(5)
probe.add_block(0, 0, near)
probe.decompose_diagonal(0)
rhs = np.arange(1.0, 6.0)
print(
    "  residual after solve:", np.linalg.norm(near @ probe.solve_diagonal(0, rhs) - rhs)
)
