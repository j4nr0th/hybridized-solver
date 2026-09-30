"""Time the block solver against SciPy's sparse LU.

The scratch version of this script, kept alongside ``bench_solve_vs_scipy``
because it answers the same question on a larger system. The difference is that
this one has no command-line handling and reports every round instead of only
the best case.

Two things had to change for the current API:

* `reorder_vector` moves a vector into the ordering `reorder_blocks` produced,
  and `unorder_vector` brings it back. The other way round silently solves the
  wrong system and lands on a completely different answer.
* The diagonal shift used to be scaled by the block count, which moved the
  conditioning enough that the reordering lost about two digits. It is back to
  the small constant the script always used, so `np.allclose` passes again.
"""

from time import perf_counter

import numpy as np
from hybsol import BlockSystem
from scipy import sparse as sp
from scipy.sparse import linalg as sla

# sys.activate_stack_trampoline only exists from 3.14 on.
try:
    from sys import activate_stack_trampoline, deactivate_stack_trampoline
except ImportError:  # pragma: no cover

    def activate_stack_trampoline(name):  # type: ignore[misc]
        """No-op stand-in on interpreters without stack trampolines."""

    def deactivate_stack_trampoline():  # type: ignore[misc]
        """No-op stand-in on interpreters without stack trampolines."""


def random_sparse_system(
    n_blocks: int, max_block_size: int, sparsity: float
) -> BlockSystem:
    """Create a random sparse system with symmetric sparsity."""
    rng = np.random.default_rng(23095)
    block_sizes = rng.integers(1, max_block_size, n_blocks)
    sys = BlockSystem(*block_sizes)
    for ir in range(n_blocks):
        sizes = block_sizes[ir:]
        # Deliberately not scaled with n_blocks: see the module docstring.
        sys.add_block(ir, ir, rng.random((sizes[0], sizes[0])) + 0.01)
        for ic, sz in zip(range(ir + 1, n_blocks), sizes[1:], strict=True):
            if sparsity > rng.random(1):
                continue
            sys.add_block(ir, ic, rng.random((sizes[0], sz)))
            sys.add_block(ic, ir, rng.random((sz, sizes[0])))

    return sys


def block_system_to_sp(sys: BlockSystem) -> sp.csc_array:
    """Convert the block system into a scipy CSC array."""
    return sp.block_array(
        [
            [
                (sys.get_block(i_row, i_col) if sys.has_block(i_row, i_col) else None)
                for i_col in range(sys.n_blocks)
            ]
            for i_row in range(sys.n_blocks)
        ],
        format="csc",
    )


if __name__ == "__main__":
    n_rounds, n_blocks, max_block_size, sparsity = 10, 500, 40, 0.995

    t_sp = 0.0
    t_me = 0.0

    sys = random_sparse_system(n_blocks, max_block_size, sparsity)
    csc = block_system_to_sp(sys)
    print(f"Solving a {csc.shape[0]} x {csc.shape[0]} system {n_rounds} times")

    # To eyeball the sparsity pattern, import pyplot and show it:
    #
    #     from matplotlib import pyplot as plt
    #     fig, ax = plt.subplots()
    #     ax.spy(csc)
    #     plt.show()

    for _irnd in range(n_rounds):
        decomp_me = sys.copy()

        rng = np.random.default_rng(n_blocks * max_block_size)

        lhs = rng.random(csc.shape[0])
        rhs = csc @ lhs

        t = perf_counter()
        activate_stack_trampoline("perf")
        decomp_scipy = sla.splu(csc)
        lhs_sp = decomp_scipy.solve(rhs)
        deactivate_stack_trampoline()
        t1 = perf_counter() - t
        print("Scipy time:", t1)
        t_sp += t1

        t = perf_counter()
        activate_stack_trampoline("perf")
        ordering = decomp_me.compute_reordering("greedy")
        decomp_me.reorder_blocks(ordering)
        decomp_me.decompose()
        # `rhs` still lives in the old ordering, so it goes through
        # reorder_vector first and the solution comes back through
        # unorder_vector.
        prhs = decomp_me.reorder_vector(ordering, rhs)
        plhs_me = decomp_me.solve(prhs)
        lhs_me = decomp_me.unorder_vector(ordering, plhs_me)
        deactivate_stack_trampoline()
        t1 = perf_counter() - t
        print("Jan time:", t1)
        t_me += t1

        print(f"My max error was {np.abs(lhs_me - lhs).max()}")
        print(f"SciPy max error was {np.abs(lhs_sp - lhs).max()}")
        assert np.allclose(lhs_sp, lhs)
        assert np.allclose(lhs_me, lhs)

    print(f"Ran {n_rounds} rounds")
    print(f"Average SciPy time {t_sp / n_rounds}")
    print(f"Average Jan time {t_me / n_rounds}")
