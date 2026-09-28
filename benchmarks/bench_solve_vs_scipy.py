"""Compare the block solver against SciPy's sparse LU.

The point of the comparison is not to win on raw dense speed but to check that
the block decomposition keeps its edge on the kind of loosely coupled block
systems the solver is meant for.
"""

import argparse
import json
from time import perf_counter

import numpy as np
from hybsol import BlockSystem
from scipy import sparse as sp
from scipy.sparse import linalg as sla


def random_sparse_system(
    n_blocks: int, max_block_size: int, sparsity: float
) -> BlockSystem:
    """Create a random sparse system with symmetric sparsity.

    Parameters
    ----------
    n_blocks : int
        Number of blocks per dimension.
    max_block_size : int
        Exclusive upper bound for the size of an individual block.
    sparsity : float
        Probability with which an off-diagonal pair of blocks is dropped.

    Returns
    -------
    BlockSystem
        A diagonally dominant system with a symmetric sparsity pattern.
    """
    rng = np.random.default_rng(23095)
    block_sizes = rng.integers(1, max_block_size, n_blocks)
    sys = BlockSystem(*block_sizes)
    for ir in range(n_blocks):
        sizes = block_sizes[ir:]
        sys.add_block(ir, ir, rng.random((sizes[0], sizes[0])) + n_blocks * 0.01)
        for ic, sz in zip(range(ir + 1, n_blocks), sizes[1:], strict=True):
            if sparsity > rng.random(1):
                continue
            sys.add_block(ir, ic, rng.random((sizes[0], sz)))
            sys.add_block(ic, ir, rng.random((sz, sizes[0])))

    return sys


def block_system_to_sp(sys: BlockSystem) -> sp.csc_array:
    """Convert the block system into a SciPy CSC array.

    Parameters
    ----------
    sys : BlockSystem
        The system to convert.

    Returns
    -------
    scipy.sparse.csc_array
        The same matrix, assembled block by block.
    """
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


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=10, help="Number of rounds.")
    parser.add_argument("--n-blocks", type=int, default=500, help="Blocks per dimension.")
    parser.add_argument(
        "--max-block-size", type=int, default=40, help="Exclusive block size bound."
    )
    parser.add_argument(
        "--sparsity", type=float, default=0.995, help="Off-diagonal drop probability."
    )
    parser.add_argument(
        "--json", action="store_true", help="Print the result as a single JSON line."
    )
    return parser.parse_args()


def main() -> None:
    """Run the comparison and report the average times."""
    args = parse_args()

    sys = random_sparse_system(args.n_blocks, args.max_block_size, args.sparsity)
    csc = block_system_to_sp(sys)

    times_scipy: list[float] = []
    times_hybsol: list[float] = []
    error_hybsol = 0.0
    error_scipy = 0.0

    for round_index in range(args.rounds):
        decomposed = sys.copy()
        rng = np.random.default_rng(args.n_blocks * args.max_block_size + round_index)

        lhs = rng.random(csc.shape[0])
        rhs = csc @ lhs

        start = perf_counter()
        lhs_scipy = sla.splu(csc).solve(rhs)
        times_scipy.append(perf_counter() - start)

        start = perf_counter()
        ordering = decomposed.compute_reordering("greedy")
        decomposed.reorder_blocks(ordering)
        decomposed.decompose()
        solution = decomposed.solve(decomposed.reorder_vector(ordering, rhs))
        lhs_hybsol = decomposed.unorder_vector(ordering, solution)
        times_hybsol.append(perf_counter() - start)

        error_hybsol = max(error_hybsol, float(np.abs(lhs_hybsol - lhs).max()))
        error_scipy = max(error_scipy, float(np.abs(lhs_scipy - lhs).max()))

        assert np.allclose(lhs_scipy, lhs)
        assert np.allclose(lhs_hybsol, lhs)

    result = {
        "rounds": args.rounds,
        "n_blocks": args.n_blocks,
        "max_block_size": args.max_block_size,
        "sparsity": args.sparsity,
        "size": int(csc.shape[0]),
        "scipy_s": min(times_scipy),
        "hybsol_s": min(times_hybsol),
        "scipy_mean_s": float(np.mean(times_scipy)),
        "hybsol_mean_s": float(np.mean(times_hybsol)),
        "max_error_hybsol": error_hybsol,
        "max_error_scipy": error_scipy,
    }

    if args.json:
        print(json.dumps(result))
    else:
        print(f"Ran {args.rounds} rounds on a {csc.shape[0]} x {csc.shape[0]} system")
        print(f"  SciPy : {result['scipy_mean_s']:.4f} s (best {result['scipy_s']:.4f})")
        print(
            f"  hybsol: {result['hybsol_mean_s']:.4f} s (best {result['hybsol_s']:.4f})"
        )
        print(f"  max error: hybsol {error_hybsol:.2e}, SciPy {error_scipy:.2e}")


if __name__ == "__main__":
    main()
