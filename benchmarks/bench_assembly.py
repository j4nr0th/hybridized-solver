"""Compare the assembly entry points against each other.

Three ways of filling a system are timed:

``add_block``
    One Python call per block, inserting into a sorted row.
``add_blocks``
    One call for the whole system, given flat COO index arrays and packed data.
``from_blocks``
    Like ``add_blocks``, but also constructs the system.
"""

import argparse
import json
from time import perf_counter

import numpy as np
from hybsol import BlockSystem


def random_pattern(
    n_blocks: int, max_block_size: int, sparsity: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a flat COO description of a random block system.

    Parameters
    ----------
    n_blocks : int
        Number of blocks per dimension.
    max_block_size : int
        Exclusive upper bound for the size of an individual block.
    sparsity : float
        Probability with which an off-diagonal pair of blocks is dropped.
    seed : int
        Seed of the random generator, so runs are reproducible.

    Returns
    -------
    tuple of array
        Block sizes, row indices, column indices, the packed block data, the
        offset of every block inside the data and the length of every block.
    """
    rng = np.random.default_rng(seed)
    block_sizes = rng.integers(1, max_block_size, n_blocks).astype(np.uint64)

    rows: list[int] = []
    cols: list[int] = []
    values: list[np.ndarray] = []
    for i_row in range(n_blocks):
        rows.append(i_row)
        cols.append(i_row)
        values.append(rng.random((block_sizes[i_row], block_sizes[i_row])) + n_blocks)
        for i_col in range(i_row + 1, n_blocks):
            if sparsity > rng.random(1):
                continue
            rows += [i_row, i_col]
            cols += [i_col, i_row]
            values.append(rng.random((block_sizes[i_row], block_sizes[i_col])))
            values.append(rng.random((block_sizes[i_col], block_sizes[i_row])))

    # `add_blocks` expects the block values packed back to back.
    data = np.concatenate([value.ravel() for value in values])
    lengths = (block_sizes[np.array(rows)] * block_sizes[np.array(cols)]).astype(np.int64)
    starts = np.cumsum(lengths) - lengths
    return block_sizes, np.array(rows), np.array(cols), data, starts, lengths


def time_assembly(
    n_blocks: int, max_block_size: int, sparsity: float, rounds: int, seed: int
) -> dict[str, float]:
    """Time the three assembly paths on the same pattern.

    Parameters
    ----------
    n_blocks : int
        Number of blocks per dimension.
    max_block_size : int
        Exclusive upper bound for the size of an individual block.
    sparsity : float
        Probability with which an off-diagonal pair of blocks is dropped.
    rounds : int
        How often every path is run; the best time is reported.
    seed : int
        Seed of the random generator.

    Returns
    -------
    dict of str to float
        Best wall-clock time of every path, in seconds.
    """
    block_sizes, rows, cols, data, starts, lengths = random_pattern(
        n_blocks, max_block_size, sparsity, seed
    )

    def block_array(index: int) -> np.ndarray:
        """Return the values of block ``index`` with its proper shape."""
        n_rows = int(block_sizes[rows[index]])
        n_cols = int(block_sizes[cols[index]])
        return data[starts[index] : starts[index] + lengths[index]].reshape(
            n_rows, n_cols
        )

    block_list = [(int(rows[k]), int(cols[k]), block_array(k)) for k in range(len(rows))]

    best: dict[str, float] = {"add_block": np.inf, "add_blocks": np.inf}
    for _ in range(rounds):
        sys = BlockSystem(*block_sizes)
        start = perf_counter()
        for k, (row, col) in enumerate(zip(rows.tolist(), cols.tolist(), strict=True)):
            sys.add_block(row, col, block_array(k))
        best["add_block"] = min(best["add_block"], perf_counter() - start)

        sys = BlockSystem(*block_sizes)
        start = perf_counter()
        sys.add_blocks(rows, cols, data)
        best["add_blocks"] = min(best["add_blocks"], perf_counter() - start)

    start = perf_counter()
    sys = BlockSystem.from_blocks(block_sizes, rows, cols, data)
    best["from_blocks"] = perf_counter() - start

    start = perf_counter()
    sys = BlockSystem.from_block_list(block_list, block_sizes)
    best["from_block_list"] = perf_counter() - start

    return best


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3, help="Rounds per path.")
    parser.add_argument("--n-blocks", type=int, default=200, help="Blocks per dimension.")
    parser.add_argument(
        "--max-block-size", type=int, default=8, help="Exclusive block size bound."
    )
    parser.add_argument(
        "--sparsity", type=float, default=0.9, help="Off-diagonal drop probability."
    )
    parser.add_argument(
        "--json", action="store_true", help="Print the result as a single JSON line."
    )
    return parser.parse_args()


def main() -> None:
    """Run the assembly benchmark and report the times."""
    args = parse_args()
    result = time_assembly(
        args.n_blocks, args.max_block_size, args.sparsity, args.rounds, seed=17
    )

    if args.json:
        print(json.dumps({"n_blocks": args.n_blocks, **result}))
    else:
        print(f"Assembled a {args.n_blocks} block system {args.rounds} times")
        for name, seconds in result.items():
            print(f"  {name:>16}: {seconds:.4f} s")


if __name__ == "__main__":
    main()
