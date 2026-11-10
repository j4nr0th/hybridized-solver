"""Compare the CPU (OpenMP), OpenCL and HIP backends on one seeded system.

Each row times create-and-factorize and solve separately, so the device
backends' upload and schedule build (both inside ``decompose``) show up
against the CPU's thread sweep rather than hiding in one number. Every
GPU solve is also checked against the CPU solve of the same system and
reports the relative difference; the GPU numbers are only comparable when
that column stays near the precision of the factors. The shape is a block
tridiagonal band plus a hub row; the shift added to each diagonal block
follows its neighbours' norms, but whole-row dominance is not asserted,
so factorizing without pivoting is an observed property of the fixed
seeds, not a guarantee of the construction.

Run on a quiet machine: the GPU is shared, and a contended run misranks
the backends. ``--json`` prints one JSON line per (shape, precision) so
the table can be parsed after the fact.
"""

import argparse
import json
import os
import statistics
import sys
from time import perf_counter
from typing import Any

import numpy as np
from hybsol import BlockSystem, DeviceError, Precision

_SYSTEM_SEED = 17
_RHS_SEED = 23095


def physical_core_count() -> int:
    """Count the physical cores this process is allowed to run on.

    Leaving ``n_threads`` at zero lets OpenMP pick the *logical* count,
    which on an SMT machine asks for more threads than there are cores;
    the CPU rows are therefore pinned to this number.

    Returns
    -------
    int
        Distinct ``(package, core)`` pairs, or the logical count when the
        CPU topology cannot be read.
    """
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - no affinity API on this platform
        return os.cpu_count() or 1

    base = "/sys/devices/system/cpu"
    cores = set()
    for cpu in cpus:
        try:
            with open(f"{base}/cpu{cpu}/topology/physical_package_id") as handle:
                package = handle.read().strip()
            with open(f"{base}/cpu{cpu}/topology/core_id") as handle:
                core = handle.read().strip()
        except OSError:  # pragma: no cover - sysfs is not mounted
            return os.cpu_count() or 1
        cores.add((package, core))

    return len(cores) or os.cpu_count() or 1


def bench_system(
    n_blocks: int, block_size: int, precision: Precision
) -> tuple[BlockSystem, np.ndarray]:
    """Build the comparison system: a block tridiagonal band plus a hub row.

    Block row 0 connects to every other block row, so the symbolic passes
    have real width; every diagonal block is shifted past the norms of its
    row, which is what makes the system factorizable without pivoting in
    every backend.

    Parameters
    ----------
    n_blocks : int
        Number of blocks per dimension.
    block_size : int
        Size of every block.
    precision : hybsol.Precision
        Storage of the system and, by default, of the factors.

    Returns
    -------
    tuple
        The system and its dense matrix, for reference solves.
    """
    rng = np.random.default_rng(_SYSTEM_SEED)
    blocks: dict[tuple[int, int], np.ndarray] = {}
    for i in range(n_blocks):
        for j in range(i + 1):
            if i == j or j == 0 or i - j == 1:
                blocks[(i, j)] = rng.standard_normal((block_size, block_size))

    for i in range(n_blocks):
        off = sum(
            float(np.linalg.norm(blocks[(max(i, j), min(i, j))], 2))
            for j in range(n_blocks)
            if j != i and (max(i, j), min(i, j)) in blocks
        )
        g = blocks[(i, i)]
        row = float(np.abs(g).sum(axis=1).max())
        col = float(np.abs(g).sum(axis=0).max())
        blocks[(i, i)] = g + (off + row + col + 1.0) * np.eye(block_size)

    rows: list[int] = []
    cols: list[int] = []
    data: list[np.ndarray] = []
    dense = np.zeros((n_blocks * block_size, n_blocks * block_size))
    offsets = np.cumsum((0, *(block_size,) * n_blocks))
    for (i, j), block in blocks.items():
        rows.append(i)
        cols.append(j)
        data.append(block)
        # Only off-diagonal blocks get a stored partner: from_blocks sums
        # duplicates, so adding (i, i) twice would double the diagonal.
        if i != j:
            rows.append(j)
            cols.append(i)
            data.append(np.ascontiguousarray(block.T))
            dense[offsets[j] : offsets[j + 1], offsets[i] : offsets[i + 1]] = block.T
        dense[offsets[i] : offsets[i + 1], offsets[j] : offsets[j + 1]] = block

    system = BlockSystem.from_blocks(
        [block_size] * n_blocks,
        np.array(rows),
        np.array(cols),
        np.concatenate([b.ravel() for b in data]),
        precision=precision,
    )
    assert system.is_valid()
    # The guard catches mis-stored blocks (duplicate entries are summed);
    # single-precision storage rounds, so compare with tolerances.
    assert np.allclose(system.as_array(), dense)
    return system, dense


def timed_pair(decompose, rhs: np.ndarray, rounds: int) -> tuple[float, float]:  # noqa: ANN001
    """Time create-and-factorize and solve apart; return the two medians.

    The solve is measured on one decomposition solved ``rounds`` times,
    which is the replay path the recorded operations exist for.

    Parameters
    ----------
    decompose : callable
        Zero-argument call producing a factorized ``Decomposition``.
    rhs : numpy.ndarray
        The right-hand side.
    rounds : int
        Rounds for each of the two measurements.

    Returns
    -------
    tuple
        Median decompose seconds and median solve seconds.
    """
    dec = decompose()  # warmup: module load, kernel build, first launch
    factor_times = []
    for _ in range(max(3, rounds)):
        start = perf_counter()
        decompose()
        factor_times.append(perf_counter() - start)

    dec.solve(rhs)
    solve_times = []
    for _ in range(max(3, rounds)):
        start = perf_counter()
        dec.solve(rhs)
        solve_times.append(perf_counter() - start)
    return (
        float(statistics.median(factor_times)),
        float(statistics.median(solve_times)),
    )


def _relative_error(got: np.ndarray, reference: np.ndarray) -> float:
    """Report the worst relative difference between two solutions."""
    scale = float(np.abs(reference).max()) or 1.0
    return float(np.abs(got - reference).max() / scale)


def device_rows(
    system: BlockSystem,
    rhs: np.ndarray,
    reference: np.ndarray,
    rounds: int,
    hip_arith: list[str],
) -> list[dict[str, Any]]:
    """Time every device that can run the system, skipping refusals.

    OpenCL's CPU device is skipped: it re-times the CPU backend through a
    different runtime. A refused ``arith`` mode prints its reason and its
    row is absent, as the C selection harness does.
    """
    rows: list[dict[str, Any]] = []
    backends = []
    try:
        import hybsol.opencl as opencl

        backends.append(("opencl", opencl, ["auto"]))
    except ImportError:
        print("opencl: built without the backend", file=sys.stderr)
    try:
        import hybsol.hip as hip

        backends.append(("hip", hip, hip_arith))
    except ImportError:
        print("hip: built without the backend", file=sys.stderr)

    for backend_name, module, modes in backends:
        for info in module.devices():
            if info["kind"] == "cpu" or not info["double"]:
                continue
            for mode in modes:
                try:
                    kwargs = {"device": int(info["index"])}
                    if backend_name == "hip":
                        kwargs["arith"] = mode

                    def decompose(m=module, k=kwargs):  # noqa: ANN001, ANN202
                        """Factorize on this device in this arith mode."""
                        return m.decompose(system, **k)

                    factor_s, solve_s = timed_pair(decompose, rhs, rounds)
                    solution = decompose().solve(rhs)
                    rows.append(
                        {
                            "backend": backend_name,
                            "device": int(info["index"]),
                            "name": str(info["name"]),
                            "mode": mode,
                            "threads": None,
                            "decompose_s": factor_s,
                            "solve_s": solve_s,
                            "rel_err": _relative_error(solution, reference),
                        }
                    )
                except (DeviceError, MemoryError) as refusal:
                    print(
                        f"{backend_name} device {info['index']} mode {mode}: {refusal}",
                        file=sys.stderr,
                    )
    return rows


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        default="128:4,256:4,64:16",
        help="Comma list of n_blocks:block_size shapes.",
    )
    parser.add_argument(
        "--precisions",
        default="double,single",
        help="Comma list of factor precisions.",
    )
    parser.add_argument(
        "--rounds", type=int, default=5, help="Timed rounds per measurement."
    )
    parser.add_argument(
        "--cpu-threads",
        default=None,
        help="Comma list of CPU thread counts; defaults to 1 and the physical "
        "core count.",
    )
    parser.add_argument(
        "--hip-arith",
        default="auto",
        help="Comma list of HIP arith modes to time; modes a device cannot "
        "serve print their refusal and are skipped.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print each table as a JSON line."
    )
    return parser.parse_args()


def main() -> None:
    """Run every (shape, precision) table and print it."""
    args = parse_args()
    cores = physical_core_count()
    cpu_threads = (
        [int(t) for t in args.cpu_threads.split(",")] if args.cpu_threads else [1, cores]
    )
    hip_arith = args.hip_arith.split(",")

    for config in args.configs.split(","):
        n_blocks, block_size = (int(x) for x in config.split(":"))
        for precision_name in args.precisions.split(","):
            precision = Precision(precision_name)
            system, _dense = bench_system(n_blocks, block_size, precision)
            total = n_blocks * block_size
            rhs = np.random.default_rng(_RHS_SEED).standard_normal(total)

            reference = system.decompose(n_threads=cores).solve(rhs)
            rows = []
            for threads in cpu_threads:
                factor_s, solve_s = timed_pair(
                    lambda t=threads: system.decompose(n_threads=t), rhs, args.rounds
                )
                solution = system.decompose(n_threads=threads).solve(rhs)
                rows.append(
                    {
                        "backend": "cpu",
                        "device": None,
                        "name": "cpu",
                        "mode": None,
                        "threads": threads,
                        "decompose_s": factor_s,
                        "solve_s": solve_s,
                        "rel_err": _relative_error(solution, reference),
                    }
                )
            rows += device_rows(system, rhs, reference, args.rounds, hip_arith)

            if args.json:
                print(
                    json.dumps(
                        {
                            "n_blocks": n_blocks,
                            "block_size": block_size,
                            "precision": precision_name,
                            "rows": rows,
                        }
                    )
                )
            else:
                print(f"\n{n_blocks} blocks of {block_size}, {precision_name} factors")
                print(
                    f"{'backend':9} {'mode':8} {'thr':>4} {'decompose':>12} "
                    f"{'solve':>12} {'rel_err':>10}  device"
                )
                for row in rows:
                    mode = row["mode"] or "-"
                    thr = "-" if row["threads"] is None else str(row["threads"])
                    print(
                        f"{row['backend']:9} {mode:8} {thr:>4} "
                        f"{row['decompose_s']:12.6f} {row['solve_s']:12.6f} "
                        f"{row['rel_err']:10.2e}  {row['name']}"
                    )


if __name__ == "__main__":
    main()
