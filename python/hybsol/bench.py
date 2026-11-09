"""Micro-benchmarks that rank the available backends on the current machine.

:func:`bench_backends` times one factorize-and-solve round on a seeded,
banded-plus-hub system for every backend this machine can actually run --
the CPU and every device of the optional OpenCL and HIP modules -- and
returns the results sorted by speed. :func:`best_backend` just takes the
winner, which is what application code asks for when it wants "wherever
this box is fastest".

The numbers are micro-benchmark numbers: one system shape, no cache
effects between backends beyond a warmup round, and factorization cost
dominating a single solve. They rank *this machine at this moment*, which
is exactly the question ``best_backend`` answers; they are not portable
constants.
"""

from __future__ import annotations

import statistics
import time
from typing import TYPE_CHECKING

import numpy as np

from hybsol._mod import BlockSystem, DeviceError

if TYPE_CHECKING:
    from hybsol import Precision

__all__ = ["bench_backends", "best_backend"]


def _bench_system(n_blocks: int, block_size: int, precision: str) -> BlockSystem:
    """Build the benchmark system: a block tridiagonal matrix plus a hub row.

    Block row 0 connects to every other block row, so the graph is not just a
    band and the passes have real width. The system is strictly dominant by
    construction -- every diagonal block is shifted past the norms of its row
    -- which is what makes it factorizable without pivoting, in every
    backend; the asserts below are that guarantee, made visible.
    """
    rng = np.random.default_rng(17)
    blocks: dict[tuple[int, int], np.ndarray] = {}
    for i in range(n_blocks):
        for j in range(i + 1):
            if i == j or j == 0 or i - j == 1:
                blocks[(i, j)] = rng.standard_normal((block_size, block_size))

    for i in range(n_blocks):
        off = [
            blocks[(max(i, j), min(i, j))]
            for j in range(n_blocks)
            if j != i and (max(i, j), min(i, j)) in blocks
        ]
        shift = sum(float(np.linalg.norm(b, 2)) for b in off)
        g = blocks[(i, i)]
        row = float(np.abs(g).sum(axis=1).max())
        col = float(np.abs(g).sum(axis=0).max())
        d = g + (shift + row + col + 1.0) * np.eye(block_size)
        off = np.abs(d) - np.diag(np.abs(d).diagonal())
        assert float(np.abs(d).diagonal().min() - off.sum(axis=1).max()) > 0.0
        blocks[(i, i)] = d

    rows: list[int] = []
    cols: list[int] = []
    data: list[np.ndarray] = []
    for (i, j), block in blocks.items():
        rows.append(i)
        cols.append(j)
        data.append(block)
        if i != j:
            rows.append(j)
            cols.append(i)
            data.append(np.ascontiguousarray(block.T))

    system = BlockSystem.from_blocks(
        [block_size] * n_blocks,
        np.array(rows),
        np.array(cols),
        np.concatenate([b.ravel() for b in data]),
        precision=precision,
    )
    assert system.is_valid()
    return system


def _median_round(decompose, rhs: np.ndarray, seconds_budget: float) -> float:  # noqa: ANN001
    """Run factorize-and-solve rounds and return the median wall time.

    One warmup round (untimed, so first-launch module loading is not charged
    to the backend), then at least three and at most thirty rounds, stopping
    once *seconds_budget* of rounds have accumulated.
    """
    decompose().solve(rhs)

    times: list[float] = []
    deadline = time.perf_counter() + seconds_budget
    while True:
        start = time.perf_counter()
        decompose().solve(rhs)
        times.append(time.perf_counter() - start)
        if time.perf_counter() >= deadline or len(times) >= 30:
            break
    while len(times) < 3:
        start = time.perf_counter()
        decompose().solve(rhs)
        times.append(time.perf_counter() - start)
    return float(statistics.median(times))


def bench_backends(
    *,
    n_blocks: int = 256,
    block_size: int = 4,
    precision: Precision | None = None,
    seconds_budget: float = 3.0,
) -> list[dict[str, object]]:
    """Time every backend this machine can run, fastest first.

    One round is a factorization plus one solve of the same seeded system,
    timed with :func:`time.perf_counter`; the reported ``seconds`` is the
    median over warmup-excluded rounds (at least three, at most thirty,
    stopping once ``seconds_budget`` has passed). Backends that cannot run
    the system -- no module, no device, a device that refuses it -- are
    absent from the table rather than ranked last.

    Parameters
    ----------
    n_blocks : int, default: 256
        Number of blocks in the benchmark system.
    block_size : int, default: 4
        Size of every block.
    precision : hybsol.Precision, optional
        Precision of the system and the factors; defaults to ``"double"``.
    seconds_budget : float, default: 3.0
        Approximately how long one backend may be timed.

    Returns
    -------
    list of dict
        One entry per runnable backend, sorted by ``seconds`` ascending,
        each with keys ``backend`` (``"cpu"``, ``"opencl"`` or ``"hip"``),
        ``device`` (the device index, or ``None`` for the CPU), ``name``
        (device name or ``"cpu"``), ``seconds`` and ``total_size`` (the
        matrix dimension).

    Examples
    --------
    >>> rows = bench_backends(n_blocks=4, seconds_budget=0.05)
    >>> print(sorted(rows[0]))
    ['backend', 'device', 'name', 'seconds', 'total_size']
    >>> print(any(row["backend"] == "cpu" for row in rows))
    True
    """
    system_precision = str(precision) if precision is not None else "double"
    system = _bench_system(n_blocks, block_size, system_precision)
    total_size = n_blocks * block_size
    rhs = np.random.default_rng(23095).standard_normal(total_size)

    results: list[tuple[float, dict[str, object]]] = []

    cpu_seconds = _median_round(
        lambda: system.decompose(n_threads=0), rhs, seconds_budget
    )
    results.append(
        (
            cpu_seconds,
            {
                "backend": "cpu",
                "device": None,
                "name": "cpu",
                "seconds": cpu_seconds,
                "total_size": total_size,
            },
        )
    )

    for backend_name, module_name in (("opencl", "hybsol.opencl"), ("hip", "hybsol.hip")):
        try:
            module = __import__(module_name, fromlist=["devices"])
        except ImportError:
            continue
        for entry in module.devices():
            index = int(entry["index"])
            try:
                seconds = _median_round(
                    lambda module=module, index=index: module.decompose(
                        system, device=index
                    ),
                    rhs,
                    seconds_budget,
                )
            except (DeviceError, MemoryError):
                continue
            results.append(
                (
                    seconds,
                    {
                        "backend": backend_name,
                        "device": index,
                        "name": str(entry["name"]),
                        "seconds": seconds,
                        "total_size": total_size,
                    },
                )
            )

    results.sort(key=lambda item: item[0])
    return [row for _, row in results]


def best_backend(**kwargs: object) -> dict[str, object]:
    """Return the fastest runnable backend for the given shape.

    Takes the same keyword arguments as :func:`bench_backends` and returns
    its first entry. Raises :exc:`RuntimeError` only if not even the CPU ran,
    which means the benchmark system itself failed to build or factorize.

    Examples
    --------
    >>> best = best_backend(n_blocks=4, seconds_budget=0.05)
    >>> print(best["backend"] in {"cpu", "opencl", "hip"})
    True
    """
    rows = bench_backends(**kwargs)  # type: ignore[arg-type]
    if not rows:
        raise RuntimeError("no backend ran the benchmark system")
    return rows[0]
