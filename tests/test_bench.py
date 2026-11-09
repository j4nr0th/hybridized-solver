"""Ranking the backends at runtime.

The benchmark is the product feature here: an application that does not know
what hardware it lands on asks :func:`hybsol.best_backend` and gets a place
to factorize. These tests keep the table's shape honest at a size that runs
in milliseconds -- and on machines with no GPU at all, which most test
machines are.
"""

import pytest
from hybsol.bench import bench_backends, best_backend

_KEYS = {"backend", "device", "name", "seconds", "total_size"}


def test_the_table_has_cpu_and_is_sorted():
    """Every runnable backend appears, with the exact keys, slowest last."""
    rows = bench_backends(n_blocks=8, seconds_budget=0.05)
    assert rows, "the CPU is always runnable"
    assert all(set(row) == _KEYS for row in rows)
    seconds = [row["seconds"] for row in rows]
    assert seconds == sorted(seconds)
    assert all(row["seconds"] > 0.0 for row in rows)

    cpu = [row for row in rows if row["backend"] == "cpu"]
    assert len(cpu) == 1
    assert cpu[0]["device"] is None
    assert cpu[0]["name"] == "cpu"
    assert cpu[0]["total_size"] == 8 * 4

    for row in rows:
        if row["backend"] != "cpu":
            assert isinstance(row["device"], int)
            assert row["name"] != "cpu"


def test_the_best_entry_is_the_first_row():
    """``best_backend`` returns the fastest row of the same table."""
    rows = bench_backends(n_blocks=4, seconds_budget=0.05)
    best = best_backend(n_blocks=4, seconds_budget=0.05)
    assert (best["backend"], best["device"]) == (rows[0]["backend"], rows[0]["device"])
    assert best["backend"] in {"cpu", "opencl", "hip"}


@pytest.mark.parametrize("precision", ["double", "single"])
def test_single_and_double_systems_both_rank(precision):
    """A single-precision system runs the CPU path at least, in any build."""
    rows = bench_backends(n_blocks=6, precision=precision, seconds_budget=0.05)
    assert any(row["backend"] == "cpu" for row in rows)
