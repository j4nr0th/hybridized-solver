"""Prove the Python layer never reaches a C precondition assert.

Every precondition in the core is enforced with ``CUTL_ASSERT``, which aborts
the process. That is the right trade for a C caller and the wrong one for an
extension module, so the bindings must reject every bad input themselves and
raise. This extension is built with those checks *live* -- see
``cmake.define.CUTL_ASSERTS`` in ``pyproject.toml`` -- precisely so that a
validation the bindings forget shows up here as a crash rather than as silent
undefined behaviour in somebody's interpreter.

Running a hostile call in-process would take the test runner down with it, so
every case runs in a worker subprocess. The worker announces each case on
stdout before running it, which is what makes an abort pinpointable: the last
``CASE`` line printed is the one that died.

The cases come in two flavours. Most are *read-only*: they hand a hostile
argument to a method and expect an exception. Some are *mutating* -- they aim at
``add_block``, ``reserve``, ``reorder_blocks``, ``eliminate_row`` and friends,
which could leave a system half-written if the binding accepted something it
should not have. After each mutating case the worker checks the target system
still holds exactly the values it did before, so a partially applied write is a
failure even when nothing crashed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# Indices and scalars that no binding should accept for an index parameter.
_BAD_SCALARS: list[tuple[str, object]] = [
    ("negative", -1),
    ("negative_far", -(2**40)),
    ("at_n", 4),
    ("past_n", 99),
    ("huge_int", 2**62),
    ("uint64_max", 2**64 - 1),
    ("float", 1.5),
    ("str", "0"),
    ("none", None),
    ("true", True),
    ("np_int64", np.int64(-1)),
    ("np_uint64_max", np.uint64(2**64 - 1)),
    ("list", [0]),
    ("complex", 1 + 0j),
]


# Arrays that are the wrong shape, type, layout or value for a double system.
def _read_only() -> np.ndarray:
    """Build a read-only view, which every writing parameter must refuse."""
    arr = np.zeros(4)
    arr.flags.writeable = False
    return arr


_BAD_ARRAYS: list[tuple[str, object]] = [
    ("scalar_0d", np.array(1.0)),
    ("empty", np.array([])),
    ("wrong_len_2", np.zeros(2)),
    ("wrong_len_8", np.zeros(8)),
    ("two_d_1x1", np.zeros((1, 1))),
    ("two_d_2x3", np.zeros((2, 3))),
    ("three_d", np.zeros((2, 2, 2))),
    ("float32", np.zeros(4, dtype=np.float32)),
    ("int64", np.zeros(4, dtype=np.int64)),
    ("complex128", np.zeros(4, dtype=np.complex128)),
    ("bool", np.zeros(4, dtype=bool)),
    ("str", np.array(["a", "b", "c", "d"])),
    ("object", np.array([None] * 4, dtype=object)),
    ("nan", np.full(4, np.nan)),
    ("inf", np.full(4, np.inf)),
    ("huge", np.full(4, 1e308)),
    ("non_contiguous_stride2", np.zeros(8)[::2]),
    ("non_contiguous_reversed", np.zeros(4)[::-1]),
    ("non_contiguous_transposed", np.zeros((4, 4)).T),
    ("read_only", _read_only()),
    ("python_list", [1.0, 2.0, 3.0, 4.0]),
    ("python_tuple", (1.0, 2.0)),
    ("none", None),
    ("string", "not an array"),
    ("dict", {"a": 1}),
    ("ragged", [[1.0, 2.0], [3.0]]),
]


def _make_system(precision: str = "double"):
    """Build a small, well-conditioned, structurally valid 3-block system."""
    from hybsol import BlockSystem

    b = np.array([[2.0, 1.0], [1.0, 2.0]])
    return BlockSystem.from_blocks(
        [2, 2, 2],
        [0, 0, 1, 1, 2],
        [0, 1, 0, 1, 2],
        np.concatenate(
            [b.ravel(), b.ravel(), b.ravel(), (b * 2).ravel(), (b * 3).ravel()]
        ),
        precision=precision,
    )


# --------------------------------------------------------------------------
# Case construction
# --------------------------------------------------------------------------


def _read_only_cases():
    """Hostile arguments to methods that only inspect or return data."""
    for name, value in _BAD_SCALARS:
        yield (
            f"get_block_size/{name}",
            lambda v=value: _make_system().get_block_size(v, 0),
        )
        yield (
            f"get_row_block_indices/{name}",
            lambda v=value: _make_system().get_row_block_indices(v),
        )
        yield f"first_column/{name}", lambda v=value: _make_system().first_column(v)
        yield (
            f"get_next_column_index/{name}",
            lambda v=value: _make_system().get_next_column_index(v, 0),
        )
        yield f"has_block/{name}", lambda v=value: _make_system().has_block(v, 0)
        yield f"get_block/{name}", lambda v=value: _make_system().get_block(v, 0)
        yield f"block_storage/{name}", lambda v=value: _make_system().block_storage(v, 0)

    for name, value in _BAD_ARRAYS:
        yield f"add_block/{name}", lambda v=value: _make_system().add_block(0, 0, v)
        yield f"matvec/{name}", lambda v=value: _make_system().matvec(v)
        yield f"multiply_row/{name}", lambda v=value: _make_system().multiply_row(0, v)
        yield (
            f"eliminate_row/{name}",
            lambda v=value: _make_system().eliminate_row(0, 1, v),
        )
        yield (
            f"decompose_workspace/{name}",
            lambda v=value: _make_system().decompose(workspace=v),
        )
        yield f"reorder_blocks/{name}", lambda v=value: _make_system().reorder_blocks(v)
        yield (
            f"reorder_vector/{name}",
            lambda v=value: _make_system().reorder_vector(np.arange(3), v),
        )

    # Block payloads of the wrong shape for a 2x2 block.
    for name, value in [
        ("1x1", np.ones((1, 1))),
        ("2x3", np.ones((2, 3))),
        ("3x2", np.ones((3, 2))),
        ("1d_len4", np.ones(4)),
        ("1d_len3", np.ones(3)),
        ("1d_len8", np.ones(8)),
    ]:
        yield f"add_block_shape/{name}", lambda v=value: _make_system().add_block(0, 0, v)

    # Thread counts. Only values the library can reject on inspection: a request
    # for more threads than the machine has cores is deliberately honoured, so
    # an absurd one is the caller's to live with rather than something this file
    # may feed to the interpreter.
    for name, value in [
        ("negative", -1),
        ("str", "2"),
        ("none", None),
        ("float", 1.5),
    ]:
        yield (
            f"decompose_threads/{name}",
            lambda v=value: _make_system().decompose(n_threads=v),
        )
        yield f"workspace_bytes/{name}", lambda v=value: _make_system().workspace_bytes(v)
        yield (
            f"matvec_threads/{name}",
            lambda v=value: _make_system().matvec(np.ones(6), n_threads=v),
        )
        yield (
            f"reorder_threads/{name}",
            lambda v=value: _make_system().reorder_blocks(np.arange(3), n_threads=v),
        )

    # Constructor spellings.
    from hybsol import BlockSystem

    for name, thunk in [
        ("no_args", lambda: BlockSystem()),
        ("zero", lambda: BlockSystem(0)),
        ("negative", lambda: BlockSystem(-1)),
        ("zero_in_sizes", lambda: BlockSystem(0, 2)),
        ("negative_in_sizes", lambda: BlockSystem(2, -3)),
        ("empty_sizes", lambda: BlockSystem(*[])),
        ("str_sizes", lambda: BlockSystem("a", "b")),
        ("float_sizes", lambda: BlockSystem(1.5, 2.5)),
        ("none_sizes", lambda: BlockSystem(None, 1)),
        ("bad_precision", lambda: BlockSystem(2, 2, precision="quad")),
        ("none_precision", lambda: BlockSystem(2, 2, precision=None)),
        ("int_precision", lambda: BlockSystem(2, 2, precision=7)),
        ("too_many_positional", lambda: BlockSystem(2, 2, 2)),
    ]:
        yield f"ctor/{name}", thunk

    for name, thunk in [
        ("short_data", lambda: BlockSystem.from_blocks([2, 2], [0], [0], [1.0])),
        ("long_data", lambda: BlockSystem.from_blocks([2, 2], [0], [0], [1.0] * 9)),
        (
            "rows_longer_than_data",
            lambda: BlockSystem.from_blocks([2, 2], [0, 1], [0, 1], [1.0]),
        ),
        ("negative_index", lambda: BlockSystem.from_blocks([2, 2], [-1], [0], [1.0] * 4)),
        ("out_of_range", lambda: BlockSystem.from_blocks([2, 2], [5], [0], [1.0] * 4)),
        ("empty_everything", lambda: BlockSystem.from_blocks([], [], [], [])),
        ("none_args", lambda: BlockSystem.from_blocks(None, None, None, None)),
        ("asymmetric", lambda: BlockSystem.from_blocks([2, 2], [0], [1], [1.0] * 4)),
        (
            "no_diagonal",
            lambda: BlockSystem.from_blocks([2, 2], [0, 1], [0, 1], [1.0] * 8),
        ),
        (
            "ragged_data",
            lambda: BlockSystem.from_blocks([2, 2], [0], [0], [[1.0, 2.0], [3.0]]),
        ),
        ("str_data", lambda: BlockSystem.from_blocks([2, 2], [0], [0], ["a"] * 4)),
    ]:
        yield f"from_blocks/{name}", thunk

    for name, thunk in [
        ("bad_strategy", lambda: _make_system().compute_reordering("nonsense")),
        ("none_strategy", lambda: _make_system().compute_reordering(None)),
        ("negative_max_colors", lambda: _make_system().compute_reordering("first", -1)),
        ("strategy_arg", lambda: _make_system().compute_reordering("first", "x")),
    ]:
        yield f"reordering/{name}", thunk

    # Decomposition and elimination accessors.
    yield (
        "elimination/row_columns_negative",
        lambda: _make_system().elimination().row_columns(-1),
    )
    yield (
        "elimination/row_columns_past_n",
        lambda: _make_system().elimination().row_columns(99),
    )
    yield (
        "elimination/row_columns_str",
        lambda: _make_system().elimination().row_columns("0"),
    )
    yield (
        "elimination/level_rows_negative",
        lambda: _make_system().elimination().level_rows(-1),
    )
    yield (
        "elimination/level_rows_past_n",
        lambda: _make_system().elimination().level_rows(99),
    )
    yield (
        "elimination/row_level_negative",
        lambda: _make_system().elimination().row_level(-1),
    )
    yield (
        "elimination/row_first_level_past_n",
        lambda: _make_system().elimination().row_first_level(99),
    )
    yield (
        "elimination/row_n_elim_past_n",
        lambda: _make_system().elimination().row_n_eliminations(99),
    )

    # Methods taking arguments that take none.
    for name in [
        "is_valid",
        "as_array",
        "copy",
        "elimination",
        "block_sizes",
        "no_lower_connections",
    ]:
        yield f"noargs/{name}", lambda n=name: getattr(_make_system(), n)(1)
    yield "operations/takes_arg", lambda: _make_system().decompose().operations(1)

    # Solving with the wrong right-hand side.
    for name, value in _BAD_ARRAYS:
        yield f"solve/{name}", lambda v=value: _make_system().decompose().solve(v)
    yield (
        "solve/out_wrong_len",
        lambda: _make_system().decompose().solve(np.ones(6), out=np.zeros(3)),
    )
    yield (
        "solve/out_read_only",
        lambda: _make_system().decompose().solve(np.ones(6), out=_read_only()),
    )
    yield (
        "solve/out_float32",
        lambda: (
            _make_system()
            .decompose()
            .solve(np.ones(6), out=np.zeros(6, dtype=np.float32))
        ),
    )
    yield (
        "solve/threads_negative",
        lambda: _make_system().decompose().solve(np.ones(6), n_threads=-1),
    )


def _mutating_cases():
    """Hostile calls aimed at methods that write to the system."""
    for name, value in _BAD_SCALARS:
        yield f"add_block_row/{name}", lambda v=value: _S.add_block(v, 0, np.ones((2, 2)))
        yield f"add_block_col/{name}", lambda v=value: _S.add_block(0, v, np.ones((2, 2)))
        yield f"reserve/{name}", lambda v=value: _S.reserve(v, 4)
        yield f"reserve_capacity/{name}", lambda v=value: _S.reserve(0, v)
        yield f"multiply_row_idx/{name}", lambda v=value: _S.multiply_row(v, np.eye(2))
        yield (
            f"eliminate_row_idx/{name}",
            lambda v=value: _S.eliminate_row(v, 1, np.eye(2)),
        )

    for name, value in _BAD_ARRAYS:
        yield f"add_block_badval/{name}", lambda v=value: _S.add_block(1, 1, v)
        yield f"add_blocks_rows/{name}", lambda v=value: _S.add_blocks(v, [1], [1.0] * 4)
        yield f"add_blocks_data/{name}", lambda v=value: _S.add_blocks([1], [1], v)
        yield f"reorder_blocks_idx/{name}", lambda v=value: _S.reorder_blocks(v)
        yield f"eliminate_row_badval/{name}", lambda v=value: _S.eliminate_row(0, 1, v)
        yield (
            f"reorder_vector_idx/{name}",
            lambda v=value: _S.reorder_vector(v, np.ones(6)),
        )
        yield (
            f"unorder_vector_idx/{name}",
            lambda v=value: _S.unorder_vector(v, np.ones(6)),
        )

    # Permutations that are not permutations.
    for name, order in [
        ("duplicate", [0, 0, 1]),
        ("missing", [0, 1]),
        ("too_long", [0, 1, 2, 3]),
        ("out_of_range", [0, 1, 7]),
        ("negative", [0, 1, -1]),
        ("float", [0.0, 1.0, 2.0]),
    ]:
        yield f"reorder_not_perm/{name}", lambda o=order: _S.reorder_blocks(np.asarray(o))
        yield (
            f"reorder_vec_not_perm/{name}",
            lambda o=order: _S.reorder_vector(np.asarray(o), np.ones(6)),
        )

    # Eliminating a row with itself, and with an absent coupling.
    yield "eliminate_row/self", lambda: _S.eliminate_row(0, 0, np.eye(2))
    yield "eliminate_row/uncoupled", lambda: _S.eliminate_row(2, 0, np.eye(2))
    yield "eliminate_row/bad_shape", lambda: _S.eliminate_row(0, 1, np.ones((3, 3)))
    yield "add_block/bad_shape", lambda: _S.add_block(0, 0, np.ones((3, 3)))


# A system the mutating cases aim at. Rebuilt by the worker between batches so a
# case that somehow corrupts it cannot mask the next one.
_S = None  # type: ignore[assignment]


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------


def _canary() -> None:
    """Build, factorize and solve a known system; raises if anything is off."""
    s = _make_system()
    d = s.decompose()
    rhs = np.arange(1.0, 7.0)
    got = d.solve(rhs)
    want = np.linalg.solve(s.as_array(), rhs)
    assert np.allclose(got, want, atol=1e-10), f"{got} vs {want}"


def _worker() -> int:
    """Run every case, announcing each one first. Returns a process exit code."""
    global _S

    raised = 0
    accepted = 0
    mutated_after_raise = 0
    total = 0

    for _label, thunk in _read_only_cases():
        print(f"CASE {_label}", flush=True)
        total += 1
        try:
            thunk()
            accepted += 1
            print("OK", flush=True)
        except BaseException as exc:  # noqa: BLE001 - any raise is acceptable here
            raised += 1
            print(f"RAISE {type(exc).__name__}", flush=True)

    for _label, thunk in _mutating_cases():
        print(f"CASE {_label}", flush=True)
        total += 1
        _S = _make_system()
        golden = _S.as_array().copy()
        rejected = False
        try:
            thunk()
            accepted += 1
            print("OK", flush=True)
        except BaseException as exc:  # noqa: BLE001
            rejected = True
            raised += 1
            print(f"RAISE {type(exc).__name__}", flush=True)

        # A call that was accepted may legitimately have written (storing NaN is
        # valid input). A call that was *rejected* must have written nothing at
        # all -- a partially applied block insert or a reordered row set is the
        # failure this is here to catch.
        if rejected and not np.array_equal(_S.as_array(), golden):
            mutated_after_raise += 1
            print("MUTATED-AFTER-RAISE", flush=True)
        _canary()

    print(
        f"STATS total={total} raised={raised} accepted={accepted} "
        f"mutated_after_raise={mutated_after_raise}",
        flush=True,
    )
    print("WORKER-CLEAN", flush=True)
    return 0


# --------------------------------------------------------------------------
# Test
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def worker():
    """Run the hostile-input worker once and hand back its outcome."""
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker"],
        capture_output=True,
        text=True,
        timeout=900,
    )
    return proc


def _last_case(stdout: str) -> str:
    cases = [line[5:] for line in stdout.splitlines() if line.startswith("CASE ")]
    return cases[-1] if cases else "<none>"


def test_hostile_input_never_aborts(worker):
    """Every hostile call either raises or succeeds -- never a C assert.

    An abort kills the worker with SIGABRT and a bad access with SIGSEGV; both
    show up here as a negative return code with no ``WORKER-CLEAN``.
    """
    if worker.returncode != 0 or "WORKER-CLEAN" not in worker.stdout:
        pytest.fail(
            f"The worker died (return code {worker.returncode}) on case "
            f"{_last_case(worker.stdout)!r}.\n"
            f"--- stdout tail ---\n{worker.stdout[-3000:]}\n"
            f"--- stderr ---\n{worker.stderr[-3000:]}"
        )


def _stats(worker) -> dict[str, int]:
    stats = [line for line in worker.stdout.splitlines() if line.startswith("STATS ")]
    assert stats, "the worker produced no statistics"
    return {k: int(v) for k, v in (kv.split("=") for kv in stats[-1].split()[1:])}


def test_some_hostile_input_is_actually_rejected(worker):
    """Guard against a fuzzer that passes because nothing was hostile."""
    s = _stats(worker)
    assert s["total"] > 400, f"only {s['total']} cases ran; the generator has shrunk"
    assert s["raised"] > s["total"] // 2, (
        f"only {s['raised']} of {s['total']} hostile inputs were rejected; most "
        "are being accepted, so either they are not hostile or they are unchecked"
    )


def test_a_rejected_call_never_writes(worker):
    """A refused mutation must leave the system byte-for-byte as it was.

    This is the failure a bound check alone does not catch: the binding rejects
    the argument, but only after having resized a row or reordered the system.
    """
    assert _stats(worker)["mutated_after_raise"] == 0, (
        "a call that raised still left the system modified, so the check ran "
        "after a partial write"
    )


def test_asserts_are_compiled_in():
    """The extension must actually have the precondition checks live.

    Without them this file would be testing nothing: a missed validation would
    read out of bounds rather than abort, which no test in this file could see.
    """
    import hybsol._mod

    binary = Path(hybsol._mod.__file__).read_bytes()
    assert b"Assertion failed" in binary, (
        "the installed extension has no CUTL_ASSERT strings compiled in. Check "
        "cmake.define.CUTL_ASSERTS in pyproject.toml, and reinstall the package "
        "with `uv sync --reinstall-package hybsol`."
    )
    assert b"is outside [0" in binary, (
        "the index-range assertion is missing from the installed extension"
    )


def test_interpreter_shuts_down_cleanly():
    """Exiting must not crash, however the module state was left.

    This guards a class of bug nothing running inside the interpreter can see:
    the extension releasing, at finalization, an object the interpreter also
    releases. The reference count then reaches zero early and shutdown itself
    walks freed memory.

    It is intermittent by nature, so the detection needs a workload that
    actually recycles the freed block. The sparse comparison example does: a
    dense-ish 300-block system, a sparse LU alongside it, and a reordering
    decomposition. A small script is not enough -- with the double release
    present, a toy system exits cleanly 20 times out of 20, while this one
    crashes about half the time. Hence the loop, and hence the cost.
    """
    example = Path(__file__).resolve().parent.parent / "examples" / "compare_scipy.py"
    if not example.is_file():
        pytest.skip("examples/compare_scipy.py is not present in this checkout")

    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    runs = 10
    crashes = []
    for _ in range(runs):
        proc = subprocess.run(
            [sys.executable, str(example)],
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
        )
        if proc.returncode != 0:
            crashes.append((proc.returncode, proc.stderr[-800:]))

    assert not crashes, (
        f"{len(crashes)} of {runs} interpreters crashed on exit; the first was "
        f"return code {crashes[0][0]}:\n{crashes[0][1]}"
    )


if __name__ == "__main__":
    if "--worker" in sys.argv:
        sys.exit(_worker())
    raise SystemExit("run this through pytest")
