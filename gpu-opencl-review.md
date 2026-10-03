# gpu/opencl — review notes

Six commits on `gpu/opencl` (from `dev`), each green on its own. Everything
below is for you to read later; nothing here needs an answer to be usable.

## What landed

| commit | what |
| --- | --- |
| `de71d9d` | mixed-precision decompositions (core, CPU) |
| `96d74d6` | the backend contract: vtable, frame helpers, device result codes, `hybsol.DeviceError`, the cross-module capsule |
| `5959393` | the OpenCL backend: `hybsol::opencl`, kernels, device-resident payload, `test_gpu` |
| `f0bd897` | `hybsol.opencl`: the Python plugin module and 17 tests |
| `4578e56`, `59cace5` | docs; sanitizer note |

Use it as:

```python
import hybsol.opencl as ocl  # the whole opt-in

system = hybsol.BlockSystem(...)  # assembled as usual
dec = ocl.decompose(system)  # factors stay on the GPU
x = dec.solve(b)  # same Decomposition as always
```

`import hybsol` loads no OpenCL. A build with `HYBSOL_ENABLE_OPENCL=OFF` has
no `hybsol.opencl` at all — the import fails, nothing else changes.

## Verification

- **C**: 5 configurations (Debug, asserts-off, OpenMP-off, **OpenCL-off**,
  Release) — all green. With OpenCL off there is no `libhybsol_opencl.a` and
  no `test_gpu` target, which is the point.
- **Python**: 378 tests pass (359 before this work + 19 new).
- **Real device**: `test_gpu` runs 194 checks on the Quadro P2000 — parity with
  the CPU in both precisions, both mixed-precision directions, the same failing
  block on a singular diagonal, replay equal to solve, device lifetime.
- **Docs**: build quieter than before (six pre-existing errors gone, none added).
- **Sanitizers**: `test_gpu` passes all checks; the leaks LeakSanitizer
  reports are the vendor's (CUDA loader, Intel runtime), now documented.

## Findings you may want to act on

1. **Performance at the target size: still a loss, and now I know why.**
   Measured at your target -- single precision, ~100k dof, this machine's 12
   CPU threads against the Quadro P2000 -- on a layered block system (the
   shape a hybridized discretization has: weak couplings, a schedule with real
   parallelism):

   | system | dof | CPU factorize | OpenCL factorize | CPU solve | OpenCL solve |
   | --- | --- | --- | --- | --- | --- |
   | 4x4 blocks, 24389 of them | 97,556 | 0.82 s | 2.5 s | 66 ms | 43 ms |
   | 16x16 blocks, 6859 | 109,744 | -- | 22.9 s | -- | 111 ms |
   | 32x32 blocks, 3375 | 108,000 | -- | 78.7 s | -- | 232 ms |

   Two things came out of this round, one fix and one diagnosis.

   **The fix**: a pass is no longer waited on. The failure slot is cleared once
   and read once, at the end, instead of a blocking round trip after every
   pass, and passes are launched over a bounded work group instead of the
   driver's choice (its maximum, 8192 here, which put a few hundred rows of a
   pass on one compute unit). Factorization at 97k dof went from **7.9 s to
   2.5 s** -- a third of what it was, and the read-back semantics are
   unchanged: the same block is named.

   **The diagnosis**: larger blocks make it *worse*, which is the opposite of
   "more arithmetic, more GPU". The reason is one work item per block row:
   a bigger block means one work item grinds more of the dense work serially,
   while the rest of the device has nothing else to do. At 4x4 blocks the GPU
   is latency-bound but the CPU is not far behind (0.95 ms per pass on twelve
   threads, against 3 ms per pass on the device) -- there simply is not enough
   arithmetic in a 4x4 factorization to move.

   So the next step, if there is one, is not more scheduling tricks: it is
   giving the device parallelism *inside* a row -- one work item per
   (row, block) pair instead of per row, with the subtractions combined
   through a reduction or atomics, or a two-stage product-then-subtract. That
   is a kernel redesign, not a tuning pass, and I have not done it.

   Back substitution, which you asked about, is on the device already and costs
   **18-33 ms of the ~2.5 s** (measured separately: the solve is 43-79 ms of
   which the back substitution is under half). There is nothing to gain by
   moving it, and moving it to the CPU would mean copying the factors back.
   Left as it is.

2. **Pre-existing UBSan findings**, unrelated to this work and present on `dev`
   too (verified in a clean worktree): `src/core/ordering.c:318-378` stores
   `hybsol_row_entry_t *` into addresses that are not 8-byte aligned (frame
   carving after odd-sized arrays), plus a LeakSanitizer report. Worth a
   follow-up issue; I left it alone.

3. **NVIDIA rejects `-cl-std=CL1.2`** (`CL_INVALID_BUILD_OPTIONS`), so the
   build passes no options and lets the runtime use its newest language. The
   kernels are 1.2-compatible; Apple's OpenCL never went past 1.2 either.

## Decisions I made without asking (reversible, but tell me if you disagree)

- **Vectors are double on the device even for single factors**, mirroring the
  CPU's narrowed kernels: `vec_multiply_sub` accumulates float, `vec_lu_solve`
  accumulates double. Your "same as CPU part" answer, implemented literally.
  The consequence: a device decomposition needs FP64 at all, and a
  single-precision *system* on a single-precision-only device would be
  refused. Every real OpenCL device I know has `cl_khr_fp64`.
- **GPU-first enumeration**: device 0 is a GPU wherever the machine has one,
  because index 0 is what a caller reaches for first. A CPU device is still a
  perfectly good target and appears in the list.
- **A singular diagonal aborts the factorization at the end of the pass** that
  failed, exactly where the CPU settles its per-row results — not at once.
- **The pattern moved into the host frame** (`row_entry_offset` + `cols`,
  `n_columns` extra uint64s) so that `operations()` and every other host-side
  query works on a decomposition whose factors are not in host memory.

## Your four answers, and what came of them

1. **Target: beat the CPU on large sparse systems, 100k+ dof, FP32.** Taken as
   the bar. The latency work is done (above); the bar is not met yet, and the
   diagnosis above says what it would take. If the answer is "keep going", the
   per-entry work items are the next piece, not more scheduling.
2. **Back substitution: put it on the device if it is faster there.** It is on
   the device, and at 18-33 ms of a ~2.5 s factorization there is no faster
   option to move to -- the alternative is copying every factor back to the
   host, which is exactly what the design avoids. Left as it is.
3. **Stable device index.** Done: the enumeration is sorted (GPUs first, then
   by name) and `device_info` carries `uuid`, the runtime's own identifier, for
   pinning a device across reinstalls. `test_gpu` checks both, and that an
   index means the same device across calls.
4. **The guarded doctest is fine as it is.** Unchanged.

## Things worth knowing about the code

- `src/opencl/gpu_kernels.h` is one string literal, compiled twice with a `-D`
  flag; the arithmetic is a transcription of `matrix.inc` and
  `decomposition_numeric.inc`. If you change one, change both — the
  accumulation types are the contract, and the tests only compare to tolerances.
- `_mod` and `_mod_opencl` each statically link `hybsol_core`. Two copies of
  the code, no shared state (allocators are per-object), and it is what lets
  the backend be unloaded with the import.
- The plugin reaches the bindings' helpers through a capsule
  (`hybsol._mod.backend_api`). If a future backend (CUDA) needs more helpers,
  that struct is the place to add them.
