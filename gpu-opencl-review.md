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
| `af0575c`, `9eebafe` | passes queued instead of waited on; device order and uuid |
| `3a21971` | the kernel redesign: work inside a block row |

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

1. **Performance: measured, diagnosed, redesigned.** The honest sequence:

   *Where the time goes* (97k dof, 4x4 blocks, single precision, compiled one
   phase at a time): 1.2 s total, of which **0.74 s elimination steps, 0.32 s
   diagonal row scaling**, and the rest diagonal LU plus launch overhead. Your
   guess about the inversion was half right -- it is a quarter, and the
   elimination is the bigger half.

   *Why larger blocks made it worse*: cost per operation grew about 900x from
   4x4 to 32x32 blocks. A work item was a whole block row, so a bigger block
   meant one work item grinding more dense work serially while the rest of the
   device idled. The row was the wrong unit of work.

   *What changed*: a pass is now three launches over a two-dimensional grid
   whose second dimension is a row's entries -- one work item per (block row,
   source block) for the eliminations, one per row for the diagonal LU, one per
   (block row, block above the diagonal) for the scaling. No atomics: each
   source entry owns its destination. The arithmetic is untouched; the
   accumulation types are the same transcription as before.

   | system | dof | before | after | gain |
   | --- | --- | --- | --- | --- |
   | 4x4 blocks, 24389 | 97,556 | 1.20 s | 1.05 s | 1.15x |
   | 16x16 blocks, 6859 | 109,744 | 22.9 s | 9.2 s | 2.5x |
   | 32x32 blocks, 3375 | 108,000 | 78.7 s | 11.3 s | **7.0x** |

   The gain grows with the block size, which is the point: the small-block case
   has little to split (one or two sources per row, 64 flops each), and there
   the per-operation latency still dominates at about 1.5 us.

   *What did not work*: interleaving the two pattern-metadata arrays
   (`entry_col` and `entry_val_offset`) into one 16-byte record, to halve the
   dependent loads on the critical path. Measured 1.066 s and 9.235 s against
   1.048 s and 9.211 s -- nothing, inside the noise. Reverted: the arrays were
   already cache-friendly, and two plain arrays are the simpler code.

   *Where this leaves the target*: beating twelve CPU threads at 100k+ dof is
   still not there for 4x4 blocks (CPU 0.82 s, device 1.05 s). For 32x32
   blocks the device is in the same league as before the redesign, which is a
   much better place to be than 7x behind. The remaining lever for small
   blocks is not more parallelism inside a row -- there is none left to take --
   but a different decomposition of the dense work (a proper blocked GEMM per
   block rather than a per-entry product), which is a larger piece again.

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
