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

1. **Performance today is a loss, not a win.** Measured on this machine
   (Quadro P2000, one thread on the CPU for a fair-ish comparison), factorize +
   solve on random sparse block systems:

   | blocks | dim | precision | CPU (1 thread) | OpenCL | |
   | --- | --- | --- | --- | --- | --- |
   | 64 | 269 | double | 38 ms | 170 ms | 0.22x |
   | 128 | 506 | double | 54 ms | 498 ms | 0.11x |
   | 256 | 1025 | double | 423 ms | 2267 ms | 0.19x |

   The arithmetic is not the problem — the *schedule* is. One kernel launch per
   pass plus a blocking 4-byte read-back per pass, and these systems have
   hundreds of passes of very little work each, so it is all latency. On a
   large enough system (thousands of DOF with real blocks) the balance flips,
   but I have not demonstrated that here. If you want speed on today's sizes,
   the levers are, in order of value: (a) make the failure-slot read-back
   asynchronous (only the last pass really needs it before returning), (b) fuse
   consecutive passes that have no dependency between them, (c) persistent
   kernels with a device-side barrier. None of that changes the arithmetic, so
   none of it was in this change — say the word if you want it.

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

## Open questions for you (none blocking)

1. **Performance target.** Is this backend meant to win on *large* systems
   (where I expect it to) or on laptop-sized ones? The answer decides whether
   the async-readback/persistent-kernel work is worth doing now.
2. **Back substitution on the device** is a single work item walking the rows
   in reverse — the same serial loop the CPU runs, on a GPU core. It is
   correct and it is slow. Fusing it into the forward kernel's last pass, or
   accepting it, is your call.
3. **Device index stability.** Enumeration is cached per process and GPUs sort
   first, so index 0 is stable for a given installation but not guaranteed by
   the runtime. Should `hybsol.opencl.devices()` expose something more stable
   (a UUID, a name-based selector) for multi-GPU machines?
4. **`--doctest-modules`** runs over `python/hybsol`, and the OpenCL example in
   `hybsol.opencl.decompose` is guarded by `if devices()`. If you prefer the
   docs to fail loudly on a machine with no device instead, say so and I will
   make it a plain comment.

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
