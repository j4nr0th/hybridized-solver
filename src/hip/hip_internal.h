/**
 * @file src/hip/hip_internal.h
 * What the HIP backend's two translation units share: the device handle, the
 * kernel entry points and the error note.
 *
 * Private to ``src/hip``.
 *
 * Unlike the OpenCL backend there is nothing to compile at run time: the
 * kernels live in ``hip_kernels.cpp``, AOT-compiled into the library for the
 * architectures CMake asked for. The host reaches them through the C runtime
 * API only -- ``hipLaunchKernel`` and friends -- so this stays C17.
 */

#pragma once

#include <hybsol/hip.h>

#include <hip/hip_runtime_api.h>

/**
 * An acquired device: everything a decomposition on it needs, behind one
 * reference count.
 *
 * Two acquisitions of the same index share one pointer, so the stream is
 * opened once. The kernels need no handle: their entry points are the
 * ``extern`` stubs declared below.
 */
struct hybsol_hip_device
{
    /** Index this device has in :c:func:`hybsol_hip_device_count`. */
    uint64_t index;
    /** The runtime's own device, its ordinal -- equal to ``index`` here. */
    hipDevice_t id;
    /** One stream: kernels and transfers see each other in order. */
    hipStream_t stream;
    /** Non-zero when the device does double arithmetic. */
    int supports_double;
    /** Acquisitions not yet released. */
    unsigned refs;
    /**
     * The allocator the acquisition handed in, kept for the handle's own
     * allocations; valid until the last release, as the header requires.
     */
    const hybsol_allocator_t *allocator;
};

/*
 * The kernel stubs hip_kernels.cpp compiled: one entry per kernel per factor
 * precision. hip_kernels.cpp is built twice from one source, the double pass
 * under -DHYBSOL_HIP_DOUBLE, which is what pastes the _f32/_f64 suffixes.
 * Declared as taking nothing: the host never calls them directly, it hands
 * their addresses to hipLaunchKernel and the argument list travels in a
 * void *[] that mirrors the kernel's parameters.
 */
extern void hybsol_eliminate_pass_f32(void);
extern void hybsol_eliminate_pass_f64(void);
extern void hybsol_diagonal_pass_f32(void);
extern void hybsol_diagonal_pass_f64(void);
extern void hybsol_scale_pass_f32(void);
extern void hybsol_scale_pass_f64(void);
extern void hybsol_forward_pass_f32(void);
extern void hybsol_forward_pass_f64(void);
extern void hybsol_back_substitute_f32(void);
extern void hybsol_back_substitute_f64(void);
extern void hybsol_apply_ops_f32(void);
extern void hybsol_apply_ops_f64(void);

/*
 * The Ozaki sets: the eliminate and scale entry points again, suffixed
 * _f64_ozi8 and _f64_oz2 -- the Ozaki-I realization the selection harness
 * kept, and the Ozaki-II (asymmetric-slicing) mode -- compiled from the
 * same source under -DHYBSOL_HIP_DOUBLE with -DHYBSOL_HIP_OZ_I8 (signed
 * int8 slices, int32 accumulate) or -DHYBSOL_HIP_OZ2 (FP16 slices against
 * a single FP16 window, FP32 accumulate). The eliminate entry point takes
 * exactly the arguments of its _f64 twin; the scaling one takes three more,
 * a scratch region, its row stride and its per-work-item stride, which is
 * where the Neumann panel inverts the diagonal -- so the host passes each
 * entry point its own matching argument array. Only the WMMA devices reach
 * them, and only the two block-GEMM
 * passes: every other pass under an Ozaki mode still runs the plain
 * _f64 set.
 */
extern void hybsol_eliminate_pass_f64_ozi8(void);
extern void hybsol_scale_pass_f64_ozi8(void);
extern void hybsol_eliminate_pass_f64_oz2(void);
extern void hybsol_scale_pass_f64_oz2(void);

/*
 * The arithmetic :c:enumerator:`HYBSOL_HIP_ARITH_AUTO` names for double
 * factors on a gfx12-family device: the fastest mode passing the
 * selection harness gate on this hardware, set in device.c from the
 * harness table and flipped there if a future rerun ranks differently.
 */
extern const hybsol_hip_arith_t HYBSOL_HIP_DEFAULT_FOR_GFX12;

/**
 * A work-group (thread block) size worth launching a kernel with.
 *
 * Left to itself the runtime answers with the kernel's *maximum* -- 1024 on
 * a consumer card -- which puts a pass of a few hundred block rows on a
 * single compute unit while the rest of the device idles. The occupancy API
 * suggests a block that fills the device instead, and the answer is still
 * capped at 64 for the same reason the OpenCL backend caps: small enough to
 * spread over every compute unit, large enough that the tail costs little.
 *
 * Preconditions: ``device`` is an acquired device; ``kernel`` is one of the
 * stubs above.
 */
size_t hybsol_hip_work_group_size(const hybsol_hip_device_t *device, const void *kernel);

/**
 * The same, split over two dimensions for a two-dimensional launch, where the
 * bound is the *product* of the two.
 *
 * Preconditions: ``device`` is an acquired device; ``kernel`` is one of the
 * stubs above; ``x`` and ``y`` are not NULL.
 */
void hybsol_hip_work_group_size_2d(const hybsol_hip_device_t *device, const void *kernel, size_t *x, size_t *y);

/**
 * Note a backend failure worth more than its result code -- a refused launch,
 * typically -- for :c:func:`hybsol_hip_last_error`. A note longer than the
 * buffer is truncated, never rejected.
 */
void hybsol_hip_set_error(const char *format, ...);

/** Note a failing HIP call, if it has nothing of its own to say. */
void hybsol_hip_note_hip_error(hipError_t error, const char *what);

/**
 * Forget the note, for the entry points about to start an operation of their
 * own: a note may only ever explain the call that is failing right now.
 */
void hybsol_hip_clear_error(void);
