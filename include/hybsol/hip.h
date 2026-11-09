/**
 * @file hybsol/hip.h
 * Enumerate the HIP/ROCm runtime's devices and factorize on one. The backend
 * is its own library and its own Python module, so a program that never asks
 * for a device never touches libamdhip64.
 *
 * The symbolic walk still runs on the CPU: a caller builds a
 * :c:type:`hybsol_elimination_t` and hands it to
 * :c:func:`hybsol_hip_decomposition_create_on_device`, after which the
 * decomposition is queried, factorized and destroyed as any other. Vectors
 * stay double throughout, so a device that cannot do double arithmetic cannot
 * take a decomposition at all.
 */

#ifndef HYBSOL_HIP_H
#define HYBSOL_HIP_H

#include <hybsol/decomposition.h>
#include <hybsol/elimination.h>
#include <hybsol/types.h>

/** What kind of device a number from :c:func:`hybsol_hip_device_count` names. */
typedef enum hybsol_hip_device_kind
{
    /** The runtime did not say, which should not happen on a working one. */
    HYBSOL_HIP_DEVICE_UNKNOWN = 0,
    /** A discrete GPU: what callers want and what the enumeration puts first. */
    HYBSOL_HIP_DEVICE_GPU,
    /**
     * An integrated GPU -- the iGPU of an APU. HIP has no property that says
     * so, so the backend classifies by compute units: an iGPU has few (the
     * 610M has 4, against the R9700's 64), and ten or fewer means integrated.
     */
    HYBSOL_HIP_DEVICE_INTEGRATED,
} hybsol_hip_device_kind_t;

/**
 * What the HIP runtime reports about one device.
 *
 * Read it with :c:func:`hybsol_hip_device_info`; the strings are NUL-terminated.
 */
/* Tagged apart from the accessor of the same name, which a C domain would otherwise register twice. */
typedef struct hybsol_hip_device_details
{
    /** The device's own name, such as the GPU's model. */
    char name[128];
    /** Who made it. */
    char vendor[128];
    /** The runtime's identifier for the device, or empty; match on this, not on the index. */
    char uuid[40];
    /** Which kind of device it is. */
    hybsol_hip_device_kind_t kind;
    /** Non-zero when it can do double arithmetic, which every decomposition needs. */
    int supports_double;
    /** Bytes of global memory it has. */
    uint64_t global_memory_bytes;
    /** Compute units the runtime reports. */
    uint64_t compute_units;
} hybsol_hip_device_info_t;

/**
 * How the factor arithmetic is done on the device.
 *
 * :c:enumerator:`HYBSOL_HIP_ARITH_AUTO` lets the backend choose for the
 * device; :c:enumerator:`HYBSOL_HIP_ARITH_NATIVE` runs the factors in their
 * own type, single accumulating in ``float`` and double in ``double``; and
 * :c:enumerator:`HYBSOL_HIP_ARITH_DOUBLE_SINGLE` carries double factors as
 * the FP32 FMA unit. The Ozaki modes below slice the
 * factors' mantissas for the WMMA matrix cores: only gfx12-family devices
 * have those, and asking for either on another device answers
 * :c:enumerator:`HYBSOL_ERROR_DEVICE_CAPABILITY`.
 */
typedef enum hybsol_hip_arith
{
    /**
     * The measured best for this device and precision: single factors run
     * :c:enumerator:`HYBSOL_HIP_ARITH_NATIVE` (FP32 needs no split), double
     * factors take the fastest mode the selection harness gates clean on a
     * gfx12-family device -- native today -- and
     * :c:enumerator:`HYBSOL_HIP_ARITH_DOUBLE_SINGLE` on every other gfx
     * target.
     */
    HYBSOL_HIP_ARITH_AUTO = 0,
    /** The factors' own arithmetic, the CPU backend's model. */
    HYBSOL_HIP_ARITH_NATIVE = 1,
    /**
     * FP64 factor arithmetic as two FP32 pieces on the FP32 FMA unit:
     * ~14 significant digits kept, and a block outside a float's comfortable
     * range falls back to native arithmetic rather than overflow the pair.
     * Only meaningful for double factors; single factors are FP32 already.
     */
    HYBSOL_HIP_ARITH_DOUBLE_SINGLE = 2,
    /**
     * FP64 on the WMMA matrix cores by mantissa slicing (Ozaki-I, signed
     * int8 slices): the two block-GEMM passes run as sliced 16x16 tiles,
     * every other pass takes the double-single set. Needs a gfx12-family
     * device; smaller blocks keep the double-single path for those passes
     * too.
     */
    HYBSOL_HIP_ARITH_OZAKI = 3,
    /**
     * Ozaki-II: asymmetric slicing -- the first operand keeps its five
     * FP16 slices while the second is kept as a single one, block-scaled
     * apart. That is ~11 bits per operand, an accuracy no direct solve
     * gates on: the mode exists as the cheap inner solve under iterative
     * refinement (``hybsol.refined_solve``), and the selection harness
     * prints its residual without gating it. Needs a gfx12-family device;
     * smaller blocks keep the double-single path, and
     * :c:enumerator:`HYBSOL_HIP_ARITH_AUTO` never picks it.
     */
    HYBSOL_HIP_ARITH_OZAKI2 = 4,
} hybsol_hip_arith_t;

/**
 * How many devices the HIP runtime offers, discrete GPUs first, so index
 * ``0`` is the card a caller means wherever the machine has one.
 *
 * Returns ``0`` when the runtime offers no device, which is the cheap way to
 * ask whether a device path is available at all.
 */
uint64_t hybsol_hip_device_count(void);

/**
 * Get what the runtime reports about one device.
 *
 * Preconditions: ``out`` non-``NULL``; ``index`` below
 * :c:func:`hybsol_hip_device_count`.
 *
 * Returns :c:enumerator:`HYBSOL_ERROR_NO_DEVICE` for an index no device has.
 */
hybsol_result_t hybsol_hip_device_info(uint64_t index, hybsol_hip_device_info_t *out);

/**
 * An acquired device: its stream, shared by every decomposition on it and
 * reference counted. The kernels are compiled into the library, so there is
 * nothing to build at run time.
 */
typedef struct hybsol_hip_device hybsol_hip_device_t;

/**
 * Acquire the device at ``index``. The first acquisition of an index opens
 * its stream, later ones hand out the same device.
 *
 * Preconditions: ``out`` non-``NULL``.
 *
 * Returns :c:enumerator:`HYBSOL_ERROR_NO_DEVICE` for an index no device has,
 * :c:enumerator:`HYBSOL_ERROR_DEVICE` when the runtime refuses the stream, or
 * :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY`. A device whose architecture the
 * kernels were not compiled for fails at the first launch instead, with the
 * reason in :c:func:`hybsol_hip_last_error`.
 */
hybsol_result_t hybsol_hip_device_acquire(uint64_t index, hybsol_hip_device_t **out);

/**
 * Release one reference to a device, destroying it when the last one goes.
 * ``NULL`` does nothing.
 */
void hybsol_hip_device_release(hybsol_hip_device_t *device);

/**
 * The last backend failure that said more than its result code -- a refused
 * launch, typically -- or ``""`` when there has been none. Owned by the
 * backend.
 */
const char *hybsol_hip_last_error(void);

/**
 * Copy the system's blocks into a decomposition whose factors live on
 * ``device``, stored in ``precision`` and computed according to ``arith``.
 *
 * The schedule stays in host memory and the factors never come back. The
 * decomposition holds its own reference to ``device``, so the caller's may be
 * released once this returns.
 *
 * Preconditions: ``sys``, ``graph``, ``device`` and ``out`` non-``NULL``;
 * ``graph`` asserted to have been built from ``sys``.
 *
 * Returns :c:enumerator:`HYBSOL_ERROR_DEVICE_CAPABILITY` when the device
 * cannot do double arithmetic, when ``arith`` asks for a split-precision
 * mode the target lacks (Ozaki on a device without WMMA matrix cores, or
 * Ozaki-II anywhere until its phase lands),
 * :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY` or
 * :c:enumerator:`HYBSOL_ERROR_DEVICE` when a buffer cannot be allocated or
 * uploaded, leaving ``*out`` ``NULL`` either way.
 */
hybsol_result_t hybsol_hip_decomposition_create_on_device(const hybsol_system_t *sys,
                                                          const hybsol_elimination_t *graph,
                                                          hybsol_precision_t precision, hybsol_hip_arith_t arith,
                                                          hybsol_hip_device_t *device, hybsol_decomposition_t **out);

#endif /* HYBSOL_HIP_H */
