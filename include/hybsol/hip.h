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
 * own type, single accumulating in ``float`` and double in ``double``. The
 * split-precision modes below exist so the ABI is stable; until their phases
 * land, asking for one is refused with
 * :c:enumerator:`HYBSOL_ERROR_DEVICE_CAPABILITY`.
 */
typedef enum hybsol_hip_arith
{
    /** The backend's best choice for this device and precision. */
    HYBSOL_HIP_ARITH_AUTO = 0,
    /** The factors' own arithmetic, the CPU backend's model. */
    HYBSOL_HIP_ARITH_NATIVE = 1,
    /** FP64 as two FP32 pieces on the FP32 FMA unit; comes in a later phase. */
    HYBSOL_HIP_ARITH_DOUBLE_SINGLE = 2,
    /** FP64 on the matrix cores by mantissa slicing; comes in a later phase. */
    HYBSOL_HIP_ARITH_OZAKI = 3,
    /** Asymmetric slicing for use under iterative refinement; a later phase. */
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
 * cannot do double arithmetic or ``arith`` asks for a split-precision mode
 * that has not landed yet, :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY` or
 * :c:enumerator:`HYBSOL_ERROR_DEVICE` when a buffer cannot be allocated or
 * uploaded, leaving ``*out`` ``NULL`` either way.
 */
hybsol_result_t hybsol_hip_decomposition_create_on_device(const hybsol_system_t *sys,
                                                          const hybsol_elimination_t *graph,
                                                          hybsol_precision_t precision, hybsol_hip_arith_t arith,
                                                          hybsol_hip_device_t *device, hybsol_decomposition_t **out);

#endif /* HYBSOL_HIP_H */
