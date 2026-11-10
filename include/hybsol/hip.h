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
     * 9900X's reports 1, against the R9700's 32), and sixteen or fewer means
     * integrated.
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
 * Ozaki modes below slice the
 * factors' mantissas for the WMMA matrix cores: only gfx12-family devices
 * have those, and asking for either on another device with double factors
 * answers :c:enumerator:`HYBSOL_ERROR_DEVICE_CAPABILITY`. Single factors are
 * FP32 already and the matrix cores hold no double, so on every device an
 * Ozaki mode asks nothing beyond what ``native`` does -- a no-op, not a
 * refusal.
 */
typedef enum hybsol_hip_arith
{
    /**
     * The measured best for this device and precision: single factors run
     * :c:enumerator:`HYBSOL_HIP_ARITH_NATIVE` (FP32 needs no split), double
     * factors take the fastest mode the selection harness gates clean on a
     * gfx12-family device -- native today -- and
     * :c:enumerator:`HYBSOL_HIP_ARITH_NATIVE` on every other gfx target:
     * auto resolves to native everywhere today.
     */
    HYBSOL_HIP_ARITH_AUTO = 0,
    /** The factors' own arithmetic, the CPU backend's model. */
    HYBSOL_HIP_ARITH_NATIVE = 1,
    /**
     * FP64 on the WMMA matrix cores by mantissa slicing (Ozaki-I, signed
     * int8 slices): the two block-GEMM passes run as sliced 16x16 tiles,
     * every other pass takes the native set. Needs a gfx12-family
     * device; a system with no 16-wide block runs those two passes
     * native too.
     */
    HYBSOL_HIP_ARITH_OZAKI = 3,
    /**
     * Ozaki-II: asymmetric slicing -- the first operand keeps its five
     * FP16 slices while the second is kept as a single one, block-scaled
     * apart. That is ~11 bits per operand, an accuracy no direct solve
     * gates on: the mode exists as the cheap inner solve under iterative
     * refinement (``hybsol.refined_solve``), and the selection harness
     * prints its residual without gating it. Needs a gfx12-family device;
     * a system with no 16-wide block runs the sliced passes native too, and
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
 * Preconditions: ``out`` and ``allocator`` non-``NULL``. Pass
 * ``&CUTL_STD_ALLOCATOR`` for plain ``malloc``/``realloc``/``free``; any
 * other allocator must stay valid until :c:func:`hybsol_hip_device_release`
 * returns for the last reference, since the device allocates through it for
 * its whole life.
 *
 * Returns :c:enumerator:`HYBSOL_ERROR_NO_DEVICE` for an index no device has,
 * :c:enumerator:`HYBSOL_ERROR_DEVICE` when the runtime refuses the stream, or
 * :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY`. A device whose architecture the
 * kernels were not compiled for fails at the first launch instead, with the
 * reason in :c:func:`hybsol_hip_last_error`.
 */
hybsol_result_t hybsol_hip_device_acquire(uint64_t index, const hybsol_allocator_t *allocator,
                                          hybsol_hip_device_t **out);

/**
 * Take one more reference on an acquired device, so it outlives the caller's
 * own release. Only legal on a handle the caller holds; the count is the
 * backend's, and the retain is what lets a payload (a decomposition, say)
 * keep the device alive while the caller releases whenever it likes.
 */
void hybsol_hip_device_retain(hybsol_hip_device_t *device);

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
 * The host-side bookkeeping -- the backend state, the per-pass schedule and
 * the transfer scratch -- is allocated from ``allocator``, which therefore
 * must stay valid until :c:func:`hybsol_decomposition_destroy` returns for
 * this decomposition.
 *
 * Preconditions: ``sys``, ``graph``, ``device``, ``allocator`` and ``out``
 * non-``NULL``; ``graph`` asserted to have been built from ``sys``.
 *
 * Returns :c:enumerator:`HYBSOL_ERROR_DEVICE_CAPABILITY` when the device
 * cannot do double arithmetic, when ``arith`` asks for a mode with double
 * factors that the target lacks (Ozaki or Ozaki-II on a
 * device without WMMA matrix cores), or when ``precision`` or ``arith`` is
 * not one of the enumerated values -- an out-of-range enum names no kernel
 * set and no element width, and is refused rather than run as whichever
 * branch its comparison happens to take.
 * :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY` or
 * :c:enumerator:`HYBSOL_ERROR_DEVICE` when a buffer cannot be allocated or
 * uploaded, leaving ``*out`` ``NULL`` either way.
 */
hybsol_result_t hybsol_hip_decomposition_create_on_device(const hybsol_system_t *sys,
                                                          const hybsol_elimination_t *graph,
                                                          hybsol_precision_t precision, hybsol_hip_arith_t arith,
                                                          hybsol_hip_device_t *device,
                                                          const hybsol_allocator_t *allocator,
                                                          hybsol_decomposition_t **out);

#endif /* HYBSOL_HIP_H */
