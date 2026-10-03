/**
 * @file hybsol/opencl.h
 * Enumerate the OpenCL runtime's devices and factorize on one. The backend is
 * its own library and its own Python module, so a program that never asks for
 * a device never touches OpenCL.
 *
 * The symbolic walk still runs on the CPU: a caller builds a
 * :c:type:`hybsol_elimination_t` and hands it to
 * :c:func:`hybsol_opencl_decomposition_create_on_device`, after which the
 * decomposition is queried, factorized and destroyed as any other. Vectors
 * stay double throughout, so a device that cannot do double arithmetic cannot
 * take a decomposition at all.
 */

#ifndef HYBSOL_OPENCL_H
#define HYBSOL_OPENCL_H

#include <hybsol/decomposition.h>
#include <hybsol/elimination.h>
#include <hybsol/types.h>

/** What kind of device a number from :c:func:`hybsol_opencl_device_count` names. */
typedef enum hybsol_opencl_device_kind
{
    /** The runtime did not say, which should not happen on a working one. */
    HYBSOL_OPENCL_DEVICE_UNKNOWN = 0,
    /** The host CPU reached through OpenCL, which is a perfectly good device. */
    HYBSOL_OPENCL_DEVICE_CPU,
    /** A discrete or integrated GPU. */
    HYBSOL_OPENCL_DEVICE_GPU,
    /** Anything else the runtime calls a device -- an accelerator of some kind. */
    HYBSOL_OPENCL_DEVICE_ACCELERATOR,
} hybsol_opencl_device_kind_t;

/**
 * What the OpenCL runtime reports about one device.
 *
 * Read it with :c:func:`hybsol_opencl_device_info`; the strings are NUL-terminated.
 */
/* Tagged apart from the accessor of the same name, which a C domain would otherwise register twice. */
typedef struct hybsol_opencl_device_details
{
    /** The device's own name, such as the GPU's model. */
    char name[128];
    /** Who made it. */
    char vendor[128];
    /** The runtime's identifier for the device, or empty; match on this, not on the index. */
    char uuid[40];
    /** Which kind of device it is. */
    hybsol_opencl_device_kind_t kind;
    /** Non-zero when it can do double arithmetic, which every decomposition needs. */
    int supports_double;
    /** Bytes of global memory it has. */
    uint64_t global_memory_bytes;
    /** Compute units the runtime reports. */
    uint64_t compute_units;
} hybsol_opencl_device_info_t;

/**
 * How many devices the OpenCL runtime offers, GPUs first, so index ``0`` is a
 * GPU wherever the machine has one.
 *
 * Returns ``0`` when no platform offers a device, which is the cheap way to ask
 * whether a device path is available at all.
 */
uint64_t hybsol_opencl_device_count(void);

/**
 * Get what the runtime reports about one device.
 *
 * Preconditions: ``out`` non-``NULL``; ``index`` below
 * :c:func:`hybsol_opencl_device_count`.
 *
 * Returns :c:enumerator:`HYBSOL_ERROR_NO_DEVICE` for an index no device has.
 */
hybsol_result_t hybsol_opencl_device_info(uint64_t index, hybsol_opencl_device_info_t *out);

/**
 * An acquired device: a context, a command queue and the compiled kernels,
 * shared by every decomposition on it and reference counted.
 */
typedef struct hybsol_opencl_device hybsol_opencl_device_t;

/**
 * Acquire the device at ``index``. The first acquisition of an index builds its
 * context and queue, later ones hand out the same device.
 *
 * Preconditions: ``out`` non-``NULL``.
 *
 * Returns :c:enumerator:`HYBSOL_ERROR_NO_DEVICE` for an index no device has,
 * :c:enumerator:`HYBSOL_ERROR_DEVICE` when the runtime refuses the context or
 * the queue, or :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY`. The kernels are
 * built on first use, so a build failure surfaces from the factorization that
 * wanted them, not from here.
 */
hybsol_result_t hybsol_opencl_device_acquire(uint64_t index, hybsol_opencl_device_t **out);

/**
 * Release one reference to a device, destroying it when the last one goes.
 * ``NULL`` does nothing.
 */
void hybsol_opencl_device_release(hybsol_opencl_device_t *device);

/**
 * The last backend failure that said more than its result code — a kernel build
 * log, typically — or ``""`` when there has been none. Owned by the backend.
 */
const char *hybsol_opencl_last_error(void);

/**
 * Copy the system's blocks into a decomposition whose factors live on ``device``,
 * stored in ``precision``.
 *
 * The schedule stays in host memory and the factors never come back. The
 * decomposition holds its own reference to ``device``, so the caller's may be
 * released once this returns.
 *
 * Preconditions: ``sys``, ``graph``, ``device`` and ``out`` non-``NULL``;
 * ``graph`` asserted to have been built from ``sys``.
 *
 * Returns :c:enumerator:`HYBSOL_ERROR_DEVICE_CAPABILITY` when the device cannot
 * do double arithmetic, :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY` or
 * :c:enumerator:`HYBSOL_ERROR_DEVICE` when a buffer cannot be allocated or
 * uploaded, leaving ``*out`` ``NULL`` either way.
 */
hybsol_result_t hybsol_opencl_decomposition_create_on_device(const hybsol_system_t *sys,
                                                             const hybsol_elimination_t *graph,
                                                             hybsol_precision_t precision,
                                                             hybsol_opencl_device_t *device,
                                                             hybsol_decomposition_t **out);

#endif /* HYBSOL_OPENCL_H */
