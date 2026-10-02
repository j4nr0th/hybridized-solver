/**
 * @file hybsol/opencl.h
 * The OpenCL backend: enumerate devices, and factorize on one.
 *
 * This is a *backend*, not part of the core. It ships as its own library
 * (``hybsol::opencl``) and its own Python module (``import hybsol.opencl``),
 * so a program that never asks for a device never touches OpenCL -- neither at
 * build time nor at run time. Building without it simply leaves this header
 * out of the picture.
 *
 * A decomposition made here behaves like any other: it is queried, solved and
 * destroyed through ``hybsol/decomposition.h``. What differs is where the
 * factors live -- on the device, and never on the host -- and that
 * :c:func:`hybsol_decomposition_factorize_with_workspace`, whose scratch is a
 * host buffer, has nothing to offer it.
 *
 * The symbolic walk still runs on the CPU: a caller has one
 * :c:type:`hybsol_elimination_t` from :c:func:`hybsol_elimination_create`, and
 * :c:func:`hybsol_opencl_decomposition_create_on_device` turns it into a
 * decomposition on the device with its values uploaded. From there the
 * factorization, the solve and the replay are ordinary
 * :c:func:`hybsol_decomposition_factorize`,
 * :c:func:`hybsol_decomposition_solve` and the rest: a device decomposition
 * answers the same queries as any other, and
 * :c:func:`hybsol_decomposition_destroy` releases it.
 *
 * The sequence is therefore: acquire a device, walk the graph, create the
 * decomposition on the device, factorize it, and release the device -- the
 * decomposition keeps its own reference until it is destroyed.
 *
 * Every device the OpenCL runtime offers is listed, GPUs first so that a
 * caller reaching for ``0`` gets a GPU wherever there is one. Vectors stay
 * double throughout, as in the CPU path; a device that cannot do double
 * arithmetic cannot take a decomposition at all.
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
 * Read it with :c:func:`hybsol_opencl_device_info`; the two strings are
 * NUL-terminated and always filled.
 */
/*
 * Tagged apart from the function of the same name on purpose: a C domain that
 * registers a struct under its tag would see two ``hybsol_opencl_device_info``
 * declarations, the type and the accessor, and complain about the second.
 */
typedef struct hybsol_opencl_device_details
{
    /** The device's own name, such as the GPU's model. */
    char name[128];
    /** Who made it. */
    char vendor[128];
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
 * How many devices the OpenCL runtime offers.
 *
 * Enumerated once per process and cached. Returns ``0`` when the OpenCL
 * loader is missing, no platform offers a device, or the backend was built
 * without OpenCL -- which makes this the cheap way to ask whether a device
 * path is available at all.
 */
uint64_t hybsol_opencl_device_count(void);

/**
 * Get what the runtime reports about one device.
 *
 * The enumeration is GPUs first, so device ``0`` is a GPU whenever the machine
 * has one; :c:func:`hybsol_opencl_device_count` is its upper bound.
 *
 * Returns :c:enumerator:`HYBSOL_ERROR_NO_DEVICE` for an index no device has.
 */
hybsol_result_t hybsol_opencl_device_info(uint64_t index, hybsol_opencl_device_info_t *out);

/**
 * An acquired device: a context, a command queue and the compiled kernels,
 * shared by every decomposition on it.
 *
 * Opaque on purpose -- it holds OpenCL handles, which no caller should see.
 * Handles are reference counted: acquiring the same index twice gives the
 * same device, and it goes away with the last release.
 */
typedef struct hybsol_opencl_device hybsol_opencl_device_t;

/**
 * Acquire the device at ``index``, ready to take decompositions.
 *
 * The first acquisition of an index builds the context, the in-order queue
 * and the kernels; later ones hand out the same device with its reference
 * count raised.
 *
 * Returns :c:enumerator:`HYBSOL_ERROR_NO_DEVICE` for an index no device has or
 * a context the runtime refused, and :c:enumerator:`HYBSOL_ERROR_DEVICE` when
 * the kernels did not build -- :c:func:`hybsol_opencl_last_error` then says
 * what the compiler complained about. ``NULL`` does nothing.
 */
hybsol_result_t hybsol_opencl_device_acquire(uint64_t index, hybsol_opencl_device_t **out);

/**
 * Release one reference to a device, destroying it when the last one goes.
 * ``NULL`` does nothing.
 */
void hybsol_opencl_device_release(hybsol_opencl_device_t *device);

/**
 * The last backend failure worth more than its result code -- a kernel build
 * log, typically.
 *
 * Returns a NUL-terminated string owned by the backend, or ``""`` when the
 * last call failed for a reason that has nothing to say. Useful right after a
 * :c:enumerator:`HYBSOL_ERROR_DEVICE`.
 */
const char *hybsol_opencl_last_error(void);

/**
 * Copy the system's blocks into a decomposition whose factors live on a
 * device, for factors in ``precision``.
 *
 * The block sizes, the pattern and the schedule stay in host memory -- they
 * are what the queries, the operation list and the host-side pass walk read --
 * but the factors themselves are uploaded and never come back. The
 * decomposition is not factorized yet; run
 * :c:func:`hybsol_decomposition_factorize` on it as usual, which hands the work
 * to the device. ``n_threads`` is then meaningless, and ``precision`` need not
 * be the system's own.
 *
 * Preconditions: ``sys``, ``graph``, ``device`` and ``out`` non-``NULL``;
 * ``graph`` asserted to have been built from ``sys``.
 *
 * Returns :c:enumerator:`HYBSOL_ERROR_NO_DEVICE` for a null or foreign device,
 * :c:enumerator:`HYBSOL_ERROR_DEVICE_CAPABILITY` when the device cannot do
 * double arithmetic, :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY` when the
 * device could not hold the factors, and otherwise whatever the upload or the
 * walk reported. ``*out`` is ``NULL`` on failure.
 */
hybsol_result_t hybsol_opencl_decomposition_create_on_device(const hybsol_system_t *sys,
                                                             const hybsol_elimination_t *graph,
                                                             hybsol_precision_t precision,
                                                             hybsol_opencl_device_t *device,
                                                             hybsol_decomposition_t **out);

#endif /* HYBSOL_OPENCL_H */
