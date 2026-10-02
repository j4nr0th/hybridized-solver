/**
 * @file src/opencl/opencl_internal.h
 * What the OpenCL backend's two translation units share: the device handle,
 * the kernel accessor and the error note.
 *
 * Private to ``src/opencl`` -- nothing here is public API.
 */

#pragma once

/*
 * CL 1.2 is the newest language every runtime this backend cares about
 * implements -- Apple's included, which never went past it. The 1.1 queue
 * creation API is deprecated from 1.2 on, so it is asked for by name: an
 * in-order queue is exactly what a pass-synchronous factorization needs, and
 * the properties-based alternative needs 2.0.
 */
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
#ifndef CL_USE_DEPRECATED_OPENCL_1_1_APIS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#endif

#include <hybsol/opencl.h>

#include <CL/cl.h>

/**
 * An acquired device: everything a decomposition on it needs, behind one
 * reference count.
 *
 * The handle is shared: two acquisitions of the same index give the same
 * pointer, so the context, the in-order queue and the compiled kernels are
 * built once no matter how many decompositions run on the device.
 */
struct hybsol_opencl_device
{
    /** Index this device has in :c:func:`hybsol_opencl_device_count`. */
    uint64_t index;
    /** The runtime's own device. */
    cl_device_id id;
    /** Context the queue belongs to. */
    cl_context context;
    /** One in-order queue: kernels and transfers see each other in order. */
    cl_command_queue queue;
    /** Non-zero when the device does double arithmetic. */
    int supports_double;
    /** Acquisitions not yet released. */
    unsigned refs;
    /** Kernels compiled per factor precision; ``CL_SUCCESS`` until asked for. */
    cl_program program_f32;
    cl_program program_f64;
};

/**
 * A kernel of the device's program for ``precision``, freshly created.
 *
 * The caller owns it and releases it. Kernel objects are not shared: setting
 * their arguments is per call, and two threads factorizing on one device must
 * not race on one kernel's argument list.
 *
 * Returns ``NULL`` with the error note set when the program would not build or
 * the kernel is not in it.
 */
cl_kernel hybsol_opencl_kernel(hybsol_opencl_device_t *device, hybsol_precision_t precision, const char *name);

/**
 * Note a backend failure worth more than its result code -- a kernel build
 * log, typically -- for :c:func:`hybsol_opencl_last_error`. Copies at most
 * ``capacity - 1`` characters and always terminates.
 */
void hybsol_opencl_set_error(const char *format, ...);

/** Note a failing OpenCL call, if it has nothing of its own to say. */
void hybsol_opencl_note_cl_error(cl_int error, const char *what);
