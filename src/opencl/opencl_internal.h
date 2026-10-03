/**
 * @file src/opencl/opencl_internal.h
 * What the OpenCL backend's two translation units share: the device handle,
 * the kernel accessor and the error note.
 *
 * Private to ``src/opencl``.
 */

#pragma once

/*
 * CL 1.2 is the newest language every runtime this backend cares about
 * implements. The 1.1 queue creation API is deprecated from 1.2 on, so it is
 * asked for by name: an in-order queue is what a pass-synchronous
 * factorization needs, and the properties-based alternative needs 2.0.
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
 * Two acquisitions of the same index share one pointer, so the context, the
 * queue and the compiled kernels are built once.
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
 * A work-group size worth launching a kernel with.
 *
 * Left to itself the driver answers with the kernel's *maximum* -- 8192 on a
 * consumer card -- which puts a pass of a few hundred block rows on a single
 * compute unit while the rest of the device idles. Zero means "let the driver
 * choose", which is the fallback when the kernel will not say.
 *
 * Preconditions: ``device`` is an acquired device.
 */
size_t hybsol_opencl_work_group_size(const hybsol_opencl_device_t *device, cl_kernel kernel);

/**
 * The same, split over two dimensions for a two-dimensional launch, where the
 * bound is the *product* of the two. Zero means "let the driver choose".
 *
 * Preconditions: ``device`` is an acquired device; ``x`` and ``y`` are not NULL.
 */
void hybsol_opencl_work_group_size_2d(const hybsol_opencl_device_t *device, cl_kernel kernel, size_t *x, size_t *y);

/**
 * A kernel of the device's program for ``precision``, freshly created.
 *
 * The caller owns it and releases it: kernel objects are not shared, since
 * setting their arguments is per call and two threads must not race on one.
 *
 * Returns ``NULL`` with the error note set when the program would not build or
 * the kernel is not in it.
 */
cl_kernel hybsol_opencl_kernel(hybsol_opencl_device_t *device, hybsol_precision_t precision, const char *name);

/**
 * Note a backend failure worth more than its result code -- a kernel build
 * log, typically -- for :c:func:`hybsol_opencl_last_error`. A note longer than
 * the buffer is truncated, never rejected.
 */
void hybsol_opencl_set_error(const char *format, ...);

/** Note a failing OpenCL call, if it has nothing of its own to say. */
void hybsol_opencl_note_cl_error(cl_int error, const char *what);
