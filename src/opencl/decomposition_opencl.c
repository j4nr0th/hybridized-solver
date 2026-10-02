/**
 * @file src/opencl/decomposition_opencl.c
 * A decomposition whose factors live in device memory.
 *
 * The schedule -- block sizes, the passes, the pattern -- stays in the host
 * frame the core laid out, because the host walks it pass by pass and the
 * queries read it. What does not stay is the factors: they are uploaded once,
 * in a flat layout the kernels index directly, and never come back.
 *
 * That flat layout replaces the host frame's pointer-chasing rows and entries:
 *
 *   row_entry_offset[n + 1]      prefix into the two entry arrays
 *   entry_col[n_columns]         column of each entry, ascending within a row
 *   entry_val_offset[n_columns]  element offset of its block in ``vals``
 *   vals[elements]               the blocks themselves, tightly packed
 *
 * plus the schedule itself (``block_offsets``, ``level_offset``,
 * ``level_rows``, ``level_k``, ``row_n_elim``, ``row_level``), a double
 * ``vec``, one scratch block per block row, and the four-byte slot a pass
 * reports a failed diagonal in.
 *
 * A pass is one kernel launch over its rows; the host reads the failure slot
 * back after each pass, exactly where the CPU settles its per-row results. The
 * factorization is therefore as sequential as the CPU one -- ``n_levels``
 * launches -- and as parallel within one: every row of a pass writes only its
 * own blocks.
 */

#include "opencl_internal.h"

#include "core/internal.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

/** The name of a kernel of ``precision``: the base name pasted with its suffix. */
#define HYBSOL_STRINGIFY_(x) #x
#define HYBSOL_STRINGIFY(x) HYBSOL_STRINGIFY_(x)
#define HYBSOL_K32(name) HYBSOL_STRINGIFY(name) "_f32"
#define HYBSOL_K64(name) HYBSOL_STRINGIFY(name) "_f64"
#define KERNEL_NAME(precision, base) ((precision) == HYBSOL_PRECISION_SINGLE ? HYBSOL_K32(base) : HYBSOL_K64(base))

/**
 * Everything one device decomposition owns beyond the core's frame.
 *
 * All the buffers are released together by the backend's ``destroy``; the
 * device handle is one reference, released with them.
 */
typedef struct
{
    /** The device this lives on; one reference held for the payload's life. */
    hybsol_opencl_device_t *device;
    /** The decomposition this state belongs to. */
    hybsol_decomposition_t *dec;
    /** The largest block either dimension has, which strides the scratch. */
    uint64_t max_block;
    /** What the kernels walk. */
    cl_mem block_offsets;
    cl_mem level_offset;
    cl_mem level_rows;
    cl_mem level_k;
    cl_mem row_n_elim;
    cl_mem row_level;
    cl_mem row_entry_offset;
    cl_mem entry_col;
    cl_mem entry_val_offset;
    /** The factors. */
    cl_mem vals;
    /** The solution vector, double whatever the factors are. */
    cl_mem vec;
    /** One scratch block per block row, holding an elimination's products. */
    cl_mem scratch;
    /** Which work item hit a zero pivot this pass, or ``INT_MAX``. */
    cl_mem fail_j;
    /** Host scratch for the transfers, so a solve does not allocate. */
    void *transfer;
    /** Bytes the transfer scratch can hold. */
    size_t transfer_bytes;
} opencl_state_t;

/** Bytes one element of ``precision`` occupies in the device's ``vals``. */
static size_t scalar_bytes(const hybsol_precision_t precision)
{
    return precision == HYBSOL_PRECISION_SINGLE ? sizeof(float) : sizeof(double);
}

/**
 * A device buffer, or ``NULL`` when there is nothing to put in it.
 *
 * An empty pattern has zero-length arrays all over; OpenCL has no zero-sized
 * buffer, and a kernel that would index them is never launched.
 */
static cl_mem buffer_create(hybsol_opencl_device_t *const device, const size_t bytes, const cl_mem_flags flags)
{
    if (bytes == 0)
    {
        return NULL;
    }
    cl_int status = CL_SUCCESS;
    cl_mem buffer = clCreateBuffer(device->context, flags, bytes, NULL, &status);
    if (status != CL_SUCCESS)
    {
        hybsol_opencl_note_cl_error(status, "Allocating a device buffer");
        return NULL;
    }
    return buffer;
}

/** Upload ``bytes`` of host data into ``buffer``, or report why it could not. */
static hybsol_result_t buffer_write(hybsol_opencl_device_t *const device, const cl_mem buffer, const void *const data,
                                    const size_t bytes)
{
    if (buffer == NULL)
    {
        return HYBSOL_SUCCESS;
    }
    const cl_int status = clEnqueueWriteBuffer(device->queue, buffer, CL_TRUE, 0, bytes, data, 0, NULL, NULL);
    if (status != CL_SUCCESS)
    {
        hybsol_opencl_note_cl_error(status, "Uploading to the device");
        return HYBSOL_ERROR_DEVICE;
    }
    return HYBSOL_SUCCESS;
}

/** Download ``bytes`` of device data into host memory. */
static hybsol_result_t buffer_read(const hybsol_opencl_device_t *const device, const cl_mem buffer, void *const data,
                                   const size_t bytes)
{
    if (buffer == NULL)
    {
        return HYBSOL_SUCCESS;
    }
    const cl_int status = clEnqueueReadBuffer(device->queue, buffer, CL_TRUE, 0, bytes, data, 0, NULL, NULL);
    if (status != CL_SUCCESS)
    {
        hybsol_opencl_note_cl_error(status, "Reading back from the device");
        return HYBSOL_ERROR_DEVICE;
    }
    return HYBSOL_SUCCESS;
}

/** Release a buffer if there is one. */
static void buffer_release(cl_mem buffer)
{
    if (buffer != NULL)
    {
        clReleaseMemObject(buffer);
    }
}

/** Grow the transfer scratch to ``bytes``. */
static hybsol_result_t transfer_reserve(opencl_state_t *const state, const size_t bytes)
{
    if (state->transfer_bytes >= bytes)
    {
        return HYBSOL_SUCCESS;
    }
    void *const grown = realloc(state->transfer, bytes);
    if (grown == NULL)
    {
        return HYBSOL_ERROR_OUT_OF_MEMORY;
    }
    state->transfer = grown;
    state->transfer_bytes = bytes;
    return HYBSOL_SUCCESS;
}

/**
 * Copy ``count`` elements of the system's storage into ``dst`` at the factors'
 * precision, converting when the two differ.
 */
static void convert_into(unsigned char *const dst, const void *const src, const size_t count,
                         const hybsol_precision_t to, const hybsol_precision_t from)
{
    if (from == to)
    {
        memcpy(dst, src, count * scalar_bytes(to));
    }
    else if (to == HYBSOL_PRECISION_SINGLE)
    {
        const double *const from_d = (const double *)src;
        float *const to_f = (float *)dst;
        for (size_t i = 0; i < count; ++i)
        {
            to_f[i] = (float)from_d[i];
        }
    }
    else
    {
        const float *const from_f = (const float *)src;
        double *const to_d = (double *)dst;
        for (size_t i = 0; i < count; ++i)
        {
            to_d[i] = (double)from_f[i];
        }
    }
}

/** Set one kernel argument, noting the failure for a later report. */
static void kernel_set(cl_kernel const kernel, const cl_uint index, const size_t size, const void *const value)
{
    const cl_int status = clSetKernelArg(kernel, index, size, value);
    hybsol_opencl_note_cl_error(status, "Setting a kernel argument");
}

/* ------------------------------------------------------------------------- */
/* The vtable's operations                                                     */
/* ------------------------------------------------------------------------- */

static hybsol_result_t backend_factorize(void *const state_ptr, const uint64_t n_threads);
static hybsol_result_t backend_solve(void *const state_ptr, double *vec, const uint64_t n_threads);
static void backend_apply_operations(void *const state_ptr, const uint64_t n_ops, const hybsol_operation_t *const ops,
                                     double *const vec);
static void backend_solve_upper(void *const state_ptr, double *const vec);
static uint64_t backend_device(const void *const state_ptr);
static void backend_destroy(void *const state_ptr);

/** The one vtable every OpenCL decomposition shares; its state is per-device. */
static const hybsol_backend_t OPENCL_BACKEND = {
    .name = "opencl",
    .factorize = backend_factorize,
    .solve = backend_solve,
    .apply_operations = backend_apply_operations,
    .solve_upper = backend_solve_upper,
    .device = backend_device,
    .destroy = backend_destroy,
};

hybsol_result_t hybsol_opencl_decomposition_create_on_device(const hybsol_system_t *const sys,
                                                             const hybsol_elimination_t *const graph,
                                                             const hybsol_precision_t precision,
                                                             hybsol_opencl_device_t *const device,
                                                             hybsol_decomposition_t **const out)
{
    CUTL_ASSERT(sys != NULL, "The system must not be NULL.");
    CUTL_ASSERT(graph != NULL, "The graph must not be NULL.");
    CUTL_ASSERT(device != NULL, "The device must not be NULL.");
    CUTL_ASSERT(out != NULL, "The output pointer must not be NULL.");
    *out = NULL;

    // Vectors are double in every decomposition, so a device that cannot do
    // double arithmetic cannot take one at all.
    if (!device->supports_double)
    {
        hybsol_opencl_set_error("Device %llu cannot do double arithmetic, which every decomposition needs.",
                                (unsigned long long)device->index);
        return HYBSOL_ERROR_DEVICE_CAPABILITY;
    }

    const uint64_t n = graph->n;
    const uint64_t n_columns = graph->n_columns;
    const uint64_t n_levels = graph->n_levels;
    const uint64_t n_occupancy = graph->n_occupancy;

    // The frame, without the value arena: the blocks go to the device instead.
    void *const storage = hybsol_alloc(sys->allocator, hybsol_decomposition_frame_bytes(graph, precision));
    if (storage == NULL)
    {
        return HYBSOL_ERROR_OUT_OF_MEMORY;
    }
    hybsol_decomposition_t *dec = NULL;
    const hybsol_result_t init_res = hybsol_decomposition_init_frame(sys, graph, precision, storage, &dec);
    if (init_res != HYBSOL_SUCCESS)
    {
        hybsol_free(sys->allocator, storage);
        return init_res;
    }
    dec->raw = storage;

    opencl_state_t *const state = (opencl_state_t *)calloc(1, sizeof(*state));
    if (state == NULL)
    {
        hybsol_decomposition_destroy(dec);
        return HYBSOL_ERROR_OUT_OF_MEMORY;
    }
    state->device = device;
    state->dec = dec;

    const size_t bytes_n = (size_t)(n + 1) * sizeof(uint64_t);
    const size_t bytes_levels = (size_t)(n_levels + 1) * sizeof(uint64_t);
    const size_t bytes_occ = (size_t)n_occupancy * sizeof(uint64_t);
    const size_t bytes_row = (size_t)n * sizeof(uint64_t);
    const size_t bytes_cols = (size_t)n_columns * sizeof(uint64_t);
    const size_t scalar = scalar_bytes(precision);

    const cl_mem_flags rw = CL_MEM_READ_WRITE;
    const cl_mem_flags ro = CL_MEM_READ_ONLY;
    state->block_offsets = buffer_create(device, bytes_n, ro);
    state->level_offset = buffer_create(device, bytes_levels, ro);
    state->level_rows = buffer_create(device, bytes_occ, ro);
    state->level_k = buffer_create(device, bytes_occ, ro);
    state->row_n_elim = buffer_create(device, bytes_row, ro);
    state->row_level = buffer_create(device, bytes_row, ro);
    state->row_entry_offset = buffer_create(device, bytes_n, ro);
    state->entry_col = buffer_create(device, bytes_cols, ro);
    state->entry_val_offset = buffer_create(device, bytes_cols, ro);
    state->fail_j = buffer_create(device, sizeof(cl_int), rw);

    const uint64_t total_size = dec->block_offsets[n];
    state->vec = buffer_create(device, (size_t)total_size * sizeof(double), rw);

    // One scratch block per block row, each as large as the largest block
    // squared: the products of the elimination fit in it.
    state->max_block = 1;
    for (uint64_t i = 0; i < n; ++i)
    {
        const uint64_t size = dec->block_offsets[i + 1] - dec->block_offsets[i];
        if (size > state->max_block)
        {
            state->max_block = size;
        }
    }
    state->scratch =
        buffer_create(device, (size_t)n * (size_t)state->max_block * (size_t)state->max_block * scalar, rw);

    // The schedule goes up unchanged: the host walks the same numbers the
    // kernels do.
    hybsol_result_t res = buffer_write(device, state->block_offsets, dec->block_offsets, bytes_n);
    if (res == HYBSOL_SUCCESS)
        res = buffer_write(device, state->level_offset, dec->level_offset, bytes_levels);
    if (res == HYBSOL_SUCCESS)
        res = buffer_write(device, state->level_rows, dec->level_rows, bytes_occ);
    if (res == HYBSOL_SUCCESS)
        res = buffer_write(device, state->level_k, dec->level_k, bytes_occ);
    if (res == HYBSOL_SUCCESS)
        res = buffer_write(device, state->row_n_elim, dec->row_n_elim, bytes_row);
    if (res == HYBSOL_SUCCESS)
        res = buffer_write(device, state->row_level, dec->row_level, bytes_row);
    if (res == HYBSOL_SUCCESS)
        // The entry prefix is the graph's own row offsets: its entries were
        // written in that order, so row i's entries start exactly there.
        res = buffer_write(device, state->row_entry_offset, graph->row_offset, bytes_n);
    if (res == HYBSOL_SUCCESS)
        res = buffer_write(device, state->entry_col, graph->cols, bytes_cols);

    // The blocks: the system's values, converted to the factors' precision and
    // packed in the graph's order, zeros where the pattern has fill-in.
    size_t total_elements = 0;
    for (uint64_t row = 0; row < n; ++row)
    {
        const uint64_t length = graph->row_offset[row + 1] - graph->row_offset[row];
        const uint64_t rows_of = dec->block_offsets[row + 1] - dec->block_offsets[row];
        for (uint64_t slot = 0; slot < length; ++slot)
        {
            const uint64_t col = graph->cols[graph->row_offset[row] + slot];
            total_elements += rows_of * (dec->block_offsets[col + 1] - dec->block_offsets[col]);
        }
    }

    state->vals = buffer_create(device, total_elements * scalar, rw);
    if (res == HYBSOL_SUCCESS && state->vals == NULL && total_elements > 0)
    {
        res = HYBSOL_ERROR_OUT_OF_MEMORY;
    }

    uint64_t *const val_offsets = n_columns > 0 ? (uint64_t *)malloc((size_t)n_columns * sizeof(uint64_t)) : NULL;
    if (res == HYBSOL_SUCCESS && n_columns > 0 && val_offsets == NULL)
    {
        res = HYBSOL_ERROR_OUT_OF_MEMORY;
    }

    void *const staging = total_elements > 0 ? malloc(total_elements * scalar) : NULL;
    if (res == HYBSOL_SUCCESS && total_elements > 0 && staging == NULL)
    {
        res = HYBSOL_ERROR_OUT_OF_MEMORY;
    }

    if (res == HYBSOL_SUCCESS && total_elements > 0)
    {
        unsigned char *const vals = (unsigned char *)staging;
        size_t at = 0;
        uint64_t entry = 0;
        for (uint64_t row = 0; row < n; ++row)
        {
            const uint64_t length = graph->row_offset[row + 1] - graph->row_offset[row];
            const uint64_t rows_of = dec->block_offsets[row + 1] - dec->block_offsets[row];
            for (uint64_t slot = 0; slot < length; ++slot)
            {
                const uint64_t col = graph->cols[graph->row_offset[row] + slot];
                const size_t n_values =
                    (size_t)rows_of * (size_t)(dec->block_offsets[col + 1] - dec->block_offsets[col]);
                val_offsets[entry] = (uint64_t)(at / scalar);
                const void *const src = hybsol_decomposition_entry_source(sys, graph, row, slot);
                if (src == NULL)
                {
                    memset(vals + at, 0, n_values * scalar);
                }
                else
                {
                    convert_into(vals + at, src, n_values, precision, sys->precision);
                }
                at += n_values * scalar;
                ++entry;
            }
        }
        res = buffer_write(device, state->vals, staging, total_elements * scalar);
    }
    if (res == HYBSOL_SUCCESS && n_columns > 0)
    {
        res = buffer_write(device, state->entry_val_offset, val_offsets, bytes_cols);
    }

    free(val_offsets);
    free(staging);
    if (res != HYBSOL_SUCCESS)
    {
        hybsol_decomposition_destroy(dec);
        return res;
    }

    // Keep a reference for as long as the payload lives: the caller's own
    // release may come first.
    device->refs++;
    dec->backend = &OPENCL_BACKEND;
    dec->backend_state = state;

    *out = dec;
    return HYBSOL_SUCCESS;
}

/**
 * Factorize: one kernel per pass, the failure slot read back after each.
 *
 * The host walks the passes exactly as the CPU factorization does. What differs
 * is that a pass is a single launch over its rows, and the diagonal that failed
 * is named by the smallest work item that hit a zero pivot -- the same row the
 * CPU settle loop would find first in the pass.
 */
static hybsol_result_t backend_factorize(void *const state_ptr, const uint64_t n_threads)
{
    HYBSOL_MARK_USED(n_threads);
    opencl_state_t *const state = (opencl_state_t *)state_ptr;
    hybsol_decomposition_t *const dec = state->dec;
    hybsol_opencl_device_t *const device = state->device;
    const hybsol_precision_t precision = dec->precision;

    cl_kernel const kernel = hybsol_opencl_kernel(device, precision, KERNEL_NAME(precision, hybsol_factorize_pass));
    if (kernel == NULL)
    {
        return HYBSOL_ERROR_DEVICE;
    }

    const cl_int none = INT_MAX;
    hybsol_result_t res = HYBSOL_SUCCESS;
    for (uint64_t pass = 0; pass < dec->n_levels && res == HYBSOL_SUCCESS; ++pass)
    {
        const uint64_t from = dec->level_offset[pass];
        const uint64_t to = dec->level_offset[pass + 1];
        if (to <= from)
        {
            continue;
        }

        const cl_int fill_status =
            clEnqueueFillBuffer(device->queue, state->fail_j, &none, sizeof(none), 0, sizeof(none), 0, NULL, NULL);
        if (fill_status != CL_SUCCESS)
        {
            hybsol_opencl_note_cl_error(fill_status, "Clearing the pass's failure slot");
            res = HYBSOL_ERROR_DEVICE;
            break;
        }

        kernel_set(kernel, 0, sizeof(state->level_rows), &state->level_rows);
        kernel_set(kernel, 1, sizeof(state->level_k), &state->level_k);
        kernel_set(kernel, 2, sizeof(state->row_n_elim), &state->row_n_elim);
        kernel_set(kernel, 3, sizeof(state->row_level), &state->row_level);
        kernel_set(kernel, 4, sizeof(state->row_entry_offset), &state->row_entry_offset);
        kernel_set(kernel, 5, sizeof(state->entry_col), &state->entry_col);
        kernel_set(kernel, 6, sizeof(state->entry_val_offset), &state->entry_val_offset);
        kernel_set(kernel, 7, sizeof(state->block_offsets), &state->block_offsets);
        kernel_set(kernel, 8, sizeof(state->vals), &state->vals);
        kernel_set(kernel, 9, sizeof(state->scratch), &state->scratch);
        kernel_set(kernel, 10, sizeof(state->fail_j), &state->fail_j);
        const cl_ulong pass_arg = (cl_ulong)pass;
        const cl_ulong from_arg = (cl_ulong)from;
        const cl_ulong max_block_arg = (cl_ulong)state->max_block;
        kernel_set(kernel, 11, sizeof(pass_arg), &pass_arg);
        kernel_set(kernel, 12, sizeof(from_arg), &from_arg);
        kernel_set(kernel, 13, sizeof(max_block_arg), &max_block_arg);

        const size_t global = (size_t)(to - from);
        const cl_int status = clEnqueueNDRangeKernel(device->queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
        if (status != CL_SUCCESS)
        {
            hybsol_opencl_note_cl_error(status, "Launching a factorization pass");
            res = HYBSOL_ERROR_DEVICE;
            break;
        }

        cl_int failed = none;
        const cl_int read_status =
            clEnqueueReadBuffer(device->queue, state->fail_j, CL_TRUE, 0, sizeof(failed), &failed, 0, NULL, NULL);
        if (read_status != CL_SUCCESS)
        {
            hybsol_opencl_note_cl_error(read_status, "Reading the pass's failure slot");
            res = HYBSOL_ERROR_DEVICE;
            break;
        }
        if (failed != none)
        {
            dec->failing_block = dec->level_rows[(uint64_t)failed];
            res = HYBSOL_ERROR_SINGULAR;
        }
    }

    clReleaseKernel(kernel);
    if (res == HYBSOL_SUCCESS)
    {
        dec->factorized = 1;
    }
    return res;
}

/**
 * Solve: the vector up, one kernel per pass of forward substitution, the
 * back substitution in one work item, and the vector back down.
 */
static hybsol_result_t backend_solve(void *const state_ptr, double *const vec, const uint64_t n_threads)
{
    HYBSOL_MARK_USED(n_threads);
    opencl_state_t *const state = (opencl_state_t *)state_ptr;
    hybsol_decomposition_t *const dec = state->dec;
    hybsol_opencl_device_t *const device = state->device;
    const hybsol_precision_t precision = dec->precision;

    const size_t bytes = (size_t)dec->block_offsets[dec->n] * sizeof(double);
    hybsol_result_t res = transfer_reserve(state, bytes);
    if (res != HYBSOL_SUCCESS)
    {
        return res;
    }
    memcpy(state->transfer, vec, bytes);
    res = buffer_write(device, state->vec, state->transfer, bytes);
    if (res != HYBSOL_SUCCESS)
    {
        return res;
    }

    cl_kernel kernel = hybsol_opencl_kernel(device, precision, KERNEL_NAME(precision, hybsol_forward_pass));
    if (kernel == NULL)
    {
        return HYBSOL_ERROR_DEVICE;
    }
    kernel_set(kernel, 0, sizeof(state->level_rows), &state->level_rows);
    kernel_set(kernel, 1, sizeof(state->level_k), &state->level_k);
    kernel_set(kernel, 2, sizeof(state->row_n_elim), &state->row_n_elim);
    kernel_set(kernel, 3, sizeof(state->row_level), &state->row_level);
    kernel_set(kernel, 4, sizeof(state->row_entry_offset), &state->row_entry_offset);
    kernel_set(kernel, 5, sizeof(state->entry_col), &state->entry_col);
    kernel_set(kernel, 6, sizeof(state->entry_val_offset), &state->entry_val_offset);
    kernel_set(kernel, 7, sizeof(state->block_offsets), &state->block_offsets);
    kernel_set(kernel, 8, sizeof(state->vals), &state->vals);
    kernel_set(kernel, 9, sizeof(state->vec), &state->vec);
    const cl_ulong zero = 0;
    kernel_set(kernel, 10, sizeof(zero), &zero);
    kernel_set(kernel, 11, sizeof(zero), &zero);

    for (uint64_t pass = 0; pass < dec->n_levels && res == HYBSOL_SUCCESS; ++pass)
    {
        const uint64_t from = dec->level_offset[pass];
        const uint64_t to = dec->level_offset[pass + 1];
        if (to <= from)
        {
            continue;
        }
        const cl_ulong pass_value = (cl_ulong)pass;
        const cl_ulong from_value = (cl_ulong)from;
        kernel_set(kernel, 10, sizeof(pass_value), &pass_value);
        kernel_set(kernel, 11, sizeof(from_value), &from_value);

        const size_t global = (size_t)(to - from);
        const cl_int status = clEnqueueNDRangeKernel(device->queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
        if (status != CL_SUCCESS)
        {
            hybsol_opencl_note_cl_error(status, "Launching a forward-substitution pass");
            res = HYBSOL_ERROR_DEVICE;
        }
    }
    clReleaseKernel(kernel);

    if (res == HYBSOL_SUCCESS)
    {
        cl_kernel const back = hybsol_opencl_kernel(device, precision, KERNEL_NAME(precision, hybsol_back_substitute));
        if (back == NULL)
        {
            res = HYBSOL_ERROR_DEVICE;
        }
        else
        {
            kernel_set(back, 0, sizeof(state->row_entry_offset), &state->row_entry_offset);
            kernel_set(back, 1, sizeof(state->entry_col), &state->entry_col);
            kernel_set(back, 2, sizeof(state->entry_val_offset), &state->entry_val_offset);
            kernel_set(back, 3, sizeof(state->block_offsets), &state->block_offsets);
            kernel_set(back, 4, sizeof(state->vals), &state->vals);
            kernel_set(back, 5, sizeof(state->vec), &state->vec);
            const cl_ulong n_arg = (cl_ulong)dec->n;
            kernel_set(back, 6, sizeof(n_arg), &n_arg);

            const size_t global = 1;
            const cl_int status = clEnqueueNDRangeKernel(device->queue, back, 1, NULL, &global, NULL, 0, NULL, NULL);
            if (status != CL_SUCCESS)
            {
                hybsol_opencl_note_cl_error(status, "Launching the back substitution");
                res = HYBSOL_ERROR_DEVICE;
            }
            clReleaseKernel(back);
        }
    }

    if (res == HYBSOL_SUCCESS)
    {
        res = buffer_read(device, state->vec, vec, bytes);
    }
    return res;
}

/**
 * Replay a recorded operation list on the device: one work item walking the
 * list, which is uploaded along with the vector.
 */
static void backend_apply_operations(void *const state_ptr, const uint64_t n_ops, const hybsol_operation_t *const ops,
                                     double *const vec)
{
    opencl_state_t *const state = (opencl_state_t *)state_ptr;
    hybsol_decomposition_t *const dec = state->dec;
    hybsol_opencl_device_t *const device = state->device;
    const hybsol_precision_t precision = dec->precision;

    if (n_ops == 0)
    {
        return;
    }

    // The operations are staged into their own buffers for the replay: a
    // replay is a host-driven list, not a hot path, so they are made and
    // released per call rather than kept with the payload.
    uint32_t *const types = (uint32_t *)malloc((size_t)n_ops * sizeof(uint32_t));
    uint64_t *const rows = (uint64_t *)malloc((size_t)n_ops * sizeof(uint64_t));
    uint64_t *const cols = (uint64_t *)malloc((size_t)n_ops * sizeof(uint64_t));
    cl_mem op_type = NULL;
    cl_mem op_row = NULL;
    cl_mem op_col = NULL;
    cl_kernel kernel = NULL;
    hybsol_result_t res = HYBSOL_ERROR_DEVICE;

    if (types != NULL && rows != NULL && cols != NULL)
    {
        for (uint64_t i = 0; i < n_ops; ++i)
        {
            types[i] = (uint32_t)ops[i].type;
            rows[i] = ops[i].idx_row;
            cols[i] = ops[i].idx_col;
        }

        op_type = buffer_create(device, (size_t)n_ops * sizeof(uint32_t), CL_MEM_READ_ONLY);
        op_row = buffer_create(device, (size_t)n_ops * sizeof(uint64_t), CL_MEM_READ_ONLY);
        op_col = buffer_create(device, (size_t)n_ops * sizeof(uint64_t), CL_MEM_READ_ONLY);
        res = buffer_write(device, op_type, types, (size_t)n_ops * sizeof(uint32_t));
        if (res == HYBSOL_SUCCESS)
            res = buffer_write(device, op_row, rows, (size_t)n_ops * sizeof(uint64_t));
        if (res == HYBSOL_SUCCESS)
            res = buffer_write(device, op_col, cols, (size_t)n_ops * sizeof(uint64_t));
    }
    else
    {
        hybsol_opencl_set_error("Out of memory staging an operation list for the device.");
    }

    const size_t bytes_vec = (size_t)dec->block_offsets[dec->n] * sizeof(double);
    if (res == HYBSOL_SUCCESS)
    {
        res = transfer_reserve(state, bytes_vec);
    }
    if (res == HYBSOL_SUCCESS)
    {
        memcpy(state->transfer, vec, bytes_vec);
        res = buffer_write(device, state->vec, state->transfer, bytes_vec);
    }
    if (res == HYBSOL_SUCCESS)
    {
        kernel = hybsol_opencl_kernel(device, precision, KERNEL_NAME(precision, hybsol_apply_ops));
        if (kernel == NULL)
        {
            res = HYBSOL_ERROR_DEVICE;
        }
        else
        {
            kernel_set(kernel, 0, sizeof(state->row_entry_offset), &state->row_entry_offset);
            kernel_set(kernel, 1, sizeof(state->entry_col), &state->entry_col);
            kernel_set(kernel, 2, sizeof(state->entry_val_offset), &state->entry_val_offset);
            kernel_set(kernel, 3, sizeof(state->block_offsets), &state->block_offsets);
            kernel_set(kernel, 4, sizeof(state->vals), &state->vals);
            kernel_set(kernel, 5, sizeof(state->vec), &state->vec);
            kernel_set(kernel, 6, sizeof(op_type), &op_type);
            kernel_set(kernel, 7, sizeof(op_row), &op_row);
            kernel_set(kernel, 8, sizeof(op_col), &op_col);
            const cl_ulong n_ops_arg = (cl_ulong)n_ops;
            kernel_set(kernel, 9, sizeof(n_ops_arg), &n_ops_arg);

            const size_t global = 1;
            const cl_int status = clEnqueueNDRangeKernel(device->queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
            if (status != CL_SUCCESS)
            {
                hybsol_opencl_note_cl_error(status, "Replaying operations on the device");
                res = HYBSOL_ERROR_DEVICE;
            }
        }
    }
    if (kernel != NULL)
    {
        clReleaseKernel(kernel);
    }
    if (res == HYBSOL_SUCCESS)
    {
        (void)buffer_read(device, state->vec, vec, bytes_vec);
    }

    buffer_release(op_type);
    buffer_release(op_row);
    buffer_release(op_col);
    free(types);
    free(rows);
    free(cols);
}

/** Solve ``U x = y`` on the device: the back substitution on its own. */
static void backend_solve_upper(void *const state_ptr, double *const vec)
{
    opencl_state_t *const state = (opencl_state_t *)state_ptr;
    hybsol_decomposition_t *const dec = state->dec;
    hybsol_opencl_device_t *const device = state->device;
    const hybsol_precision_t precision = dec->precision;

    const size_t bytes = (size_t)dec->block_offsets[dec->n] * sizeof(double);
    if (transfer_reserve(state, bytes) != HYBSOL_SUCCESS)
    {
        return;
    }
    memcpy(state->transfer, vec, bytes);
    if (buffer_write(device, state->vec, state->transfer, bytes) != HYBSOL_SUCCESS)
    {
        return;
    }

    cl_kernel const kernel = hybsol_opencl_kernel(device, precision, KERNEL_NAME(precision, hybsol_back_substitute));
    if (kernel == NULL)
    {
        return;
    }
    kernel_set(kernel, 0, sizeof(state->row_entry_offset), &state->row_entry_offset);
    kernel_set(kernel, 1, sizeof(state->entry_col), &state->entry_col);
    kernel_set(kernel, 2, sizeof(state->entry_val_offset), &state->entry_val_offset);
    kernel_set(kernel, 3, sizeof(state->block_offsets), &state->block_offsets);
    kernel_set(kernel, 4, sizeof(state->vals), &state->vals);
    kernel_set(kernel, 5, sizeof(state->vec), &state->vec);
    const cl_ulong n_arg = (cl_ulong)dec->n;
    kernel_set(kernel, 6, sizeof(n_arg), &n_arg);

    const size_t global = 1;
    const cl_int status = clEnqueueNDRangeKernel(device->queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    if (status != CL_SUCCESS)
    {
        hybsol_opencl_note_cl_error(status, "Solving the upper system on the device");
    }
    clReleaseKernel(kernel);
    (void)buffer_read(device, state->vec, vec, bytes);
}

static uint64_t backend_device(const void *const state_ptr)
{
    const opencl_state_t *const state = (const opencl_state_t *)state_ptr;
    return state->device->index;
}

static void backend_destroy(void *const state_ptr)
{
    opencl_state_t *const state = (opencl_state_t *)state_ptr;
    if (state == NULL)
    {
        return;
    }

    buffer_release(state->block_offsets);
    buffer_release(state->level_offset);
    buffer_release(state->level_rows);
    buffer_release(state->level_k);
    buffer_release(state->row_n_elim);
    buffer_release(state->row_level);
    buffer_release(state->row_entry_offset);
    buffer_release(state->entry_col);
    buffer_release(state->entry_val_offset);
    buffer_release(state->vals);
    buffer_release(state->vec);
    buffer_release(state->scratch);
    buffer_release(state->fail_j);

    free(state->transfer);
    hybsol_opencl_device_release(state->device);
    free(state);
}
