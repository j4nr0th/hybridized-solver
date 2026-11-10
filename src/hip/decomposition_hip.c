/**
 * @file src/hip/decomposition_hip.c
 * A decomposition whose factors live in device memory.
 *
 * The schedule -- block sizes, the passes, the pattern -- stays in the host
 * frame the core laid out, because the host walks it and the queries read it.
 * The factors do not: they are uploaded once, in a flat layout that turns the
 * frame's rows and entries into the prefix arrays ``row_entry_offset``,
 * ``entry_col`` and ``entry_val_offset``, plus a tightly packed ``vals``, and
 * they never come back. A pass is one launch, with its four-byte failure slot
 * read back afterwards, so the factorization is as sequential as the CPU one.
 *
 * The transcription of the OpenCL backend's file of the same shape; the
 * differences the runtime forces are few. A ``void *`` replaces ``cl_mem``,
 * and because ``hipMalloc`` allocates on the *current* device -- where
 * ``clCreateBuffer`` took the context -- the buffer helpers pick the device
 * first. There is no program to build: the kernels are already in the
 * library, chosen by precision through the stub addresses in
 * ``hip_internal.h``.
 */

#include "hip_internal.h"

#include "core/internal.h"

#include <limits.h>
#include <math.h>
#include <string.h>

/**
 * The compiled kernel for this decomposition: the base name pasted with its
 * suffix. Single factors always take the FP32 set; double factors take the
 * native double set.
 */
#define KERNEL(state, precision, base)                                                                                       \
    ((precision) == HYBSOL_PRECISION_SINGLE ? (const void *)&base##_f32 : (const void *)&base##_f64)

/** Everything one device decomposition owns beyond the core's frame; ``destroy`` releases all of it. */
typedef struct
{
    /** The device this lives on; one reference held for the payload's life. */
    hybsol_hip_device_t *device;
    /** The decomposition this state belongs to. */
    hybsol_decomposition_t *dec;
    /**
     * The allocator handed to the creation, kept for every host allocation
     * this state makes; valid until the decomposition is destroyed, as the
     * header requires.
     */
    const hybsol_allocator_t *allocator;
    /** The largest block either dimension has, which strides the scratch. */
    uint64_t max_block;
    /**
     * Which Ozaki path the two block-GEMM passes run on: 0 none,
     * 1 the Ozaki-I signed int8 slices, 2 the Ozaki-II asymmetric walk
     * (FP16 slices against a single FP16 window). Only set on a gfx12
     * device, only for double factors, and only once a block is big
     * enough for a 16x16 tile to pay for itself; every other pass, and
     * every smaller system, runs the plain native double set -- the
     * matrix cores just do not apply to it.
     */
    int kernels_oz;
    /** What the kernels walk. */
    void *block_offsets;
    void *level_offset;
    void *level_rows;
    void *level_k;
    void *row_n_elim;
    void *row_level;
    void *row_entry_offset;
    void *entry_col;
    void *entry_val_offset;
    /** The factors. */
    void *vals;
    /** The solution vector, double whatever the factors are. */
    void *vec;
    /** One block per elimination work item of a pass, indexed by ``(row, source)``. */
    void *scratch;
    /** The per-pass work lists, on the device and on the host that sizes launches. */
    void *dev_diag_offset;
    void *dev_scale_offset;
    void *dev_diag_row;
    void *dev_scale_row;
    void *dev_scale_tail;
    /**
     * One arena for the seven host work lists below, sized exactly up
     * front: ``(n_levels + 1) * 2 + n * 3 + n_levels * 2`` counts. The
     * pointers under it are carved views, and destroy frees the arena
     * as a whole.
     */
    uint64_t *schedule;
    uint64_t *diag_offset;  /**< ``n_levels + 1``, into ``diag_row``. */
    uint64_t *scale_offset; /**< ``n_levels + 1``, into ``scale_row``. */
    uint64_t *diag_row;     /**< The rows whose diagonal a pass factors, in pass order. */
    uint64_t *scale_row;    /**< The same rows again, for the scaling that follows. */
    uint64_t *scale_tail;   /**< Blocks above each of those diagonals. */
    /** Widest source list and widest tail in a pass, so a grid can be sized. */
    uint64_t *max_sources;
    uint64_t *max_tail;
    /** Which work item hit a zero pivot this pass, or ``INT_MAX``. */
    void *fail_j;
    /** Host scratch for the transfers, so a solve does not allocate. */
    void *transfer;
    /** Bytes the transfer scratch can hold. */
    size_t transfer_bytes;
} hip_state_t;

/** Bytes one element of ``precision`` occupies in the device's ``vals``. */
static size_t scalar_bytes(const hybsol_precision_t precision)
{
    return precision == HYBSOL_PRECISION_SINGLE ? sizeof(float) : sizeof(double);
}

/**
 * The scratch stride, in elements, one work item's block walk owns.
 *
 * The plain path needs one max-sized block; the Ozaki
 * scaling pass lays a panel solve in its slot -- a 16 x max_block GEMM
 * result, the panel's inverse, and the Neumann series' working matrix,
 * 16*max_block + 512 doubles -- while its GEMM combines stay inside one
 * max-sized block. The larger of the two is the stride every launch uses.
 */
static uint64_t scratch_stride_of(const hip_state_t *const state)
{
    const uint64_t gemm = state->max_block * state->max_block;
    if (state->kernels_oz == 0)
    {
        return gemm;
    }
    const uint64_t solve = 16 * state->max_block + 512;
    return (solve > gemm) ? solve : gemm;
}

/**
 * A device buffer, or ``NULL`` for an empty array: no zero-sized allocation.
 *
 * ``hipMalloc`` takes the device from the thread's current one, so this pins
 * the device before asking -- the counterpart of ``clCreateBuffer`` taking a
 * context.
 */
static void *buffer_create(const hybsol_hip_device_t *const device, const size_t bytes)
{
    if (bytes == 0)
    {
        return NULL;
    }
    (void)hipSetDevice(device->id);
    void *buffer = NULL;
    const hipError_t status = hipMalloc(&buffer, bytes);
    if (status != hipSuccess)
    {
        hybsol_hip_note_hip_error(status, "Allocating a device buffer");
        return NULL;
    }
    return buffer;
}

/** Upload ``bytes`` of host data into ``buffer``, or report why it could not. */
static hybsol_result_t buffer_write(const hybsol_hip_device_t *const device, void *const buffer,
                                    const void *const data, const size_t bytes)
{
    if (buffer == NULL)
    {
        return HYBSOL_SUCCESS;
    }
    // Enqueued on the device's stream, where the kernels see it in order, and
    // waited for -- the blocking read of the OpenCL queue's counterpart.
    hipError_t status = hipMemcpyAsync(buffer, data, bytes, hipMemcpyHostToDevice, device->stream);
    if (status == hipSuccess)
    {
        status = hipStreamSynchronize(device->stream);
    }
    if (status != hipSuccess)
    {
        hybsol_hip_note_hip_error(status, "Uploading to the device");
        return HYBSOL_ERROR_DEVICE;
    }
    return HYBSOL_SUCCESS;
}

/** Download ``bytes`` of device data into host memory. */
static hybsol_result_t buffer_read(const hybsol_hip_device_t *const device, const void *const buffer,
                                   void *const data, const size_t bytes)
{
    if (buffer == NULL)
    {
        return HYBSOL_SUCCESS;
    }
    hipError_t status = hipMemcpyAsync(data, buffer, bytes, hipMemcpyDeviceToHost, device->stream);
    if (status == hipSuccess)
    {
        status = hipStreamSynchronize(device->stream);
    }
    if (status != hipSuccess)
    {
        hybsol_hip_note_hip_error(status, "Reading back from the device");
        return HYBSOL_ERROR_DEVICE;
    }
    return HYBSOL_SUCCESS;
}

/** Release a buffer if there is one. */
static void buffer_release(void *const buffer)
{
    if (buffer != NULL)
    {
        hipFree(buffer);
    }
}

/** Grow the transfer scratch to ``bytes``. */
static hybsol_result_t transfer_reserve(hip_state_t *const state, const size_t bytes)
{
    if (state->transfer_bytes >= bytes)
    {
        return HYBSOL_SUCCESS;
    }
    void *const grown = hybsol_grow(state->allocator, state->transfer, bytes);
    if (grown == NULL)
    {
        return HYBSOL_ERROR_OUT_OF_MEMORY;
    }
    state->transfer = grown;
    state->transfer_bytes = bytes;
    return HYBSOL_SUCCESS;
}

static hybsol_result_t backend_factorize(void *state_ptr, uint64_t n_threads);
static hybsol_result_t backend_solve(void *state_ptr, double *vec, uint64_t n_threads);
static void backend_apply_operations(void *state_ptr, uint64_t n_ops, const hybsol_operation_t *ops, double *vec);
static void backend_solve_upper(void *state_ptr, double *vec);
static uint64_t backend_device(const void *state_ptr);
static void backend_destroy(void *state_ptr);

/** The one vtable every HIP decomposition shares; its state is per-device. */
static const hybsol_backend_t HIP_BACKEND = {
    .name = "hip",
    .factorize = backend_factorize,
    .solve = backend_solve,
    .apply_operations = backend_apply_operations,
    .solve_upper = backend_solve_upper,
    .device = backend_device,
    .destroy = backend_destroy,
};

/** Copy ``count`` elements of ``src`` into ``dst`` at precision ``to``, converting when the two differ. */
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

/** Round a grid dimension up to whole work groups, which a device without non-uniform groups demands. */
static size_t round_up_to(const size_t count, const size_t local)
{
    if (local == 0 || count == 0)
    {
        return count;
    }
    return (count + local - 1) / local * local;
}

/**
 * One kernel launch on the device's in-order stream, with the arguments in
 * the order the kernel declares them. The kernel travels as its stub address;
 * the C runtime API hands each argument through a pointer to it.
 */
static hybsol_result_t kernel_launch(const hybsol_hip_device_t *const device, const void *const kernel,
                                     const uint64_t grid_x, const uint64_t grid_y, const size_t block_x,
                                     const size_t block_y, void **const args)
{
    dim3 grid;
    grid.x = (uint32_t)grid_x;
    grid.y = (uint32_t)grid_y;
    grid.z = 1u;
    dim3 block;
    block.x = (uint32_t)block_x;
    block.y = (uint32_t)block_y;
    block.z = 1u;

    const hipError_t status = hipLaunchKernel(kernel, grid, block, args, 0, device->stream);
    if (status != hipSuccess)
    {
        hybsol_hip_note_hip_error(status, "Launching a kernel");
        return HYBSOL_ERROR_DEVICE;
    }
    return HYBSOL_SUCCESS;
}

/* ------------------------------------------------------------------------- */
/* The vtable's operations                                                     */
/* ------------------------------------------------------------------------- */

hybsol_result_t hybsol_hip_decomposition_create_on_device(const hybsol_system_t *const sys,
                                                          const hybsol_elimination_t *const graph,
                                                          const hybsol_precision_t precision,
                                                          const hybsol_hip_arith_t arith,
                                                          hybsol_hip_device_t *const device,
                                                          const hybsol_allocator_t *const allocator,
                                                          hybsol_decomposition_t **const out)
{
    CUTL_ASSERT(sys != NULL, "The system must not be NULL.");
    CUTL_ASSERT(graph != NULL, "The graph must not be NULL.");
    CUTL_ASSERT(device != NULL, "The device must not be NULL.");
    CUTL_ASSERT(allocator != NULL, "The allocator must not be NULL; pass &CUTL_STD_ALLOCATOR for the default.");
    CUTL_ASSERT(out != NULL, "The output pointer must not be NULL.");
    *out = NULL;
    hybsol_hip_clear_error();

    // A value outside either enum names nothing this backend implements, and every test below
    // compares against a named mode -- an unchecked one would silently mean whichever answer the
    // comparisons happen to give. The bindings parse both arguments before reaching this call, so
    // no Python caller sees the refusal; it is here for the C caller who writes the enum himself.
    if (precision != HYBSOL_PRECISION_SINGLE && precision != HYBSOL_PRECISION_DOUBLE)
    {
        hybsol_hip_set_error("Precision %d is neither single nor double.", (int)precision);
        return HYBSOL_ERROR_DEVICE_CAPABILITY;
    }
    switch (arith)
    {
    case HYBSOL_HIP_ARITH_AUTO:
    case HYBSOL_HIP_ARITH_NATIVE:
    case HYBSOL_HIP_ARITH_OZAKI:
    case HYBSOL_HIP_ARITH_OZAKI2: break;
    default:
        hybsol_hip_set_error("Arithmetic mode %d is not one of auto, native, Ozaki or Ozaki-II.",
                             (int)arith);
        return HYBSOL_ERROR_DEVICE_CAPABILITY;
    }

    // The gfx12 family owns the WMMA matrix cores the Ozaki kernels run on,
    // and the kernels for this exact target are the ones CMake compiled.
    hipDeviceProp_t prop;
    memset(&prop, 0, sizeof(prop));
    const hipError_t probe = hipGetDeviceProperties(&prop, (int)device->id);
    const int gfx12 = (probe == hipSuccess && strncmp(prop.gcnArchName, "gfx12", 5) == 0);

    // AUTO: single factors are FP32 already and always run native; double
    // factors take the measured gfx12 default -- the selection harness
    // clocked native FP64 the fastest gating mode at every block size it
    // benches -- and native on every other gfx target: auto resolves to
    // native everywhere today.
    hybsol_hip_arith_t chosen = arith;
    if (chosen == HYBSOL_HIP_ARITH_AUTO)
    {
        if (precision == HYBSOL_PRECISION_SINGLE)
        {
            chosen = HYBSOL_HIP_ARITH_NATIVE;
        }
        else
        {
            chosen = gfx12 ? HYBSOL_HIP_DEFAULT_FOR_GFX12 : HYBSOL_HIP_ARITH_NATIVE;
        }
    }

    // An Ozaki mode without the WMMA cores is a capability miss, not a
    // silent fallback -- but only for the factors that could ever reach
    // them. Single factors run the FP32 kernels whatever ``arith`` says, on
    // every device: there the mode names no path at all, so refusing it on
    // one device and accepting it on another would be a distinction without
    // a difference. (Ozaki-II is the asymmetric variant -- one operand
    // sliced, the other a single window -- meant as the inner solve under
    // iterative refinement; the harness prints its accuracy and never
    // gates on it, and AUTO never names it.)
    if ((chosen == HYBSOL_HIP_ARITH_OZAKI || chosen == HYBSOL_HIP_ARITH_OZAKI2)
        && precision == HYBSOL_PRECISION_DOUBLE && !gfx12)
    {
        hybsol_hip_set_error("Device %llu reports %s; the Ozaki block kernels need the WMMA matrix "
                             "cores of a gfx12 target. Ask for auto or native here.",
                             (unsigned long long)device->index,
                             (probe == hipSuccess) ? prop.gcnArchName : "(unknown)");
        return HYBSOL_ERROR_DEVICE_CAPABILITY;
    }

    // Vectors are double everywhere, so a device that cannot do double takes no decomposition at all.
    if (!device->supports_double)
    {
        hybsol_hip_set_error("Device %llu cannot do double arithmetic, which every decomposition needs.",
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

    hip_state_t *const state = (hip_state_t *)hybsol_alloc(allocator, sizeof(*state));
    if (state == NULL)
    {
        hybsol_decomposition_destroy(dec);
        return HYBSOL_ERROR_OUT_OF_MEMORY;
    }
    memset(state, 0, sizeof(*state));
    state->device = device;
    state->dec = dec;
    state->allocator = allocator;
    dec->backend = &HIP_BACKEND;
    dec->backend_state = state;

    const size_t bytes_n = (size_t)(n + 1) * sizeof(uint64_t);
    const size_t bytes_levels = (size_t)(n_levels + 1) * sizeof(uint64_t);
    const size_t bytes_occ = (size_t)n_occupancy * sizeof(uint64_t);
    const size_t bytes_row = (size_t)n * sizeof(uint64_t);
    const size_t bytes_cols = (size_t)n_columns * sizeof(uint64_t);
    const size_t scalar = scalar_bytes(precision);

    state->block_offsets = buffer_create(device, bytes_n);
    state->level_offset = buffer_create(device, bytes_levels);
    state->level_rows = buffer_create(device, bytes_occ);
    state->level_k = buffer_create(device, bytes_occ);
    state->row_n_elim = buffer_create(device, bytes_row);
    state->row_level = buffer_create(device, bytes_row);
    state->row_entry_offset = buffer_create(device, bytes_n);
    state->entry_col = buffer_create(device, bytes_cols);
    state->entry_val_offset = buffer_create(device, bytes_cols);
    state->fail_j = buffer_create(device, sizeof(int));

    // Every one of these has to be there: a kernel reading a buffer that never
    // got one reads whatever the driver hands back.
    if ((n > 0 && (state->block_offsets == NULL || state->row_entry_offset == NULL || state->row_n_elim == NULL ||
                   state->row_level == NULL)) ||
        (n_levels > 0 && (state->level_offset == NULL || state->level_rows == NULL || state->level_k == NULL)) ||
        (n_columns > 0 && (state->entry_col == NULL || state->entry_val_offset == NULL)) || state->fail_j == NULL)
    {
        hybsol_hip_set_error("Out of device memory for the decomposition's buffers.");
        hybsol_decomposition_destroy(dec);
        return HYBSOL_ERROR_OUT_OF_MEMORY;
    }

    const uint64_t total_size = dec->block_offsets[n];
    state->vec = buffer_create(device, (size_t)total_size * sizeof(double));

    state->max_block = 1;
    for (uint64_t i = 0; i < n; ++i)
    {
        const uint64_t size = dec->block_offsets[i + 1] - dec->block_offsets[i];
        if (size > state->max_block)
        {
            state->max_block = size;
        }
    }

    // An Ozaki mode on a WMMA device runs the two block-GEMM passes on the
    // matrix cores once some block reaches 16 -- the tile a WMMA call answers
    // for. Below that the tile never applies and the host launches the plain
    // _f64 pair instead; every other pass runs the native set under an Ozaki
    // mode anyway. The Ozaki-I realization is the
    // signed int8 slice walk: the selection harness clocked it 1.3x ahead of
    // the FP16 slice sibling at 32-wide blocks (a tie at 16), and the loser
    // was cut along with the environment hook that had compared them.
    if ((chosen == HYBSOL_HIP_ARITH_OZAKI || chosen == HYBSOL_HIP_ARITH_OZAKI2)
        && precision == HYBSOL_PRECISION_DOUBLE)
    {
        if (state->max_block >= 16)
        {
            state->kernels_oz = (chosen == HYBSOL_HIP_ARITH_OZAKI2) ? 2 : 1;
        }
    }

    hybsol_result_t res = HYBSOL_SUCCESS;

    // One read of the schedule sizes every launch: per pass, the rows it factors, their tails, its widest source list.
    // The seven lists live in one arena sized for the exact counts, so the
    // state asks the allocator once here and gives the block back once in
    // destroy; the pointers are carved views into it.
    const size_t schedule_words = (size_t)(n_levels + 1) * 2 + (size_t)n * 3 + (size_t)n_levels * 2;
    state->schedule = (uint64_t *)hybsol_alloc(allocator, schedule_words * sizeof(uint64_t));
    if (state->schedule == NULL)
    {
        res = HYBSOL_ERROR_OUT_OF_MEMORY;
    }
    else
    {
        memset(state->schedule, 0, schedule_words * sizeof(uint64_t));
        uint64_t *carved = state->schedule;
        state->diag_offset = carved;
        carved += n_levels + 1;
        state->scale_offset = carved;
        carved += n_levels + 1;
        state->diag_row = carved;
        carved += n;
        state->scale_row = carved;
        carved += n;
        state->scale_tail = carved;
        carved += n;
        state->max_sources = carved;
        carved += n_levels;
        state->max_tail = carved;
    }

    size_t scratch_elements = 0;
    uint64_t diag_count = 0;
    uint64_t scale_count = 0;
    for (uint64_t pass = 0; pass < n_levels && res == HYBSOL_SUCCESS; ++pass)
    {
        state->diag_offset[pass] = diag_count;
        state->scale_offset[pass] = scale_count;

        uint64_t sources = 0;
        for (uint64_t j = dec->level_offset[pass]; j < dec->level_offset[pass + 1]; ++j)
        {
            const uint64_t row = dec->level_rows[j];
            const uint64_t step = dec->level_k[j];
            const uint64_t count = graph->row_offset[row + 1] - graph->row_offset[row];

            // An elimination's source blocks, counted on a row's last pass too, which can carry one as well.
            if (step < dec->row_n_elim[row])
            {
                const uint64_t source_row = graph->cols[dec->row_entry_offset[row] + step];
                const uint64_t source_count = graph->row_offset[source_row + 1] - graph->row_offset[source_row];
                const uint64_t available = source_count - dec->row_n_elim[source_row] - 1;
                if (available > sources)
                {
                    sources = available;
                }
            }

            // The row's last pass: its diagonal, and the blocks standing above it.
            if (pass != dec->row_level[row])
            {
                continue;
            }
            state->diag_row[diag_count++] = row;
            state->scale_row[scale_count] = row;
            const uint64_t tail = count - dec->row_n_elim[row] - 1;
            state->scale_tail[scale_count++] = tail;
            if (tail > state->max_tail[pass])
            {
                state->max_tail[pass] = tail;
            }
        }
        if (sources > state->max_sources[pass])
        {
            state->max_sources[pass] = sources;
        }

        // One scratch block per work item of the pass, at the stride every
        // launch of this decomposition will pass. The plain
        // elimination walk owns ``rows x sources`` items; the Ozaki scaling
        // walk owns ``diagonals x tail`` instead, and each of its waves writes
        // its panel solve into its own slot. A pass that only factors
        // diagonals has no elimination items at all -- which is exactly the
        // case where an Ozaki scale wave still runs, so the scale count is the
        // only thing that sizes the buffer there. Getting this wrong is not a
        // wrong answer but a write past the end of the buffer: for a system
        // with no eliminations anywhere, the buffer is empty, and the kernel
        // writes through a null pointer.
        const size_t elim_items = (size_t)(dec->level_offset[pass + 1] - dec->level_offset[pass]) * sources;
        size_t items = elim_items;
        if (state->kernels_oz != 0)
        {
            const size_t scale_items = (size_t)(diag_count - state->diag_offset[pass]) * (size_t)state->max_tail[pass];
            if (scale_items > items)
            {
                items = scale_items;
            }
        }
        const size_t need = items * (size_t)scratch_stride_of(state);
        if (need > scratch_elements)
        {
            scratch_elements = need;
        }
    }
    if (res == HYBSOL_SUCCESS)
    {
        state->diag_offset[n_levels] = diag_count;
        state->scale_offset[n_levels] = scale_count;
    }

    if (res == HYBSOL_SUCCESS)
    {
        state->dev_diag_offset = buffer_create(device, (size_t)(n_levels + 1) * sizeof(uint64_t));
        state->dev_scale_offset = buffer_create(device, (size_t)(n_levels + 1) * sizeof(uint64_t));
        state->dev_diag_row = buffer_create(device, diag_count * sizeof(uint64_t));
        state->dev_scale_row = buffer_create(device, scale_count * sizeof(uint64_t));
        state->dev_scale_tail = buffer_create(device, scale_count * sizeof(uint64_t));
        state->scratch = buffer_create(device, scratch_elements * scalar);

        res = buffer_write(device, state->dev_diag_offset, state->diag_offset, bytes_levels);
        if (res == HYBSOL_SUCCESS)
            res = buffer_write(device, state->dev_scale_offset, state->scale_offset, bytes_levels);
        if (res == HYBSOL_SUCCESS && diag_count > 0)
        {
            res = buffer_write(device, state->dev_diag_row, state->diag_row, diag_count * sizeof(uint64_t));
            if (res == HYBSOL_SUCCESS)
                res = buffer_write(device, state->dev_scale_row, state->scale_row, scale_count * sizeof(uint64_t));
            if (res == HYBSOL_SUCCESS)
                res = buffer_write(device, state->dev_scale_tail, state->scale_tail, scale_count * sizeof(uint64_t));
        }
    }

    // The schedule itself goes up unchanged: the host walks the same numbers.
    if (res == HYBSOL_SUCCESS)
        res = buffer_write(device, state->block_offsets, dec->block_offsets, bytes_n);
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
        res = buffer_write(device, state->row_entry_offset, dec->row_entry_offset, bytes_n);
    if (res == HYBSOL_SUCCESS)
        res = buffer_write(device, state->entry_col, graph->cols, bytes_cols);

    // The blocks, in the graph's order, converted to the factors' precision and zero where it has fill-in.
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

    state->vals = buffer_create(device, total_elements * scalar);
    if (res == HYBSOL_SUCCESS && state->vals == NULL && total_elements > 0)
    {
        res = HYBSOL_ERROR_OUT_OF_MEMORY;
    }

    uint64_t *const val_offsets =
        n_columns > 0 ? (uint64_t *)hybsol_alloc(allocator, (size_t)n_columns * sizeof(uint64_t)) : NULL;
    if (res == HYBSOL_SUCCESS && n_columns > 0 && val_offsets == NULL)
    {
        res = HYBSOL_ERROR_OUT_OF_MEMORY;
    }

    void *const staging = total_elements > 0 ? hybsol_alloc(allocator, total_elements * scalar) : NULL;
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

    hybsol_free(allocator, val_offsets);
    hybsol_free(allocator, staging);
    if (res != HYBSOL_SUCCESS)
    {
        hybsol_decomposition_destroy(dec);
        return res;
    }

    // A reference for the payload's life: the caller's own release may come first.
    device->refs++;
    *out = dec;
    return HYBSOL_SUCCESS;
}

/** One launch per pass: the eliminations, the diagonals, then the blocks above them. */
static hybsol_result_t backend_factorize(void *const state_ptr, const uint64_t n_threads)
{
    HYBSOL_MARK_USED(n_threads);
    hip_state_t *const state = (hip_state_t *)state_ptr;
    hybsol_hip_clear_error();
    hybsol_decomposition_t *const dec = state->dec;
    hybsol_hip_device_t *const device = state->device;
    const hybsol_precision_t precision = dec->precision;

    // The stream routes the launches to this device; making it current too costs nothing
    // and keeps the occupancy queries below honest.
    (void)hipSetDevice(device->id);

    // The kernels are compiled into the library; nothing to build, so nothing to fetch. Ozaki mode
    // hands the two block-GEMM passes to the matrix-core set -- same walk, extra arguments.
    const int oz = (state->kernels_oz != 0) ? 1 : 0;
    const void *const eliminate = KERNEL(state, precision, hybsol_eliminate_pass);
    const void *const scale = KERNEL(state, precision, hybsol_scale_pass);
    const void *const oz_eliminate = (state->kernels_oz == 2) ? (const void *)&hybsol_eliminate_pass_f64_oz2
                                                              : (const void *)&hybsol_eliminate_pass_f64_ozi8;
    const void *const oz_scale =
        (state->kernels_oz == 2) ? (const void *)&hybsol_scale_pass_f64_oz2 : (const void *)&hybsol_scale_pass_f64_ozi8;
    const void *const diagonal = KERNEL(state, precision, hybsol_diagonal_pass);

    // Bounded work groups rather than the runtime's maximum; the grids round up to whole ones.
    // (The Ozaki launches ignore these: one whole wavefront per work item, in work groups of
    // exactly two wavefronts -- WMMA is a wave32 instruction on this hardware, and a 16-lane
    // group would split a wavefront across two work items.)
    size_t local_x = 0;
    size_t local_y = 0;
    hybsol_hip_work_group_size_2d(device, eliminate, &local_x, &local_y);
    const size_t local_rows = hybsol_hip_work_group_size(device, diagonal);
    const size_t scale_local_x = local_x;
    const size_t scale_local_y = local_y;

    const int none = INT_MAX;
    hybsol_result_t res = buffer_write(device, state->fail_j, &none, sizeof(none));

    uint64_t scratch_stride = scratch_stride_of(state);
    for (uint64_t pass = 0; pass < dec->n_levels && res == HYBSOL_SUCCESS; ++pass)
    {
        const uint64_t from = dec->level_offset[pass];
        const uint64_t rows = dec->level_offset[pass + 1] - from;
        const uint64_t sources = state->max_sources[pass];
        const uint64_t diagonals = state->diag_offset[pass + 1] - state->diag_offset[pass];
        const uint64_t tail = state->max_tail[pass];

        // The eliminations: one work item per (block row, source block).
        if (rows > 0 && sources > 0)
        {
            uint64_t from_arg = from;
            uint64_t rows_arg = rows;
            uint64_t sources_arg = sources;
            if (oz == 0)
            {
                void *args[] = {&state->level_rows,
                                       &state->level_k,
                                       &state->row_n_elim,
                                       &state->row_entry_offset,
                                       &state->entry_col,
                                       &state->entry_val_offset,
                                       &state->block_offsets,
                                       &state->vals,
                                       &state->scratch,
                                       &from_arg,
                                       &rows_arg,
                                       &sources_arg,
                                       &scratch_stride};
                res = kernel_launch(device, eliminate, round_up_to(rows, local_x), round_up_to(sources, local_y),
                                    local_x, local_y, args);
            }
            else
            {
                // The matrix-core walk wants whole wavefronts: 1-D, 64 lanes a work group,
                // two work items each, the pair's scales taken on the device.
                const uint64_t waves = rows * sources;
                void *args[] = {&state->level_rows,
                                       &state->level_k,
                                       &state->row_n_elim,
                                       &state->row_entry_offset,
                                       &state->entry_col,
                                       &state->entry_val_offset,
                                       &state->block_offsets,
                                       &state->vals,
                                       &state->scratch,
                                       &from_arg,
                                       &rows_arg,
                                       &sources_arg,
                                       &scratch_stride};
                res = kernel_launch(device, oz_eliminate, (waves + 1u) / 2u, 1, 64, 1, args);
            }
        }

        // The diagonals of the rows whose last pass this is.
        if (res == HYBSOL_SUCCESS && diagonals > 0)
        {
            uint64_t first_row = state->diag_offset[pass];
            uint64_t count_rows = diagonals;
            void *args[] = {&state->dev_diag_row,
                                   &state->row_n_elim,
                                   &state->row_entry_offset,
                                   &state->entry_val_offset,
                                   &state->block_offsets,
                                   &state->vals,
                                   &state->fail_j,
                                   &first_row,
                                   &count_rows};
            res = kernel_launch(device, diagonal, round_up_to(diagonals, local_rows), 1, local_rows, 1,
                                args);
        }

        // The blocks above them, reading the diagonals the launch before it wrote: the stream is in order.
        if (res == HYBSOL_SUCCESS && diagonals > 0 && tail > 0)
        {
            uint64_t first_row = state->scale_offset[pass];
            uint64_t count_rows = diagonals;
            if (oz == 0)
            {
                void *args[] = {&state->dev_scale_row,
                                       &state->dev_scale_tail,
                                       &state->row_n_elim,
                                       &state->row_entry_offset,
                                       &state->entry_col,
                                       &state->entry_val_offset,
                                       &state->block_offsets,
                                       &state->vals,
                                       &first_row,
                                       &count_rows};
                res = kernel_launch(device, scale, round_up_to(diagonals, scale_local_x), round_up_to(tail, scale_local_y),
                                    scale_local_x, scale_local_y, args);
            }
            else
            {
                // Same whole-wavefront walk as the eliminations, and the panel solve's
                // scratch: the wave's index splits into (row, slot) on the tail stride.
                const uint64_t waves = diagonals * tail;
                uint64_t tail_stride = tail;
                void *args[] = {&state->dev_scale_row,
                                       &state->dev_scale_tail,
                                       &state->row_n_elim,
                                       &state->row_entry_offset,
                                       &state->entry_col,
                                       &state->entry_val_offset,
                                       &state->block_offsets,
                                       &state->vals,
                                       &first_row,
                                       &count_rows,
                                       &state->scratch,
                                       &tail_stride,
                                       &scratch_stride};
                res = kernel_launch(device, oz_scale, (waves + 1u) / 2u, 1, 64, 1, args);
            }
        }

    }

    int failed = none;
    if (res == HYBSOL_SUCCESS)
    {
        res = buffer_read(device, state->fail_j, &failed, sizeof(failed));
    }
    if (res == HYBSOL_SUCCESS && failed != none)
    {
        // The slot holds the smallest absolute index into diag_row -- the kernels add their
        // pass's offset -- so the row it names is the row that failed, in whichever pass it
        // failed, and the earliest such row in the pass order.
        dec->failing_block = state->diag_row[(uint64_t)failed];
        res = HYBSOL_ERROR_SINGULAR;
    }

    if (res == HYBSOL_SUCCESS)
    {
        dec->factorized = 1;
    }
    return res;
}

/** One forward-substitution launch per pass, then the back substitution in one work item. */
static hybsol_result_t backend_solve(void *const state_ptr, double *const vec, const uint64_t n_threads)
{
    HYBSOL_MARK_USED(n_threads);
    hip_state_t *const state = (hip_state_t *)state_ptr;
    hybsol_decomposition_t *const dec = state->dec;
    hybsol_hip_device_t *const device = state->device;
    const hybsol_precision_t precision = dec->precision;

    (void)hipSetDevice(device->id);

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

    const void *const kernel = KERNEL(state, precision, hybsol_forward_pass);
    const size_t local_size = hybsol_hip_work_group_size(device, kernel);
    for (uint64_t pass = 0; pass < dec->n_levels && res == HYBSOL_SUCCESS; ++pass)
    {
        const uint64_t from = dec->level_offset[pass];
        const uint64_t to = dec->level_offset[pass + 1];
        if (to <= from)
        {
            continue;
        }
        uint64_t pass_value = pass;
        uint64_t from_value = from;
        uint64_t rows_value = to - from;
        void *args[] = {&state->level_rows,
                               &state->level_k,
                               &state->row_n_elim,
                               &state->row_level,
                               &state->row_entry_offset,
                               &state->entry_col,
                               &state->entry_val_offset,
                               &state->block_offsets,
                               &state->vals,
                               &state->vec,
                               &pass_value,
                               &from_value,
                               &rows_value};

        // Same bounded work group as the factorization, and the same padding:
        // the padded work items drop out on the kernel's own bound.
        res = kernel_launch(device, kernel, round_up_to((size_t)(to - from), local_size), 1, local_size, 1,
                            args);
    }

    if (res == HYBSOL_SUCCESS)
    {
        const void *const back = KERNEL(state, precision, hybsol_back_substitute);
        uint64_t n_arg = dec->n;
        void *args[] = {&state->row_entry_offset,
                               &state->entry_col,
                               &state->entry_val_offset,
                               &state->block_offsets,
                               &state->vals,
                               &state->vec,
                               &n_arg};
        res = kernel_launch(device, back, 1, 1, 1, 1, args);
    }

    if (res == HYBSOL_SUCCESS)
    {
        res = buffer_read(device, state->vec, vec, bytes);
    }
    return res;
}

/** Replay a recorded operation list in one work item, uploaded alongside the vector. */
static void backend_apply_operations(void *const state_ptr, const uint64_t n_ops, const hybsol_operation_t *const ops,
                                     double *const vec)
{
    hip_state_t *const state = (hip_state_t *)state_ptr;
    hybsol_decomposition_t *const dec = state->dec;
    hybsol_hip_device_t *const device = state->device;
    const hybsol_precision_t precision = dec->precision;

    if (n_ops == 0)
    {
        return;
    }

    (void)hipSetDevice(device->id);

    // Staged per call: a replay is a host-driven list, not a hot path.
    uint32_t *const types = (uint32_t *)hybsol_alloc(state->allocator, (size_t)n_ops * sizeof(uint32_t));
    uint64_t *const rows = (uint64_t *)hybsol_alloc(state->allocator, (size_t)n_ops * sizeof(uint64_t));
    uint64_t *const cols = (uint64_t *)hybsol_alloc(state->allocator, (size_t)n_ops * sizeof(uint64_t));
    void *op_type = NULL;
    void *op_row = NULL;
    void *op_col = NULL;
    hybsol_result_t res = HYBSOL_ERROR_DEVICE;

    if (types != NULL && rows != NULL && cols != NULL)
    {
        for (uint64_t i = 0; i < n_ops; ++i)
        {
            types[i] = (uint32_t)ops[i].type;
            rows[i] = ops[i].idx_row;
            cols[i] = ops[i].idx_col;
        }

        op_type = buffer_create(device, (size_t)n_ops * sizeof(uint32_t));
        op_row = buffer_create(device, (size_t)n_ops * sizeof(uint64_t));
        op_col = buffer_create(device, (size_t)n_ops * sizeof(uint64_t));
        res = buffer_write(device, op_type, types, (size_t)n_ops * sizeof(uint32_t));
        if (res == HYBSOL_SUCCESS)
            res = buffer_write(device, op_row, rows, (size_t)n_ops * sizeof(uint64_t));
        if (res == HYBSOL_SUCCESS)
            res = buffer_write(device, op_col, cols, (size_t)n_ops * sizeof(uint64_t));
    }
    else
    {
        hybsol_hip_set_error("Out of memory staging an operation list for the device.");
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
        const void *const kernel = KERNEL(state, precision, hybsol_apply_ops);
        uint64_t n_ops_arg = n_ops;
        void *args[] = {&state->row_entry_offset,
                               &state->entry_col,
                               &state->entry_val_offset,
                               &state->block_offsets,
                               &state->vals,
                               &state->vec,
                               &op_type,
                               &op_row,
                               &op_col,
                               &n_ops_arg};
        res = kernel_launch(device, kernel, 1, 1, 1, 1, args);
    }
    if (res == HYBSOL_SUCCESS)
    {
        (void)buffer_read(device, state->vec, vec, bytes_vec);
    }

    buffer_release(op_type);
    buffer_release(op_row);
    buffer_release(op_col);
    hybsol_free(state->allocator, types);
    hybsol_free(state->allocator, rows);
    hybsol_free(state->allocator, cols);
}

/** Solve ``U x = y`` on the device: the back substitution on its own. */
static void backend_solve_upper(void *const state_ptr, double *const vec)
{
    hip_state_t *const state = (hip_state_t *)state_ptr;
    hybsol_decomposition_t *const dec = state->dec;
    hybsol_hip_device_t *const device = state->device;
    const hybsol_precision_t precision = dec->precision;

    (void)hipSetDevice(device->id);

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

    const void *const kernel = KERNEL(state, precision, hybsol_back_substitute);
    uint64_t n_arg = dec->n;
    void *args[] = {&state->row_entry_offset,
                           &state->entry_col,
                           &state->entry_val_offset,
                           &state->block_offsets,
                           &state->vals,
                           &state->vec,
                           &n_arg};
    (void)kernel_launch(device, kernel, 1, 1, 1, 1, args);
    (void)buffer_read(device, state->vec, vec, bytes);
}

static uint64_t backend_device(const void *const state_ptr)
{
    const hip_state_t *const state = (const hip_state_t *)state_ptr;
    return state->device->index;
}

static void backend_destroy(void *const state_ptr)
{
    hip_state_t *const state = (hip_state_t *)state_ptr;
    CUTL_ASSERT(state != NULL, "The backend state must not be NULL; the core only destroys a created decomposition.");

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
    buffer_release(state->dev_diag_offset);
    buffer_release(state->dev_scale_offset);
    buffer_release(state->dev_diag_row);
    buffer_release(state->dev_scale_row);
    buffer_release(state->dev_scale_tail);

    hybsol_free(state->allocator, state->schedule);
    hybsol_free(state->allocator, state->transfer);
    hybsol_hip_device_release(state->device);
    hybsol_free(state->allocator, state);
}
