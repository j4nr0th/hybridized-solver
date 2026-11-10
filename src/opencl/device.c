/**
 * @file src/opencl/device.c
 * Devices, contexts and kernels of the OpenCL backend.
 *
 * Everything but the kernels is guarded by one mutex: the enumeration, the
 * handle table and the error note. The kernels are issued on each device's
 * single in-order queue.
 */

#include "opencl_internal.h"
#include "core/internal.h"
#include <cutl/common_defs.h>

#include "gpu_kernels.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif

/* ------------------------------------------------------------------------- */
/* A mutex, spelled twice                                                      */
/* ------------------------------------------------------------------------- */

/* A critical section on Windows, a pthread mutex elsewhere. */

#ifdef _WIN32
static CRITICAL_SECTION g_lock;
static INIT_ONCE g_lock_once = INIT_ONCE_STATIC_INIT;
static BOOL CALLBACK init_lock(PINIT_ONCE once, PVOID param, PVOID *context)
{
    (void)once;
    (void)param;
    (void)context;
    InitializeCriticalSection(&g_lock);
    return TRUE;
}
static void lock_acquire(void)
{
    InitOnceExecuteOnce(&g_lock_once, init_lock, NULL, NULL);
    EnterCriticalSection(&g_lock);
}
static void lock_release(void)
{
    LeaveCriticalSection(&g_lock);
}
#else
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static void lock_acquire(void)
{
    pthread_mutex_lock(&g_lock);
}
static void lock_release(void)
{
    pthread_mutex_unlock(&g_lock);
}
#endif

/* ------------------------------------------------------------------------- */
/* The error note                                                              */
/* ------------------------------------------------------------------------- */

/** Room for the note; a kernel build log is truncated to fit, never rejected. */
#define ERROR_NOTE_BYTES 1024

static char g_last_error[ERROR_NOTE_BYTES];

void hybsol_opencl_set_error(const char *const format, ...)
{
    va_list args;
    va_start(args, format);
    lock_acquire();
    vsnprintf(g_last_error, sizeof(g_last_error), format, args);
    lock_release();
    va_end(args);
}

void hybsol_opencl_note_cl_error(const cl_int error, const char *const what)
{
    if (error == CL_SUCCESS)
        return;
    if (g_last_error[0] != '\0')
        return; // Something more specific already got here first.
    hybsol_opencl_set_error("%s: OpenCL error %d.", what, (int)error);
}

/* For callers that already hold the lock; the locking setter would self-deadlock. */
static void set_error_locked(const char *const format, ...)
{
    va_list args;
    va_start(args, format);
    vsnprintf(g_last_error, sizeof(g_last_error), format, args);
    va_end(args);
}

const char *hybsol_opencl_last_error(void)
{
    return g_last_error;
}

/*
 * Drop the note. The note answers "why did *this* call fail", so every entry
 * point that starts an operation of its own opens by clearing it: a note from
 * an earlier refusal would otherwise be quoted against an unrelated one,
 * naming whichever device failed last time rather than this one.
 */
void hybsol_opencl_clear_error(void)
{
    lock_acquire();
    g_last_error[0] = '\0';
    lock_release();
}

/* For callers that already hold the lock: the program build runs under it. */

static void note_cl_error_locked(const cl_int error, const char *const what)
{
    if (error == CL_SUCCESS || g_last_error[0] != '\0')
    {
        return;
    }
    snprintf(g_last_error, sizeof(g_last_error), "%s: OpenCL error %d.", what, (int)error);
}

/* ------------------------------------------------------------------------- */
/* Enumeration                                                                 */
/* ------------------------------------------------------------------------- */

/** One device as the enumeration lists it: the runtime's handle and its facts. */
typedef struct
{
    cl_device_id id;
    hybsol_opencl_device_kind_t kind;
    int supports_double;
    /** The device's name, kept for the order the enumeration is sorted into. */
    char name[128];
    /** Its position in the platform-then-device walk; the tie-break below the name. */
    int ordinal;
} listed_device_t;

/*
 * A machine has a handful of devices, so the enumeration lives in a fixed
 * table rather than a growable list: the cache never allocates. Exceeding
 * the cap is a refused enumeration with a note, never a silent truncation.
 */
#define HYBSOL_OPENCL_MAX_LISTED_DEVICES 32

/** The cached enumeration; empty until :c:func:`hybsol_opencl_device_count` asks. */
static listed_device_t g_devices[HYBSOL_OPENCL_MAX_LISTED_DEVICES];
static uint64_t g_device_count = 0;
static int g_enumerated = 0;

/**
 * Copy a device's name out of the runtime, leaving it empty rather than NULL.
 *
 * Preconditions: ``out`` holds at least ``capacity`` characters, which is positive.
 */
static void device_string(cl_device_id device, cl_device_info what, char *const out, const size_t capacity)
{
    CUTL_ASSERT(out != NULL && capacity > 0, "The output buffer must be non-empty.");

    size_t size = 0;
    out[0] = '\0';
    if (clGetDeviceInfo(device, what, 0, NULL, &size) != CL_SUCCESS || size == 0 || size > capacity)
    {
        return;
    }
    if (clGetDeviceInfo(device, what, size, out, NULL) != CL_SUCCESS)
    {
        out[0] = '\0';
    }
    out[capacity - 1] = '\0';
}

/* CL 1.2 headers declare neither CL_DEVICE_UUID nor its extension. */
#define HYBSOL_CL_DEVICE_UUID 0x106A
typedef cl_uchar hybsol_cl_uuid_t[16];

/**
 * The runtime's device UUID as a string, or an empty one where there is none.
 *
 * Preconditions: ``out`` holds at least 33 characters.
 */
static void device_uuid_string(cl_device_id device, char *const out, const size_t capacity)
{
    CUTL_ASSERT(out != NULL && capacity >= 33, "The output buffer must hold a 32-character UUID.");

    hybsol_cl_uuid_t uuid = {0};
    out[0] = '\0';
    if (clGetDeviceInfo(device, HYBSOL_CL_DEVICE_UUID, sizeof(uuid), uuid, NULL) != CL_SUCCESS)
    {
        return;
    }

    size_t at = 0;
    for (size_t i = 0; i < sizeof(uuid) && at + 3 <= capacity; ++i)
    {
        at += (size_t)snprintf(out + at, capacity - at, "%02x", uuid[i]);
    }
    out[at] = '\0';
}

/** One device's kind, mapping the runtime's bit on GPUs. */
static hybsol_opencl_device_kind_t device_kind(cl_device_id device)
{
    cl_device_type type = 0;
    if (clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, NULL) != CL_SUCCESS)
    {
        return HYBSOL_OPENCL_DEVICE_UNKNOWN;
    }
    if ((type & CL_DEVICE_TYPE_GPU) != 0)
        return HYBSOL_OPENCL_DEVICE_GPU;
    if ((type & CL_DEVICE_TYPE_CPU) != 0)
        return HYBSOL_OPENCL_DEVICE_CPU;
    if ((type & CL_DEVICE_TYPE_ACCELERATOR) != 0)
        return HYBSOL_OPENCL_DEVICE_ACCELERATOR;
    return HYBSOL_OPENCL_DEVICE_UNKNOWN;
}

/** Whether the device can do double arithmetic, which every decomposition needs. */
static int device_has_double(cl_device_id device)
{
    cl_device_fp_config config = 0;
    if (clGetDeviceInfo(device, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(config), &config, NULL) != CL_SUCCESS)
    {
        return 0;
    }
    return config != 0;
}

/**
 * Append one device to the enumeration at ``count``, remembering the walk
 * position ``ordinal`` it was listed at: two same-model devices (the same
 * card behind two ICDs, say) tie on name, and only the platform-then-device
 * order tells them apart.
 *
 * Returns ``HYBSOL_ERROR_DEVICE`` when the fixed table is full; the cap only
 * binds on a machine with more devices than this backend has ever seen.
 */
static hybsol_result_t device_append(const uint64_t count, const cl_device_id id)
{
    if (count >= HYBSOL_OPENCL_MAX_LISTED_DEVICES)
    {
        // Reached from the enumeration, which holds the lock.
        set_error_locked(
            "The OpenCL runtime lists more than %d devices; the enumeration is capped at that many.",
            HYBSOL_OPENCL_MAX_LISTED_DEVICES);
        return HYBSOL_ERROR_DEVICE;
    }

    g_devices[count].id = id;
    g_devices[count].kind = device_kind(id);
    g_devices[count].supports_double = device_has_double(id);
    device_string(id, CL_DEVICE_NAME, g_devices[count].name, sizeof(g_devices[count].name));
    g_devices[count].ordinal = (int)count;
    return HYBSOL_SUCCESS;
}

/** GPUs before everything else, then by name: an index that does not move. */
static int device_before(const void *const a_ptr, const void *const b_ptr)
{
    const listed_device_t *const a = (const listed_device_t *)a_ptr;
    const listed_device_t *const b = (const listed_device_t *)b_ptr;
    const int a_gpu = a->kind == HYBSOL_OPENCL_DEVICE_GPU;
    const int b_gpu = b->kind == HYBSOL_OPENCL_DEVICE_GPU;
    if (a_gpu != b_gpu)
    {
        return a_gpu ? -1 : 1; // A GPU sorts ahead of anything else.
    }
    const int by_name = strcmp(a->name, b->name);
    if (by_name != 0)
    {
        return by_name;
    }
    // Two same-model devices tie on the name -- one card behind two ICDs
    // spells itself identically twice over; the walk position they were
    // listed at breaks the tie, and it is the same in every process on the
    // same hardware -- what keeps qsort's total order, and the caller's
    // index, pinned down.
    return a->ordinal < b->ordinal ? -1 : (a->ordinal > b->ordinal ? 1 : 0);
}

/**
 * Collect every device of every platform, GPUs first.
 *
 * No platform at all is not a failure but a machine without a device, which
 * the caller sees as a count of zero. The table itself is static; only the
 * runtime's own lists need memory, and those come from ``allocator`` and are
 * back with it before this returns.
 */
static hybsol_result_t enumerate_devices(const hybsol_allocator_t *const allocator)
{
    cl_uint n_platforms = 0;
    if (clGetPlatformIDs(0, NULL, &n_platforms) != CL_SUCCESS || n_platforms == 0)
    {
        // The caller holds the lock; the locking setter here would self-deadlock.
        set_error_locked("No OpenCL platform answered; the loader found no vendor.");
        return HYBSOL_ERROR_NO_DEVICE;
    }

    cl_platform_id *const platforms =
        (cl_platform_id *)hybsol_alloc(allocator, (size_t)n_platforms * sizeof(*platforms));
    if (platforms == NULL)
    {
        return HYBSOL_ERROR_OUT_OF_MEMORY;
    }
    if (clGetPlatformIDs(n_platforms, platforms, NULL) != CL_SUCCESS)
    {
        hybsol_free(allocator, platforms);
        set_error_locked("The OpenCL platforms went away while they were being listed.");
        return HYBSOL_ERROR_NO_DEVICE;
    }

    uint64_t count = 0;
    hybsol_result_t res = HYBSOL_SUCCESS;

    // Two sweeps, so a GPU sorts ahead of everything else.
    for (int gpu_pass = 1; gpu_pass >= 0; --gpu_pass)
    {
        for (cl_uint p = 0; p < n_platforms && res == HYBSOL_SUCCESS; ++p)
        {
            cl_uint n_devices = 0;
            if (clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, NULL, &n_devices) != CL_SUCCESS || n_devices == 0)
            {
                continue;
            }

            cl_device_id *const ids = (cl_device_id *)hybsol_alloc(allocator, (size_t)n_devices * sizeof(*ids));
            if (ids == NULL)
            {
                res = HYBSOL_ERROR_OUT_OF_MEMORY;
                break;
            }
            if (clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, n_devices, ids, NULL) != CL_SUCCESS)
            {
                hybsol_free(allocator, ids);
                continue;
            }

            for (cl_uint d = 0; d < n_devices && res == HYBSOL_SUCCESS; ++d)
            {
                const hybsol_opencl_device_kind_t kind = device_kind(ids[d]);
                const int is_gpu = kind == HYBSOL_OPENCL_DEVICE_GPU;
                if (is_gpu == gpu_pass)
                {
                    res = device_append(count, ids[d]);
                    if (res == HYBSOL_SUCCESS)
                    {
                        ++count;
                    }
                }
            }
            hybsol_free(allocator, ids);
        }
    }

    hybsol_free(allocator, platforms);
    if (res != HYBSOL_SUCCESS)
    {
        g_device_count = 0;
        return res;
    }

    // Sort once: an index a caller writes down should still mean the same device tomorrow.
    // Two same-model devices tie on name alone, as in the HIP backend; the
    // walk ordinal they were listed at breaks the tie, so the same ICD
    // configuration sorts to the same indices in every process.
    if (count > 1)
    {
        qsort(g_devices, (size_t)count, sizeof(*g_devices), device_before);
    }

    g_device_count = count;
    return HYBSOL_SUCCESS;
}

/**
 * The enumeration, run once. The caller holds the lock.
 *
 * The entry points without an allocator argument hand in
 * ``&CUTL_STD_ALLOCATOR``: the enumeration's temporaries live and die inside
 * it, so the default is as good a pool as any there.
 */
static void ensure_enumerated(const hybsol_allocator_t *const allocator)
{
    if (g_enumerated)
    {
        return;
    }
    g_enumerated = 1;
    if (enumerate_devices(allocator) != HYBSOL_SUCCESS)
    {
        g_device_count = 0;
    }
}

uint64_t hybsol_opencl_device_count(void)
{
    lock_acquire();
    ensure_enumerated(&CUTL_STD_ALLOCATOR);
    const uint64_t count = g_device_count;
    lock_release();
    return count;
}

hybsol_result_t hybsol_opencl_device_info(const uint64_t index, hybsol_opencl_device_info_t *const out)
{
    CUTL_ASSERT(out != NULL, "The output pointer must not be NULL.");

    lock_acquire();
    ensure_enumerated(&CUTL_STD_ALLOCATOR);
    if (index >= g_device_count)
    {
        lock_release();
        return HYBSOL_ERROR_NO_DEVICE;
    }
    const cl_device_id id = g_devices[index].id;
    const hybsol_opencl_device_kind_t kind = g_devices[index].kind;
    const int supports_double = g_devices[index].supports_double;
    lock_release();

    device_string(id, CL_DEVICE_NAME, out->name, sizeof(out->name));
    device_string(id, CL_DEVICE_VENDOR, out->vendor, sizeof(out->vendor));
    out->kind = kind;
    out->supports_double = supports_double;
    device_uuid_string(id, out->uuid, sizeof(out->uuid));

    out->global_memory_bytes = 0;
    (void)clGetDeviceInfo(id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(out->global_memory_bytes), &out->global_memory_bytes,
                          NULL);
    out->compute_units = 0;
    (void)clGetDeviceInfo(id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(out->compute_units), &out->compute_units, NULL);
    return HYBSOL_SUCCESS;
}

/* ------------------------------------------------------------------------- */
/* Acquired devices                                                            */
/* ------------------------------------------------------------------------- */

/** The one handle per index, if it has been built; only touched under the lock. */
static hybsol_opencl_device_t *g_devices_acquired[HYBSOL_OPENCL_MAX_LISTED_DEVICES];

/**
 * Build the context and the queue for one enumerated device.
 *
 * Returns ``HYBSOL_ERROR_NO_DEVICE`` for an index no device has; the rest is
 * the runtime's answer. The handle itself comes from ``allocator``, which the
 * acquisition's caller owns and which the handle keeps for its own later
 * allocations.
 */
static hybsol_result_t device_create(const uint64_t index, const hybsol_allocator_t *const allocator,
                                     hybsol_opencl_device_t **const out)
{
    CUTL_ASSERT(out != NULL, "The output pointer must not be NULL.");

    lock_acquire();
    ensure_enumerated(allocator);
    if (index >= g_device_count)
    {
        lock_release();
        return HYBSOL_ERROR_NO_DEVICE;
    }
    const cl_device_id id = g_devices[index].id;
    const int supports_double = g_devices[index].supports_double;
    lock_release();

    cl_int status = CL_SUCCESS;
    cl_context context = clCreateContext(NULL, 1, &id, NULL, NULL, &status);
    if (status != CL_SUCCESS)
    {
        hybsol_opencl_note_cl_error(status, "Creating a context for the device");
        return HYBSOL_ERROR_DEVICE;
    }

    // In-order: each pass reads what the previous one wrote.
    cl_command_queue queue = clCreateCommandQueue(context, id, 0, &status);
    if (status != CL_SUCCESS)
    {
        hybsol_opencl_note_cl_error(status, "Creating a command queue for the device");
        clReleaseContext(context);
        return HYBSOL_ERROR_DEVICE;
    }

    hybsol_opencl_device_t *const device = (hybsol_opencl_device_t *)hybsol_alloc(allocator, sizeof(*device));
    if (device == NULL)
    {
        hybsol_opencl_set_error("Out of memory building the device handle.");
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return HYBSOL_ERROR_OUT_OF_MEMORY;
    }
    memset(device, 0, sizeof(*device));

    device->index = index;
    device->id = id;
    device->context = context;
    device->queue = queue;
    device->supports_double = supports_double;
    device->refs = 1;
    device->allocator = allocator;
    device->program_f32 = NULL;
    device->program_f64 = NULL;
    *out = device;
    return HYBSOL_SUCCESS;
}

hybsol_result_t hybsol_opencl_device_acquire(const uint64_t index, const hybsol_allocator_t *const allocator,
                                             hybsol_opencl_device_t **const out)
{
    CUTL_ASSERT(out != NULL, "The output pointer must not be NULL.");
    CUTL_ASSERT(allocator != NULL, "The allocator must not be NULL; pass &CUTL_STD_ALLOCATOR for the default.");
    *out = NULL;
    hybsol_opencl_clear_error();

    lock_acquire();
    ensure_enumerated(allocator);
    if (index >= g_device_count)
    {
        lock_release();
        hybsol_opencl_set_error("No OpenCL device has index %llu; the runtime lists %llu.",
                                (unsigned long long)index, (unsigned long long)g_device_count);
        return HYBSOL_ERROR_NO_DEVICE;
    }
    // g_device_count never exceeds the table, so a passing index is inside it too.
    CUTL_ASSERT(index < HYBSOL_OPENCL_MAX_LISTED_DEVICES, "The handle table is %d wide but was asked for index %llu.",
                HYBSOL_OPENCL_MAX_LISTED_DEVICES, (unsigned long long)index);
    hybsol_opencl_device_t *existing = g_devices_acquired[index];
    if (existing != NULL)
    {
        // The increment rides the same lock hold as the sample: a release
        // cannot slip between the two and free the handle we are about to
        // hand back.
        ++existing->refs;
        lock_release();
        *out = existing;
        return HYBSOL_SUCCESS;
    }
    lock_release();

    hybsol_opencl_device_t *device = NULL;
    const hybsol_result_t res = device_create(index, allocator, &device);
    if (res != HYBSOL_SUCCESS)
    {
        return res;
    }

    lock_acquire();
    if (g_devices_acquired[index] == NULL)
    {
        g_devices_acquired[index] = device;
        lock_release();
        *out = device;
        return HYBSOL_SUCCESS;
    }
    // Another thread got there first; drop this one and use theirs.
    ++g_devices_acquired[index]->refs;
    const hybsol_opencl_device_t *const winner = g_devices_acquired[index];
    lock_release();

    clReleaseCommandQueue(device->queue);
    clReleaseContext(device->context);
    hybsol_free(allocator, device);
    *out = (hybsol_opencl_device_t *)winner;
    return HYBSOL_SUCCESS;
}

void hybsol_opencl_device_retain(hybsol_opencl_device_t *const device)
{
    CUTL_ASSERT(device != NULL, "The device handle must not be NULL.");
    CUTL_ASSERT(device->refs > 0, "A released device handle must not be retained again; it had %d references.",
                device->refs);

    lock_acquire();
    ++device->refs;
    lock_release();
}

void hybsol_opencl_device_release(hybsol_opencl_device_t *const device)
{
    if (device == NULL)
    {
        return;
    }

    lock_acquire();
    if (device->refs > 1)
    {
        --device->refs;
        lock_release();
        return;
    }
    // Creation only ever hands out an index the table is wide enough for.
    CUTL_ASSERT(device->index < HYBSOL_OPENCL_MAX_LISTED_DEVICES,
                "The handle table is %d wide but a device claims index %llu.", HYBSOL_OPENCL_MAX_LISTED_DEVICES,
                (unsigned long long)device->index);
    if (g_devices_acquired[device->index] == device)
    {
        g_devices_acquired[device->index] = NULL;
    }
    lock_release();

    if (device->program_f64 != NULL)
    {
        clReleaseProgram(device->program_f64);
    }
    if (device->program_f32 != NULL)
    {
        clReleaseProgram(device->program_f32);
    }
    clReleaseCommandQueue(device->queue);
    clReleaseContext(device->context);
    hybsol_free(device->allocator, device);
}

/* ------------------------------------------------------------------------- */
/* Kernels                                                                     */
/* ------------------------------------------------------------------------- */

/**
 * The device's program for one precision, built once and cached on the handle.
 *
 * The caller holds the lock. Returns ``NULL`` with the build log in the error
 * note when the kernels would not compile.
 */
static cl_program device_program(hybsol_opencl_device_t *const device, const hybsol_precision_t precision)
{
    CUTL_ASSERT(device != NULL, "The device must not be NULL.");

    cl_program program = precision == HYBSOL_PRECISION_SINGLE ? device->program_f32 : device->program_f64;
    if (program != NULL)
    {
        return program;
    }

    const char *source = HYBSOL_GPU_KERNEL_SOURCE;
    const size_t length = strlen(source);
    cl_int status = CL_SUCCESS;
    program = clCreateProgramWithSource(device->context, 1, &source, &length, &status);
    if (status != CL_SUCCESS)
    {
        hybsol_opencl_note_cl_error(status, "Creating the kernel program");
        return NULL;
    }

    // No -cl-std: the runtime's default is its newest language, and some drivers reject the option.
    const char *const options = precision == HYBSOL_PRECISION_SINGLE ? "" : "-D HYBSOL_GPU_DOUBLE";
    status = clBuildProgram(program, 1, &device->id, options, NULL, NULL);
    if (status != CL_SUCCESS)
    {
        size_t log_size = 0;
        char *log = NULL;
        if (clGetProgramBuildInfo(program, device->id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size) == CL_SUCCESS &&
            log_size > 1)
        {
            log = (char *)hybsol_alloc(device->allocator, log_size);
        }
        if (log != NULL)
        {
            clGetProgramBuildInfo(program, device->id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            snprintf(g_last_error, sizeof(g_last_error), "The %s kernels did not build: %s",
                     precision == HYBSOL_PRECISION_SINGLE ? "single" : "double", log);
            hybsol_free(device->allocator, log);
        }
        else
        {
            note_cl_error_locked(status, "Building the kernel program");
        }
        clReleaseProgram(program);
        return NULL;
    }

    if (precision == HYBSOL_PRECISION_SINGLE)
    {
        device->program_f32 = program;
    }
    else
    {
        device->program_f64 = program;
    }
    return program;
}

size_t hybsol_opencl_work_group_size(const hybsol_opencl_device_t *const device, cl_kernel kernel)
{
    CUTL_ASSERT(device != NULL, "The device must not be NULL.");

    size_t maximum = 0;
    if (clGetKernelWorkGroupInfo(kernel, device->id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(maximum), &maximum, NULL) !=
            CL_SUCCESS ||
        maximum == 0)
    {
        return 0; // The kernel will not say: let the driver choose.
    }

    // Small enough to spread over every compute unit, large enough that the tail costs little.
    size_t wanted = 64;
    if (maximum < wanted)
    {
        wanted = maximum;
    }
    return wanted;
}

void hybsol_opencl_work_group_size_2d(const hybsol_opencl_device_t *const device, cl_kernel kernel, size_t *const x,
                                      size_t *const y)
{
    CUTL_ASSERT(device != NULL && x != NULL && y != NULL, "The device and the output pointers must not be NULL.");
    size_t maximum = 0;
    if (clGetKernelWorkGroupInfo(kernel, device->id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(maximum), &maximum, NULL) !=
            CL_SUCCESS ||
        maximum == 0)
    {
        *x = 0;
        *y = 0;
        return;
    }

    // A two-dimensional launch is bounded by the product, so the budget is split, not repeated.
    const size_t wanted = 64;
    size_t first = maximum < wanted ? maximum : wanted;
    size_t second = maximum / first;
    if (second > wanted)
    {
        second = wanted;
    }
    *x = first;
    *y = second == 0 ? 1 : second;
}

cl_kernel hybsol_opencl_kernel(hybsol_opencl_device_t *const device, const hybsol_precision_t precision,
                               const char *const name)
{
    CUTL_ASSERT(device != NULL && name != NULL, "The device and the kernel name must not be NULL.");

    cl_program program = NULL;
    lock_acquire();
    program = device_program(device, precision);
    lock_release();
    if (program == NULL)
    {
        return NULL;
    }

    cl_int status = CL_SUCCESS;
    cl_kernel kernel = clCreateKernel(program, name, &status);
    hybsol_opencl_note_cl_error(status, "Creating a kernel");
    return kernel;
}
