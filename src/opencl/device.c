/**
 * @file src/opencl/device.c
 * Devices, contexts and kernels of the OpenCL backend.
 *
 * The enumeration happens once per process and is cached: devices do not
 * appear and disappear while a program runs, and asking the runtime on every
 * call would be the only thing this file does otherwise. Every other piece of
 * state is behind a small mutex -- enumeration, the handle table and the error
 * note -- so a program using several devices from several threads does not
 * race; the work itself is issued on one in-order queue per device, which is
 * where OpenCL's own thread safety applies.
 */

#include "opencl_internal.h"
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

/*
 * The backend is linked into extension modules and test binaries on any
 * platform, so the one thing it needs beyond OpenCL is whatever mutex that
 * platform has: a critical section on Windows, a pthread mutex elsewhere.
 */

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

const char *hybsol_opencl_last_error(void)
{
    return g_last_error;
}

/* The same notes, for callers that already hold the lock: the program build
   runs under it, and a second acquisition would deadlock. */

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
} listed_device_t;

/** The cached enumeration; empty until :c:func:`hybsol_opencl_device_count` asks. */
static listed_device_t *g_devices = NULL;
static uint64_t g_device_count = 0;
static int g_enumerated = 0;

/** Copy a device's name out of the runtime, leaving it empty rather than NULL. */
static void device_string(cl_device_id device, cl_device_info what, char *const out, const size_t capacity)
{
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

/* ``CL_DEVICE_UUID`` is OpenCL 2.0 proper and the cl_khr_device_uuid extension
 * before it, with the same value; the CL 1.2 headers this file targets declare
 * neither, so both are spelled here. */
#define HYBSOL_CL_DEVICE_UUID 0x106A
typedef cl_uchar hybsol_cl_uuid_t[16];

/**
 * The runtime's device UUID as a string, or an empty one where there is none.
 *
 * A runtime that does not know the query answers with an error and leaves the
 * string empty -- which is exactly what a caller matching on this needs to be
 * able to see.
 */
static void device_uuid_string(cl_device_id device, char *const out, const size_t capacity)
{
    hybsol_cl_uuid_t uuid = {0};
    out[0] = '\0';
    if (clGetDeviceInfo(device, HYBSOL_CL_DEVICE_UUID, sizeof(uuid), uuid, NULL) != CL_SUCCESS)
    {
        return;
    }

    size_t at = 0;
    for (size_t i = 0; i < sizeof(uuid) && at + 3 < capacity; ++i)
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

/** Append one device to the enumeration, or fail and say why. */
static hybsol_result_t device_append(listed_device_t **const list, uint64_t *const count, const cl_device_id id)
{
    listed_device_t *const grown = (listed_device_t *)realloc(*list, (size_t)(*count + 1) * sizeof(**list));
    if (grown == NULL)
    {
        hybsol_opencl_set_error("Out of memory listing the OpenCL devices.");
        return HYBSOL_ERROR_OUT_OF_MEMORY;
    }

    grown[*count].id = id;
    grown[*count].kind = device_kind(id);
    grown[*count].supports_double = device_has_double(id);
    device_string(id, CL_DEVICE_NAME, grown[*count].name, sizeof(grown[*count].name));
    *list = grown;
    ++*count;
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
        return by_name < 0;
    }
    return 0;
}

/**
 * Collect every device of every platform, GPUs first.
 *
 * The order is the runtime's within each kind, which is stable for a given
 * installation: a caller reaching for index 0 gets the GPU when there is one.
 * A runtime that answers with nothing is not an error -- it is a machine with
 * no OpenCL device, and the caller learns that from a count of zero.
 */
static hybsol_result_t enumerate_devices(void)
{
    cl_uint n_platforms = 0;
    if (clGetPlatformIDs(0, NULL, &n_platforms) != CL_SUCCESS || n_platforms == 0)
    {
        hybsol_opencl_set_error("No OpenCL platform answered; the loader found no vendor.");
        return HYBSOL_ERROR_NO_DEVICE;
    }

    cl_platform_id *const platforms = (cl_platform_id *)malloc((size_t)n_platforms * sizeof(*platforms));
    if (platforms == NULL)
    {
        return HYBSOL_ERROR_OUT_OF_MEMORY;
    }
    if (clGetPlatformIDs(n_platforms, platforms, NULL) != CL_SUCCESS)
    {
        free(platforms);
        hybsol_opencl_set_error("The OpenCL platforms went away while they were being listed.");
        return HYBSOL_ERROR_NO_DEVICE;
    }

    listed_device_t *list = NULL;
    uint64_t count = 0;
    hybsol_result_t res = HYBSOL_SUCCESS;

    // Two sweeps, so a GPU sorts ahead of everything else without the runtime
    // promising any particular order within a kind.
    for (int gpu_pass = 1; gpu_pass >= 0; --gpu_pass)
    {
        for (cl_uint p = 0; p < n_platforms && res == HYBSOL_SUCCESS; ++p)
        {
            cl_uint n_devices = 0;
            if (clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, NULL, &n_devices) != CL_SUCCESS || n_devices == 0)
            {
                continue;
            }

            cl_device_id *const ids = (cl_device_id *)malloc((size_t)n_devices * sizeof(*ids));
            if (ids == NULL)
            {
                res = HYBSOL_ERROR_OUT_OF_MEMORY;
                break;
            }
            if (clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, n_devices, ids, NULL) != CL_SUCCESS)
            {
                free(ids);
                continue;
            }

            for (cl_uint d = 0; d < n_devices && res == HYBSOL_SUCCESS; ++d)
            {
                const hybsol_opencl_device_kind_t kind = device_kind(ids[d]);
                const int is_gpu = kind == HYBSOL_OPENCL_DEVICE_GPU;
                if (is_gpu == gpu_pass)
                {
                    res = device_append(&list, &count, ids[d]);
                }
            }
            free(ids);
        }
    }

    free(platforms);
    if (res != HYBSOL_SUCCESS)
    {
        free(list);
        return res;
    }

    // Sort once, here, rather than trusting what order the platforms answered
    // in: an index a caller can write down should still mean the same device
    // tomorrow.
    if (count > 1)
    {
        qsort(list, (size_t)count, sizeof(*list), device_before);
    }

    g_devices = list;
    g_device_count = count;
    return HYBSOL_SUCCESS;
}

/** The enumeration, run once. The caller holds the lock. */
static void ensure_enumerated(void)
{
    if (g_enumerated)
    {
        return;
    }
    g_enumerated = 1;
    if (enumerate_devices() != HYBSOL_SUCCESS)
    {
        free(g_devices);
        g_devices = NULL;
        g_device_count = 0;
    }
}

uint64_t hybsol_opencl_device_count(void)
{
    lock_acquire();
    ensure_enumerated();
    const uint64_t count = g_device_count;
    lock_release();
    return count;
}

hybsol_result_t hybsol_opencl_device_info(const uint64_t index, hybsol_opencl_device_info_t *const out)
{
    CUTL_ASSERT(out != NULL, "The output pointer must not be NULL.");

    lock_acquire();
    ensure_enumerated();
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

/**
 * The one handle per index, if it has been built.
 *
 * A small array indexed by the device index, grown as devices are acquired.
 * Holding it under the lock keeps two threads acquiring the same index from
 * building two contexts.
 */
static hybsol_opencl_device_t **g_devices_acquired = NULL;
static uint64_t g_acquired_capacity = 0;

/** Build the context, the queue and the kernels for one enumerated device. */
static hybsol_result_t device_create(const uint64_t index, hybsol_opencl_device_t **const out)
{
    lock_acquire();
    ensure_enumerated();
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

    // In-order: the factorization reads what the previous pass wrote, and the
    // host waits on its own transfers. The event queue would be correct but
    // needless here.
    cl_command_queue queue = clCreateCommandQueue(context, id, 0, &status);
    if (status != CL_SUCCESS)
    {
        hybsol_opencl_note_cl_error(status, "Creating a command queue for the device");
        clReleaseContext(context);
        return HYBSOL_ERROR_DEVICE;
    }

    hybsol_opencl_device_t *const device = (hybsol_opencl_device_t *)calloc(1, sizeof(*device));
    if (device == NULL)
    {
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return HYBSOL_ERROR_OUT_OF_MEMORY;
    }

    device->index = index;
    device->id = id;
    device->context = context;
    device->queue = queue;
    device->supports_double = supports_double;
    device->refs = 1;
    device->program_f32 = NULL;
    device->program_f64 = NULL;
    *out = device;
    return HYBSOL_SUCCESS;
}

hybsol_result_t hybsol_opencl_device_acquire(const uint64_t index, hybsol_opencl_device_t **const out)
{
    CUTL_ASSERT(out != NULL, "The output pointer must not be NULL.");
    *out = NULL;

    lock_acquire();
    ensure_enumerated();
    if (index >= g_device_count)
    {
        lock_release();
        return HYBSOL_ERROR_NO_DEVICE;
    }
    if (index >= g_acquired_capacity)
    {
        const uint64_t wanted = index + 1;
        hybsol_opencl_device_t **const grown =
            (hybsol_opencl_device_t **)realloc(g_devices_acquired, (size_t)wanted * sizeof(*grown));
        if (grown == NULL)
        {
            lock_release();
            return HYBSOL_ERROR_OUT_OF_MEMORY;
        }
        memset(grown + g_acquired_capacity, 0, (size_t)(wanted - g_acquired_capacity) * sizeof(*grown));
        g_devices_acquired = grown;
        g_acquired_capacity = wanted;
    }
    hybsol_opencl_device_t *const existing = g_devices_acquired[index];
    lock_release();

    if (existing != NULL)
    {
        lock_acquire();
        ++existing->refs;
        const hybsol_opencl_device_t *const handle = existing;
        lock_release();
        *out = (hybsol_opencl_device_t *)handle;
        return HYBSOL_SUCCESS;
    }

    hybsol_opencl_device_t *device = NULL;
    const hybsol_result_t res = device_create(index, &device);
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
    free(device);
    *out = (hybsol_opencl_device_t *)winner;
    return HYBSOL_SUCCESS;
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
    if (device->index < g_acquired_capacity && g_devices_acquired[device->index] == device)
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
    free(device);
}

/* ------------------------------------------------------------------------- */
/* Kernels                                                                     */
/* ------------------------------------------------------------------------- */

/**
 * The device's program for one precision, built once and cached on the handle.
 *
 * The kernels are the transcription in ``gpu_kernels.h``, compiled with the
 * precision's definition and CL 1.2 -- the newest language every runtime this
 * backend cares about implements. The build log is worth more than the code,
 * so a failure says what the compiler said.
 */
static cl_program device_program(hybsol_opencl_device_t *const device, const hybsol_precision_t precision)
{
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

    // No -cl-std: the source is 1.2-compatible and the runtime's default is
    // its newest language, while some drivers -- NVIDIA's among them -- reject
    // the option outright. Only the variant's definition is passed.
    const char *const options = precision == HYBSOL_PRECISION_SINGLE ? "" : "-D HYBSOL_GPU_DOUBLE";
    status = clBuildProgram(program, 1, &device->id, options, NULL, NULL);
    if (status != CL_SUCCESS)
    {
        size_t log_size = 0;
        char *log = NULL;
        if (clGetProgramBuildInfo(program, device->id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size) == CL_SUCCESS &&
            log_size > 1)
        {
            log = (char *)malloc(log_size);
        }
        if (log != NULL)
        {
            clGetProgramBuildInfo(program, device->id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            snprintf(g_last_error, sizeof(g_last_error), "The %s kernels did not build: %s",
                     precision == HYBSOL_PRECISION_SINGLE ? "single" : "double", log);
            free(log);
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

    // One pass of the schedule is a few hundred block rows of small dense work,
    // so the interesting range is small: big enough to spread over every
    // compute unit, small enough that the last partial wave is not most of it.
    size_t wanted = 64;
    if (maximum < wanted)
    {
        wanted = maximum;
    }
    return wanted;
}

cl_kernel hybsol_opencl_kernel(hybsol_opencl_device_t *const device, const hybsol_precision_t precision,
                               const char *const name)
{
    CUTL_ASSERT(device != NULL, "The device must not be NULL.");

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
