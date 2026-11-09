/**
 * @file src/hip/device.c
 * Devices, streams and launch bounds of the HIP backend.
 *
 * Everything but the kernels is guarded by one mutex: the enumeration, the
 * handle table and the error note. The kernels are issued on each device's
 * single stream, which is in-order like the OpenCL backend's queue.
 */

#include "hip_internal.h"
#include <cutl/common_defs.h>

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

/** Room for the note; a runtime's complaint is truncated to fit, never rejected. */
#define ERROR_NOTE_BYTES 1024

static char g_last_error[ERROR_NOTE_BYTES];

void hybsol_hip_set_error(const char *const format, ...)
{
    va_list args;
    va_start(args, format);
    lock_acquire();
    vsnprintf(g_last_error, sizeof(g_last_error), format, args);
    lock_release();
    va_end(args);
}

void hybsol_hip_note_hip_error(const hipError_t error, const char *const what)
{
    if (error == hipSuccess)
        return;
    if (g_last_error[0] != '\0')
        return; // Something more specific already got here first.
    hybsol_hip_set_error("%s: HIP error %d (%s).", what, (int)error, hipGetErrorString(error));
}

const char *hybsol_hip_last_error(void)
{
    return g_last_error;
}

/* ------------------------------------------------------------------------- */
/* The measured default                                                      */
/* ------------------------------------------------------------------------- */

/**
 * The arithmetic AUTO names for double factors on a gfx12-family device.
 *
 * The fastest mode passing the selection harness gate (tests/c/
 * test_hip_precision.c) on the R9700 at every block size the harness
 * benches, 8/16/32: native FP64 clocked ahead of double-single and of
 * both Ozaki realizations on the harness's 128-block banded system --
 * these block sizes live on launch overhead, and native's per-work-item
 * walk is the cheapest. The harness table that picked it lives in that
 * test's commit message (a contended-GPU run; the idle rerun's numbers
 * supersede them, and this line flips with them if the ranking moves).
 * The split paths stay explicitly reachable by name.
 */
const hybsol_hip_arith_t HYBSOL_HIP_DEFAULT_FOR_GFX12 = HYBSOL_HIP_ARITH_NATIVE;

/* ------------------------------------------------------------------------- */
/* Enumeration                                                               */
/* ------------------------------------------------------------------------- */

/** One device as the enumeration lists it: the runtime's handle and its facts. */
typedef struct
{
    hipDevice_t id;
    hybsol_hip_device_kind_t kind;
    int supports_double;
    /** The device's name, kept for the order the enumeration is sorted into. */
    char name[128];
} listed_device_t;

/** The cached enumeration; empty until :c:func:`hybsol_hip_device_count` asks. */
static listed_device_t *g_devices = NULL;
static uint64_t g_device_count = 0;
static int g_enumerated = 0;

/**
 * The vendor of everything the HIP runtime enumerates.
 *
 * ``hipDeviceProp_t`` carries no vendor string the way OpenCL's
 * ``CL_DEVICE_VENDOR`` does; the runtime lists AMD's own GPUs only, so there
 * is exactly one answer and it is not worth a query.
 */
static const char HIP_VENDOR[] = "Advanced Micro Devices";

/**
 * The runtime's device UUID as a string, or an empty one where there is none.
 *
 * Preconditions: ``out`` holds at least 33 characters.
 */
static void device_uuid_string(const hipDevice_t device, char *const out, const size_t capacity)
{
    CUTL_ASSERT(out != NULL && capacity >= 33, "The output buffer must hold a 32-character UUID.");

    hipUUID uuid = {{0}};
    out[0] = '\0';
    if (hipDeviceGetUuid(&uuid, device) != hipSuccess)
    {
        return;
    }

    size_t at = 0;
    for (size_t i = 0; i < sizeof(uuid.bytes) && at + 3 < capacity; ++i)
    {
        at += (size_t)snprintf(out + at, capacity - at, "%02x", (unsigned char)uuid.bytes[i]);
    }
    out[at] = '\0';
}

/**
 * One device's kind, from its compute-unit count.
 *
 * HIP has no property that marks a device integrated, so the rule is the
 * device's shape: an iGPU has a handful of compute units (the 610M has 4,
 * the R9700 has 64), and ten or fewer of them means integrated.
 */
static hybsol_hip_device_kind_t device_kind(const hipDevice_t device)
{
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, device) != hipSuccess)
    {
        return HYBSOL_HIP_DEVICE_UNKNOWN;
    }
    if (prop.multiProcessorCount <= 16)
    {
        return HYBSOL_HIP_DEVICE_INTEGRATED;
    }
    return HYBSOL_HIP_DEVICE_GPU;
}

/**
 * Whether the device can do double arithmetic, which every decomposition needs.
 *
 * Every gfx target the runtime enumerates has native double: it is slow on
 * RDNA -- an entire phase's subject -- but it is there, unlike an OpenCL ICD
 * that can withhold fp64 outright.
 */
static int device_has_double(void)
{
    return 1;
}

/** The device's name, copied out of the properties, truncated to fit. */
static void device_name(const hipDevice_t device, char *const out, const size_t capacity)
{
    CUTL_ASSERT(out != NULL && capacity > 0, "The output buffer must be non-empty.");

    out[0] = '\0';
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, device) != hipSuccess)
    {
        return;
    }
    // The properties' name is 256 wide, the field's 128: the copy truncates on purpose.
    const size_t length = strlen(prop.name);
    const size_t take = length < capacity - 1 ? length : capacity - 1;
    memcpy(out, prop.name, take);
    out[take] = '\0';
}

/**
 * Append one device to the enumeration.
 *
 * Returns ``HYBSOL_ERROR_OUT_OF_MEMORY`` when the list cannot grow.
 */
static hybsol_result_t device_append(listed_device_t **const list, uint64_t *const count, const hipDevice_t id)
{
    CUTL_ASSERT(list != NULL && count != NULL, "The list and its count must not be NULL.");

    listed_device_t *const grown = (listed_device_t *)realloc(*list, (size_t)(*count + 1) * sizeof(**list));
    if (grown == NULL)
    {
        hybsol_hip_set_error("Out of memory listing the HIP devices.");
        return HYBSOL_ERROR_OUT_OF_MEMORY;
    }

    grown[*count].id = id;
    grown[*count].kind = device_kind(id);
    grown[*count].supports_double = device_has_double();
    device_name(id, grown[*count].name, sizeof(grown[*count].name));
    *list = grown;
    ++*count;
    return HYBSOL_SUCCESS;
}

/** Discrete GPUs before integrated ones, then by name: an index that does not move. */
static int device_before(const void *const a_ptr, const void *const b_ptr)
{
    const listed_device_t *const a = (const listed_device_t *)a_ptr;
    const listed_device_t *const b = (const listed_device_t *)b_ptr;
    const int a_discrete = a->kind == HYBSOL_HIP_DEVICE_GPU;
    const int b_discrete = b->kind == HYBSOL_HIP_DEVICE_GPU;
    if (a_discrete != b_discrete)
    {
        return a_discrete ? -1 : 1; // The card a caller means sorts first.
    }
    return strcmp(a->name, b->name); // strcmp itself, negative when a comes first.
}

/**
 * Collect every device the runtime offers, discrete GPUs first.
 *
 * No device at all is not a failure but a machine without one, which the
 * caller sees as a count of zero.
 */
static hybsol_result_t enumerate_devices(void)
{
    int n_runtime = 0;
    if (hipGetDeviceCount(&n_runtime) != hipSuccess || n_runtime <= 0)
    {
        hybsol_hip_set_error("The HIP runtime answered no devices.");
        return HYBSOL_ERROR_NO_DEVICE;
    }

    listed_device_t *list = NULL;
    uint64_t count = 0;
    hybsol_result_t res = HYBSOL_SUCCESS;

    // Two sweeps, so a discrete GPU sorts ahead of an integrated one.
    for (int discrete_pass = 1; discrete_pass >= 0; --discrete_pass)
    {
        for (int d = 0; d < n_runtime && res == HYBSOL_SUCCESS; ++d)
        {
            const hybsol_hip_device_kind_t kind = device_kind(d);
            const int discrete = kind == HYBSOL_HIP_DEVICE_GPU;
            if (discrete == discrete_pass)
            {
                res = device_append(&list, &count, (hipDevice_t)d);
            }
        }
    }

    if (res != HYBSOL_SUCCESS)
    {
        free(list);
        return res;
    }

    // Sort once: an index a caller writes down should still mean the same device tomorrow.
    // The order is total only while the names are distinct; two identical cards tie on name
    // alone, exactly as in the OpenCL backend, and the uuid is what disambiguates those.
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

uint64_t hybsol_hip_device_count(void)
{
    lock_acquire();
    ensure_enumerated();
    const uint64_t count = g_device_count;
    lock_release();
    return count;
}

hybsol_result_t hybsol_hip_device_info(const uint64_t index, hybsol_hip_device_info_t *const out)
{
    CUTL_ASSERT(out != NULL, "The output pointer must not be NULL.");

    lock_acquire();
    ensure_enumerated();
    if (index >= g_device_count)
    {
        lock_release();
        return HYBSOL_ERROR_NO_DEVICE;
    }
    const hipDevice_t id = g_devices[index].id;
    const hybsol_hip_device_kind_t kind = g_devices[index].kind;
    const int supports_double = g_devices[index].supports_double;
    lock_release();

    device_name(id, out->name, sizeof(out->name));
    snprintf(out->vendor, sizeof(out->vendor), "%s", HIP_VENDOR);
    out->kind = kind;
    out->supports_double = supports_double;
    device_uuid_string(id, out->uuid, sizeof(out->uuid));

    out->global_memory_bytes = 0;
    out->compute_units = 0;
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, id) == hipSuccess)
    {
        out->global_memory_bytes = (uint64_t)prop.totalGlobalMem;
        out->compute_units = (uint64_t)prop.multiProcessorCount;
    }
    return HYBSOL_SUCCESS;
}

/* ------------------------------------------------------------------------- */
/* Acquired devices                                                            */
/* ------------------------------------------------------------------------- */

/** The one handle per index, if it has been built; only touched under the lock. */
static hybsol_hip_device_t **g_devices_acquired = NULL;
static uint64_t g_acquired_capacity = 0;

/**
 * Open the stream for one enumerated device.
 *
 * Returns ``HYBSOL_ERROR_NO_DEVICE`` for an index no device has; the rest is
 * the runtime's answer.
 */
static hybsol_result_t device_create(const uint64_t index, hybsol_hip_device_t **const out)
{
    CUTL_ASSERT(out != NULL, "The output pointer must not be NULL.");

    lock_acquire();
    ensure_enumerated();
    if (index >= g_device_count)
    {
        lock_release();
        return HYBSOL_ERROR_NO_DEVICE;
    }
    const hipDevice_t id = g_devices[index].id;
    const int supports_double = g_devices[index].supports_double;
    lock_release();

    // The stream belongs to the current device, so make this one current first.
    if (hipSetDevice(id) != hipSuccess)
    {
        hybsol_hip_note_hip_error(hipGetLastError(), "Selecting the device for a stream");
        return HYBSOL_ERROR_DEVICE;
    }

    // In-order: each pass reads what the previous one wrote, as on the OpenCL queue.
    hipStream_t stream = NULL;
    if (hipStreamCreate(&stream) != hipSuccess)
    {
        hybsol_hip_note_hip_error(hipGetLastError(), "Creating a stream for the device");
        return HYBSOL_ERROR_DEVICE;
    }

    hybsol_hip_device_t *const device = (hybsol_hip_device_t *)calloc(1, sizeof(*device));
    if (device == NULL)
    {
        hipStreamDestroy(stream);
        return HYBSOL_ERROR_OUT_OF_MEMORY;
    }

    device->index = index;
    device->id = id;
    device->stream = stream;
    device->supports_double = supports_double;
    device->refs = 1;
    *out = device;
    return HYBSOL_SUCCESS;
}

hybsol_result_t hybsol_hip_device_acquire(const uint64_t index, hybsol_hip_device_t **const out)
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
        hybsol_hip_device_t **const grown = (hybsol_hip_device_t **)realloc(g_devices_acquired,
                                                                            (size_t)wanted * sizeof(*grown));
        if (grown == NULL)
        {
            lock_release();
            return HYBSOL_ERROR_OUT_OF_MEMORY;
        }
        memset(grown + g_acquired_capacity, 0, (size_t)(wanted - g_acquired_capacity) * sizeof(*grown));
        g_devices_acquired = grown;
        g_acquired_capacity = wanted;
    }
    hybsol_hip_device_t *const existing = g_devices_acquired[index];
    lock_release();

    if (existing != NULL)
    {
        lock_acquire();
        ++existing->refs;
        const hybsol_hip_device_t *const handle = existing;
        lock_release();
        *out = (hybsol_hip_device_t *)handle;
        return HYBSOL_SUCCESS;
    }

    hybsol_hip_device_t *device = NULL;
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
    const hybsol_hip_device_t *const winner = g_devices_acquired[index];
    lock_release();

    hipStreamDestroy(device->stream);
    free(device);
    *out = (hybsol_hip_device_t *)winner;
    return HYBSOL_SUCCESS;
}

void hybsol_hip_device_release(hybsol_hip_device_t *const device)
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

    hipStreamDestroy(device->stream);
    free(device);
}

/* ------------------------------------------------------------------------- */
/* Launch bounds                                                               */
/* ------------------------------------------------------------------------- */

/** The block size the backend is happy with at most; see the OpenCL backend's same cap. */
#define HYBSOL_HIP_WANTED_BLOCK 64

size_t hybsol_hip_work_group_size(const hybsol_hip_device_t *const device, const void *const kernel)
{
    CUTL_ASSERT(device != NULL && kernel != NULL, "The device and the kernel must not be NULL.");

    // The occupancy API answers for the current device.
    (void)hipSetDevice(device->id);
    int grid_suggestion = 0;
    int block_suggestion = 0;
    if (hipOccupancyMaxPotentialBlockSize(&grid_suggestion, &block_suggestion, kernel, 0, 0) != hipSuccess ||
        block_suggestion <= 0)
    {
        // The API will not say: the bounded default, never the runtime's maximum.
        return HYBSOL_HIP_WANTED_BLOCK;
    }

    // Small enough to spread over every compute unit, large enough that the tail costs little.
    return (size_t)(block_suggestion > HYBSOL_HIP_WANTED_BLOCK ? HYBSOL_HIP_WANTED_BLOCK : block_suggestion);
}

void hybsol_hip_work_group_size_2d(const hybsol_hip_device_t *const device, const void *const kernel, size_t *const x,
                                   size_t *const y)
{
    CUTL_ASSERT(device != NULL && kernel != NULL && x != NULL && y != NULL,
                "The device, the kernel and the output pointers must not be NULL.");

    // The occupancy API answers for the current device.
    (void)hipSetDevice(device->id);
    int grid_suggestion = 0;
    int block_suggestion = 0;
    if (hipOccupancyMaxPotentialBlockSize(&grid_suggestion, &block_suggestion, kernel, 0, 0) != hipSuccess ||
        block_suggestion <= 0)
    {
        // The API will not say: the bounded default, and one dimension of it.
        *x = HYBSOL_HIP_WANTED_BLOCK;
        *y = 1;
        return;
    }

    // A two-dimensional launch is bounded by the product, so the budget is split, not repeated.
    const size_t maximum = (size_t)block_suggestion;
    size_t first = maximum < HYBSOL_HIP_WANTED_BLOCK ? maximum : HYBSOL_HIP_WANTED_BLOCK;
    size_t second = maximum / first;
    if (second > HYBSOL_HIP_WANTED_BLOCK)
    {
        second = HYBSOL_HIP_WANTED_BLOCK;
    }
    *x = first;
    *y = second == 0 ? 1 : second;
}
