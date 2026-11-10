/**
 * @file src/hip/device.c
 * Devices, streams and launch bounds of the HIP backend.
 *
 * Everything but the kernels is guarded by one mutex: the enumeration, the
 * handle table and the error note. The kernels are issued on each device's
 * single stream, which is in-order like the OpenCL backend's queue.
 */

#include "hip_internal.h"
#include "core/internal.h"
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

/* For callers that already hold the lock, as the device table's refusals do. */
static void set_error_locked(const char *const format, ...)
{
    va_list args;
    va_start(args, format);
    vsnprintf(g_last_error, sizeof(g_last_error), format, args);
    va_end(args);
}

/*
 * Drop the note. The note answers "why did *this* call fail", so every entry
 * point that starts an operation of its own opens by clearing it: a note from
 * an earlier refusal would otherwise be quoted against an unrelated one,
 * naming whichever device failed last time rather than this one.
 */
void hybsol_hip_clear_error(void)
{
    lock_acquire();
    g_last_error[0] = '\0';
    lock_release();
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
 * The Ozaki modes stay explicitly reachable by name.
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
    /** The runtime index the device was listed at; the tie-break below the name. */
    int ordinal;
} listed_device_t;

/*
 * A machine has a handful of devices, so the enumeration lives in a fixed
 * table rather than a growable list: the cache never allocates. Exceeding
 * the cap is a refused enumeration with a note, never a silent truncation.
 */
#define HYBSOL_HIP_MAX_LISTED_DEVICES 32

/** The cached enumeration; empty until :c:func:`hybsol_hip_device_count` asks. */
static listed_device_t g_devices[HYBSOL_HIP_MAX_LISTED_DEVICES];
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
    for (size_t i = 0; i < sizeof(uuid.bytes) && at + 3 <= capacity; ++i)
    {
        at += (size_t)snprintf(out + at, capacity - at, "%02x", (unsigned char)uuid.bytes[i]);
    }
    out[at] = '\0';
}

/**
 * One device's kind, from its compute-unit count.
 *
 * HIP has no property that marks a device integrated, so the rule is the
 * device's shape: an iGPU has a handful of compute units (the 9900X's
 * reports 1, the R9700 reports 32), and sixteen or fewer of them means
 * integrated.
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
 * Append one device to the enumeration at ``count``, remembering the runtime
 * index ``ordinal`` it was read at: two same-model cards tie on name, and
 * only the runtime order tells them apart.
 *
 * Returns ``HYBSOL_ERROR_DEVICE`` when the fixed table is full; the cap only
 * binds on a machine with more devices than this backend has ever seen.
 */
static hybsol_result_t device_append(const uint64_t count, const hipDevice_t id, const int ordinal)
{
    if (count >= HYBSOL_HIP_MAX_LISTED_DEVICES)
    {
        // Reached from the enumeration, which holds the lock.
        set_error_locked("The HIP runtime lists more than %d devices; the enumeration is capped at that many.",
                         HYBSOL_HIP_MAX_LISTED_DEVICES);
        return HYBSOL_ERROR_DEVICE;
    }

    g_devices[count].id = id;
    g_devices[count].kind = device_kind(id);
    g_devices[count].supports_double = device_has_double();
    device_name(id, g_devices[count].name, sizeof(g_devices[count].name));
    g_devices[count].ordinal = ordinal;
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
    const int by_name = strcmp(a->name, b->name); // strcmp itself, negative when a comes first.
    if (by_name != 0)
    {
        return by_name;
    }
    // Two cards of one model tie on the name; the runtime order they were
    // listed in breaks the tie, and it is the same in every process on the
    // same hardware -- what keeps qsort's total order, and the caller's
    // index, pinned down.
    return a->ordinal < b->ordinal ? -1 : (a->ordinal > b->ordinal ? 1 : 0);
}

/**
 * Collect every device the runtime offers, discrete GPUs first.
 *
 * No device at all is not a failure but a machine without one, which the
 * caller sees as a count of zero. The runtime's devices are named by ordinal,
 * so this allocates nothing at all: the table is static.
 */
static hybsol_result_t enumerate_devices(void)
{
    int n_runtime = 0;
    if (hipGetDeviceCount(&n_runtime) != hipSuccess || n_runtime <= 0)
    {
        // The caller holds the lock; the locking setter here would self-deadlock.
        set_error_locked("The HIP runtime answered no devices.");
        return HYBSOL_ERROR_NO_DEVICE;
    }

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
                res = device_append(count, (hipDevice_t)d, d);
                if (res == HYBSOL_SUCCESS)
                {
                    ++count;
                }
            }
        }
    }

    if (res != HYBSOL_SUCCESS)
    {
        g_device_count = 0;
        return res;
    }

    // Sort once: an index a caller writes down should still mean the same device tomorrow.
    // Two cards of one model tie on name alone, as in the OpenCL backend; the
    // runtime ordinal they were listed at breaks the tie, so the same
    // hardware sorts to the same indices in every process.
    if (count > 1)
    {
        qsort(g_devices, (size_t)count, sizeof(*g_devices), device_before);
    }

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
static hybsol_hip_device_t *g_devices_acquired[HYBSOL_HIP_MAX_LISTED_DEVICES];

/**
 * Open the stream for one enumerated device.
 *
 * Returns ``HYBSOL_ERROR_NO_DEVICE`` for an index no device has; the rest is
 * the runtime's answer. The handle itself comes from ``allocator``, which the
 * acquisition's caller owns and which the handle keeps for its own later
 * allocations.
 */
static hybsol_result_t device_create(const uint64_t index, const hybsol_allocator_t *const allocator,
                                     hybsol_hip_device_t **const out)
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

    hybsol_hip_device_t *const device = (hybsol_hip_device_t *)hybsol_alloc(allocator, sizeof(*device));
    if (device == NULL)
    {
        hybsol_hip_set_error("Out of memory building the device handle.");
        hipStreamDestroy(stream);
        return HYBSOL_ERROR_OUT_OF_MEMORY;
    }
    memset(device, 0, sizeof(*device));

    device->index = index;
    device->id = id;
    device->stream = stream;
    device->supports_double = supports_double;
    device->refs = 1;
    device->allocator = allocator;
    *out = device;
    return HYBSOL_SUCCESS;
}

hybsol_result_t hybsol_hip_device_acquire(const uint64_t index, const hybsol_allocator_t *const allocator,
                                          hybsol_hip_device_t **const out)
{
    CUTL_ASSERT(out != NULL, "The output pointer must not be NULL.");
    CUTL_ASSERT(allocator != NULL, "The allocator must not be NULL; pass &CUTL_STD_ALLOCATOR for the default.");
    *out = NULL;
    hybsol_hip_clear_error();

    lock_acquire();
    ensure_enumerated();
    if (index >= g_device_count)
    {
        set_error_locked("No HIP device has index %llu; the runtime lists %llu.", (unsigned long long)index,
                         (unsigned long long)g_device_count);
        lock_release();
        return HYBSOL_ERROR_NO_DEVICE;
    }
    // g_device_count never exceeds the table, so a passing index is inside it too.
    CUTL_ASSERT(index < HYBSOL_HIP_MAX_LISTED_DEVICES, "The handle table is %d wide but was asked for index %llu.",
                HYBSOL_HIP_MAX_LISTED_DEVICES, (unsigned long long)index);
    hybsol_hip_device_t *existing = g_devices_acquired[index];
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

    hybsol_hip_device_t *device = NULL;
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
    const hybsol_hip_device_t *const winner = g_devices_acquired[index];
    lock_release();

    hipStreamDestroy(device->stream);
    hybsol_free(allocator, device);
    *out = (hybsol_hip_device_t *)winner;
    return HYBSOL_SUCCESS;
}

void hybsol_hip_device_retain(hybsol_hip_device_t *const device)
{
    CUTL_ASSERT(device != NULL, "The device handle must not be NULL.");
    CUTL_ASSERT(device->refs > 0, "A released device handle must not be retained again; it had %d references.",
                device->refs);

    lock_acquire();
    ++device->refs;
    lock_release();
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
    // Creation only ever hands out an index the table is wide enough for.
    CUTL_ASSERT(device->index < HYBSOL_HIP_MAX_LISTED_DEVICES,
                "The handle table is %d wide but a device claims index %llu.", HYBSOL_HIP_MAX_LISTED_DEVICES,
                (unsigned long long)device->index);
    if (g_devices_acquired[device->index] == device)
    {
        g_devices_acquired[device->index] = NULL;
    }
    lock_release();

    hipStreamDestroy(device->stream);
    hybsol_free(device->allocator, device);
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
