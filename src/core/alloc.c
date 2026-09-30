/**
 * @file core/alloc.c
 * Result descriptions and the per-thread bump regions a decomposition
 * allocates its fill-in from.
 */

#include "internal.h"

const char *hybsol_result_str(const hybsol_result_t result)
{
    switch (result)
    {
    case HYBSOL_SUCCESS:
        return "success";
    case HYBSOL_ERROR_OUT_OF_MEMORY:
        return "out of memory";
    case HYBSOL_ERROR_EMPTY_ROW:
        return "row has no entries";
    case HYBSOL_ERROR_NO_MORE_COLUMNS:
        return "row has no entries after the given column";
    case HYBSOL_ERROR_MAX_COLORS:
        return "maximum number of colors exceeded";
    case HYBSOL_ERROR_SINGULAR:
        return "zero pivot in LU decomposition";
    case HYBSOL_ERROR_SYSTEM_INVALID:
        return "block system is not valid";
    case HYBSOL_ERROR_NOT_DECOMPOSED:
        return "system has not been decomposed";
    case HYBSOL_ERROR_ALREADY_DECOMPOSED:
        return "system has already been decomposed";
    case HYBSOL_ERROR_INTERNAL:
        return "internal error";
    }
    return "unknown result code";
}

/* ------------------------------------------------------------------------- */
/* Per-thread bump regions                                                     */
/* ------------------------------------------------------------------------- */

/*
 * A decomposition allocates from inside OpenMP parallel regions, so a caller
 * supplying an allocator that is not thread-safe would be raced. Each thread
 * bumps against a region of its own instead: the fast path touches only that
 * thread's cursor, so it needs no lock, no atomic, and never calls the system's
 * allocator at all.
 *
 * Regions start large and double when they run out, so growth stays logarithmic
 * in however much fill-in a pattern turns out to produce, and the one place
 * that still calls the allocator is hit a handful of times rather than once per
 * entry.
 *
 * A region is released only when the system is destroyed: entries carved out of
 * it outlive the decomposition.
 */

#define HYBSOL_REGION_MIN_BYTES (256u * 1024u)

/** Every block carries this, so a grow knows how much of the old one to copy. */
typedef struct hybsol_region_hdr
{
    size_t size;
    size_t pad;
} hybsol_region_hdr_t;

static size_t align_up(const size_t value)
{
    return (value + (HYBSOL_REGION_ALIGN - 1)) & ~(size_t)(HYBSOL_REGION_ALIGN - 1);
}

/** Carve ``size`` bytes off ``region``, or ``NULL`` when it is exhausted. */
static void *region_allocate(hybsol_region_t *const region, const size_t size)
{
    const size_t want = sizeof(hybsol_region_hdr_t) + align_up(size);
    if (want < size) /* the size plus its header wrapped around */
        return NULL;
    if (region->used > region->size || want > region->size - region->used)
        return NULL;

    unsigned char *const at = region->base + region->used;
    region->used += want;
    hybsol_region_hdr_t *const hdr = (hybsol_region_hdr_t *)at;
    hdr->size = size;
    hdr->pad = 0;
    return at + sizeof(hybsol_region_hdr_t);
}

/**
 * Add a region to the system's list and report its index.
 *
 * The only place a decomposition still touches the system's allocator, so it is
 * serialized: a caller may have supplied an allocator that is not thread-safe.
 */
static hybsol_result_t region_push(hybsol_system_t *const sys, const size_t bytes, uint64_t *const out_idx)
{
    hybsol_result_t res = HYBSOL_SUCCESS;
#pragma omp critical(hybsol_region_push)
    {
        if (sys->n_regions == sys->regions_capacity)
        {
            const uint64_t next = sys->regions_capacity ? sys->regions_capacity * 2 : 8;
            hybsol_region_t *const grown = hybsol_grow(sys->allocator, sys->regions, (size_t)next * sizeof(*grown));
            if (grown == NULL)
            {
                res = HYBSOL_ERROR_OUT_OF_MEMORY;
            }
            else
            {
                sys->regions = grown;
                sys->regions_capacity = next;
            }
        }
        if (res == HYBSOL_SUCCESS)
        {
            unsigned char *const base = hybsol_alloc(sys->allocator, bytes);
            if (base == NULL)
            {
                res = HYBSOL_ERROR_OUT_OF_MEMORY;
            }
            else
            {
                sys->regions[sys->n_regions] = (hybsol_region_t){.base = base, .size = bytes, .used = 0};
                *out_idx = sys->n_regions;
                sys->n_regions += 1;
            }
        }
    }
    return res;
}

/**
 * Carve ``size`` bytes for one thread, doubling its region when exhausted.
 *
 * Only ever called on the owning thread, so the cursor is not contended. The
 * region it runs out of is kept alive, not freed: entries carved out of it are
 * still in use, which is why the list holds superseded regions too.
 */
static void *region_take(hybsol_thread_alloc_t *const ta, const size_t size)
{
    void *const got = region_allocate(&ta->sys->regions[ta->region], size);
    if (got != NULL)
        return got;

    const size_t current = ta->sys->regions[ta->region].size;
    if (region_push(ta->sys, current ? current * 2 : HYBSOL_REGION_MIN_BYTES, &ta->region) != HYBSOL_SUCCESS)
        return NULL;

    return region_allocate(&ta->sys->regions[ta->region], size);
}

static void *thread_allocate(void *const state, const size_t size)
{
    return region_take((hybsol_thread_alloc_t *)state, size);
}

static void *thread_reallocate(void *const state, void *const ptr, const size_t size)
{
    if (ptr == NULL)
        return region_take((hybsol_thread_alloc_t *)state, size);

    const hybsol_region_hdr_t *const hdr =
        (const hybsol_region_hdr_t *)((const unsigned char *)ptr - sizeof(hybsol_region_hdr_t));
    void *const fresh = region_take((hybsol_thread_alloc_t *)state, size);
    if (fresh != NULL)
        memcpy(fresh, ptr, hdr->size < size ? hdr->size : size);
    return fresh;
}

static void thread_deallocate(void *const state, void *const ptr)
{
    HYBSOL_MARK_USED(state);
    HYBSOL_MARK_USED(ptr);
    /* A region is released whole, when the system is destroyed. */
}

hybsol_result_t hybsol_thread_allocs_open(hybsol_system_t *const sys, const uint64_t n_threads)
{
    hybsol_regions_free(sys);

    if (n_threads == 0)
        return HYBSOL_SUCCESS;

    sys->thread_allocs = hybsol_alloc(sys->allocator, (size_t)n_threads * sizeof(*sys->thread_allocs));
    if (sys->thread_allocs == NULL)
        return HYBSOL_ERROR_OUT_OF_MEMORY;
    sys->n_thread_allocs = 0;

    for (uint64_t t = 0; t < n_threads; ++t)
    {
        hybsol_thread_alloc_t *const ta = sys->thread_allocs + t;
        ta->sys = sys;
        ta->thread = t;
        ta->region = 0;
        ta->alloc = (cutl_allocator_t){
            .state = ta, .allocate = thread_allocate, .deallocate = thread_deallocate, .reallocate = thread_reallocate};
        if (region_push(sys, HYBSOL_REGION_MIN_BYTES, &ta->region) != HYBSOL_SUCCESS)
        {
            hybsol_regions_free(sys);
            return HYBSOL_ERROR_OUT_OF_MEMORY;
        }
    }

    sys->n_thread_allocs = n_threads;
    return HYBSOL_SUCCESS;
}

void hybsol_thread_allocs_done(hybsol_system_t *const sys)
{
    hybsol_free(sys->allocator, sys->thread_allocs);
    sys->thread_allocs = NULL;
    sys->n_thread_allocs = 0;
}

void hybsol_regions_free(hybsol_system_t *const sys)
{
    for (uint64_t r = 0; r < sys->n_regions; ++r)
        hybsol_free(sys->allocator, sys->regions[r].base);
    sys->n_regions = 0;
    sys->regions_capacity = 0;

    hybsol_free(sys->allocator, sys->regions);
    sys->regions = NULL;
}

int hybsol_ptr_is_pooled(const hybsol_system_t *const sys, const void *const ptr)
{
    if (ptr == NULL)
        return 0;
    const unsigned char *const p = (const unsigned char *)ptr;
    for (uint64_t r = 0; r < sys->n_regions; ++r)
    {
        if (sys->regions[r].base != NULL && p >= sys->regions[r].base &&
            p < sys->regions[r].base + sys->regions[r].size)
            return 1;
    }
    return 0;
}
/* ------------------------------------------------------------------------- */
/* Decomposition scratch                                                       */
/* ------------------------------------------------------------------------- */

/*
 * A decomposition needs three per-row arrays and one scratch block per thread.
 * They are short-lived and entirely internal, so the caller can own them: size
 * the buffer with hybsol_workspace_bytes, hand it over, and no allocation
 * happens at all before the parallel regions.
 *
 * Everything lands in one contiguous block, aligned, so a single allocation (or
 * a single NumPy buffer from Python) is all it takes.
 */

size_t hybsol_workspace_bytes(const hybsol_system_t *const sys, const uint64_t n_threads)
{
    if (sys == NULL)
        return 0;

    const uint64_t threads = (uint64_t)hybsol_resolve_threads(n_threads);

    /* Every intermediate in an elimination is at most a block times a block. */
    uint64_t max_block = 1;
    for (uint64_t i = 0; i < sys->n; ++i)
    {
        const uint64_t size = hybsol_block_size(sys, i);
        if (size > max_block)
            max_block = size;
    }
    const size_t scratch_stride = align_up((size_t)max_block * (size_t)max_block * hybsol_scalar_size(sys->precision));

    size_t total = sizeof(hybsol_workspace_t);
    total += align_up((size_t)sys->n * sizeof(target_row_t));
    total += align_up((size_t)sys->n * sizeof(uint64_t));
    total += align_up((size_t)sys->n * sizeof(hybsol_result_t));
    total += scratch_stride * threads;
    return align_up(total);
}

/**
 * Point the header at the arrays laid out behind it.
 *
 * Only ever called on a buffer the caller sized with
 * :c:func:`hybsol_workspace_bytes`, so the arithmetic here matches what that
 * computed. The magic is written afterwards, once the buffer is known good.
 */
void hybsol_workspace_bind(hybsol_workspace_t *const ws, void *const buffer, const size_t buffer_bytes,
                           const hybsol_system_t *const sys, const uint64_t threads)
{
    uint64_t max_block = 1;
    for (uint64_t i = 0; i < sys->n; ++i)
    {
        const uint64_t size = hybsol_block_size(sys, i);
        if (size > max_block)
            max_block = size;
    }

    ws->magic = HYBSOL_WORKSPACE_MAGIC;
    ws->n = sys->n;
    ws->n_threads = threads;
    ws->scratch_stride = align_up((size_t)max_block * (size_t)max_block * hybsol_scalar_size(sys->precision));
    ws->total_bytes = buffer_bytes;

    HYBSOL_MARK_USED(buffer);
    size_t at = align_up(sizeof(hybsol_workspace_t));
    ws->off_target_status = at;
    at += align_up((size_t)sys->n * sizeof(target_row_t));
    ws->off_ready = at;
    at += align_up((size_t)sys->n * sizeof(uint64_t));
    ws->off_results = at;
    at += align_up((size_t)sys->n * sizeof(hybsol_result_t));
    ws->off_scratch = at;
}

/**
 * Scratch block belonging to the calling thread.
 *
 * Called from inside the parallel regions, so it must be a pure offset into a
 * buffer the caller already owns.
 */
void *hybsol_workspace_scratch(const hybsol_workspace_t *const ws)
{
    const uint64_t t = (uint64_t)HYBSOL_THREAD_NUM();
    if (ws == NULL || t >= ws->n_threads)
        return NULL;

    unsigned char *const base = (unsigned char *)ws;
    return base + ws->off_scratch + (size_t)t * ws->scratch_stride;
}
