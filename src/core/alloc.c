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
/* The fill-in pool                                                            */
/* ------------------------------------------------------------------------- */

/*
 * A decomposition allocates from inside OpenMP parallel regions, so a caller
 * supplying an allocator that is not thread-safe would be raced. Instead the
 * fill-in comes out of one pool, allocated here before the first region starts
 * and sized exactly by `hybsol_fill_plan_compute`.
 *
 * Threads share the pool, so a carve is one relaxed atomic bump: no lock, and
 * no call into the system's allocator on the common path. Every carve keeps a
 * small header recording its size, so a grow can copy the right amount -- the
 * same contract the bump regions had.
 *
 * The memory outlives the decomposition: the entries carved out of it are the
 * system's fill-in, released with the system rather than individually.
 */

/*
 * Prefix on every carve, so a grow knows how much of the old one to copy. Its
 * size has to match HYBSOL_POOL_HDR_BYTES, which the symbolic pass adds when
 * sizing the pool.
 */
typedef struct
{
    size_t size;
    size_t pad;
} hybsol_pool_hdr_t;

static size_t align_up(const size_t value)
{
    return (value + (HYBSOL_REGION_ALIGN - 1)) & ~(size_t)(HYBSOL_REGION_ALIGN - 1);
}

/**
 * Carve out of the pool, or fall back to the allocator once it is exhausted.
 *
 * The bump is never rolled back: `pool_used` is a single cursor every thread
 * reads, so undoing it would hand the same bytes to a second thread. Running
 * past the end therefore just means the tail of the pool is unused, and the
 * allocation comes from the system allocator instead.
 */
static void *pool_allocate(void *const state, const size_t size)
{
    hybsol_system_t *const sys = (hybsol_system_t *)state;
    const size_t payload = sizeof(hybsol_pool_hdr_t) + align_up(size);

    const size_t at = __atomic_fetch_add(&sys->pool_used, payload, __ATOMIC_RELAXED);
    if (at + payload <= sys->pool_size)
    {
        unsigned char *const base = sys->pool + at;
        ((hybsol_pool_hdr_t *)base)->size = size;
        ((hybsol_pool_hdr_t *)base)->pad = 0;
        return base + sizeof(hybsol_pool_hdr_t);
    }

    // Exact sizing should make this unreachable. It is taken under a lock
    // because it is the one place a parallel region can reach the system's
    // allocator, and a caller may have supplied one that is not thread-safe.
    void *fallback;
#pragma omp critical(hybsol_pool_overflow)
    {
        fallback = hybsol_alloc(sys->allocator, size);
    }
    return fallback;
}

static void *pool_reallocate(void *const state, void *const ptr, const size_t size)
{
    if (ptr == NULL)
        return pool_allocate(state, size);

    const hybsol_pool_hdr_t *const hdr =
        (const hybsol_pool_hdr_t *)((const unsigned char *)ptr - sizeof(hybsol_pool_hdr_t));
    void *const fresh = pool_allocate(state, size);
    if (fresh != NULL)
        memcpy(fresh, ptr, hdr->size < size ? hdr->size : size);
    return fresh;
}

static void pool_deallocate(void *const state, void *const ptr)
{
    HYBSOL_MARK_USED(state);
    HYBSOL_MARK_USED(ptr);
    /*
     * Nothing inside a parallel region releases anything: the rows are grown to
     * their peak before the first region starts, so the release in
     * hybsol_row_reserve is never reached from one. Memory the overflow path
     * handed out is freed by whoever owns it, not here.
     */
}

hybsol_result_t hybsol_pool_open(hybsol_system_t *const sys, const hybsol_fill_plan_t *const plan)
{
    CUTL_ASSERT(sizeof(hybsol_pool_hdr_t) == HYBSOL_POOL_HDR_BYTES,
                "The pool header and the size the symbolic pass adds have drifted apart.");

    hybsol_pool_free(sys);

    // Over-allocate by the alignment so the base can be rounded up to it; cutl
    // only guarantees max_align_t, and every carve is a multiple of the
    // granularity, so aligning the base aligns all of them.
    const size_t want = plan->pool_bytes + HYBSOL_REGION_ALIGN;
    unsigned char *const raw = hybsol_alloc(sys->allocator, want);
    if (raw == NULL)
        return HYBSOL_ERROR_OUT_OF_MEMORY;

    sys->pool_raw = raw;
    sys->pool = (unsigned char *)align_up((size_t)raw);
    sys->pool_size = plan->pool_bytes;
    sys->pool_used = 0;
    sys->pool_alloc = (cutl_allocator_t){
        .state = sys,
        .allocate = pool_allocate,
        .deallocate = pool_deallocate,
        .reallocate = pool_reallocate,
    };
    sys->pool_armed = 1;
    return HYBSOL_SUCCESS;
}

void hybsol_pool_close(hybsol_system_t *const sys)
{
    // `pool` deliberately stays: the fill-in carved out of it is still live, and
    // hybsol_ptr_is_pooled has to keep recognising it until the system dies.
    sys->pool_armed = 0;
    sys->pool_used = 0;
}

void hybsol_pool_free(hybsol_system_t *const sys)
{
    hybsol_free(sys->allocator, sys->pool_raw);
    sys->pool_raw = NULL;
    sys->pool = NULL;
    sys->pool_size = 0;
    sys->pool_used = 0;
    sys->pool_armed = 0;
}

int hybsol_ptr_is_pooled(const hybsol_system_t *const sys, const void *const ptr)
{
    if (ptr == NULL || sys->pool == NULL)
        return 0;
    const unsigned char *const p = (const unsigned char *)ptr;
    return p >= sys->pool && p < sys->pool + sys->pool_size;
} /* ------------------------------------------------------------------------- */
/* Decomposition scratch                                                       */
/* ------------------------------------------------------------------------- */

/*
 * A decomposition needs three per-row arrays and one scratch block per thread.
 * They are short-lived and entirely internal, so the caller can own them: size
 * the buffer with hybsol_workspace_bytes, hand it over, and no allocation
 * happens at all before the parallel regions. Everything lands in one
 * contiguous aligned block, so a single allocation -- or a single NumPy
 * buffer -- is all it takes.
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
 * The buffer was sized by :c:func:`hybsol_workspace_bytes`, so the arithmetic
 * here matches what that computed. The magic is written once the buffer is
 * known good.
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

/** Scratch block belonging to the calling thread; a pure offset into ``ws``. */
void *hybsol_workspace_scratch(const hybsol_workspace_t *const ws)
{
    const uint64_t t = (uint64_t)HYBSOL_THREAD_NUM();
    if (ws == NULL || t >= ws->n_threads)
        return NULL;

    unsigned char *const base = (unsigned char *)ws;
    return base + ws->off_scratch + (size_t)t * ws->scratch_stride;
}
