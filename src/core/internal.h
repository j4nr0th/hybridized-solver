/**
 * @file core/internal.h
 * Private declarations shared by the hybsol core translation units.
 *
 * Nothing here is public: consumers see only ``include/hybsol/``. This file
 * may pull in whatever the implementation needs; it never pulls in Python.
 */

#ifndef HYBSOL_CORE_INTERNAL_H
#define HYBSOL_CORE_INTERNAL_H

#include <hybsol/hybsol.h>

#include <cutl/allocators.h>
#include <cutl/common_defs.h>

#include <limits.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#define HYBSOL_DEFAULT_NUM_THREADS omp_get_max_threads()
#else
/* Without OpenMP the pragmas below are inert and everything runs serially. */
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#define HYBSOL_DEFAULT_NUM_THREADS 1
#endif

#ifdef _OPENMP
#define HYBSOL_THREAD_NUM() omp_get_thread_num()
#else
#define HYBSOL_THREAD_NUM() 0
#endif

/* ------------------------------------------------------------------------- */
/* Assertions                                                                 */
/* ------------------------------------------------------------------------- */
/*
 * Preconditions are checked with cutl's CUTL_ASSERT, which reports the file,
 * line, function and the failed condition before aborting.
 *
 * Without CUTL_ENABLE_ASSERTS it degrades to CUTL_ASSUME, which lets the
 * optimizer assume the condition holds and may delete the expression
 * outright. Never put a side effect in a CUTL_ASSERT condition: write it to
 * a variable first and assert on that variable.
 */
#ifndef CUTL_ENABLE_ASSERTS
#define CUTL_ENABLE_ASSERTS 0
#endif

/* ------------------------------------------------------------------------- */
/* Allocation                                                                 */
/* ------------------------------------------------------------------------- */

/*
 * The allocation helpers every core file goes through. They take the
 * allocator explicitly rather than reading a global, which is what lets two
 * systems use different allocators and keeps concurrent use safe.
 *
 * A zero size yields NULL, matching cutl's allocators, so the `== NULL`
 * out-of-memory checks callers make stay correct.
 */
static inline void *hybsol_alloc(const cutl_allocator_t *const alloc, const size_t size)
{
    return cutl_alloc(alloc, size);
}

static inline void *hybsol_grow(const cutl_allocator_t *const alloc, void *const ptr, const size_t size)
{
    return cutl_realloc(alloc, ptr, size);
}

static inline void hybsol_free(const cutl_allocator_t *const alloc, void *const ptr)
{
    cutl_dealloc(alloc, ptr);
}

/* ------------------------------------------------------------------------- */
/* The fill-in pool                                                           */
/* ------------------------------------------------------------------------- */

/*
 * A decomposition allocates from inside OpenMP parallel regions, so a caller
 * supplying an allocator that is not thread-safe would be raced. Instead, every
 * allocation a decomposition makes comes out of one pool the library allocates
 * before the first region starts, sized exactly by the symbolic pass in
 * `hybsol_fill_plan_compute`.
 *
 * Threads share that pool, so the carve is a single relaxed atomic bump: no
 * lock, and no call into the system's allocator at all on the common path. A
 * pool that turns out too small -- which would mean the symbolic pass and the
 * real elimination disagree -- falls back to the allocator under a lock, so a
 * divergence costs speed rather than correctness.
 *
 * The pool is spelled as a `cutl_allocator_t`, so every existing
 * `hybsol_alloc(alloc, size)` call site keeps its shape and only the allocator
 * pointer changes between phases.
 */

/** Granularity the pool carves on; wide enough for any scalar stored. */
#define HYBSOL_REGION_ALIGN 64u

/**
 * Bytes of prefix on every pool carve.
 *
 * The symbolic pass adds this to each payload when sizing the pool, so it has
 * to match the header `pool_allocate` actually writes; ``alloc.c`` builds it
 * from the same expression.
 */
#define HYBSOL_POOL_HDR_BYTES (2u * sizeof(size_t))

/* ------------------------------------------------------------------------- */
/* Decomposition scratch                                                      */
/* ------------------------------------------------------------------------- */

/** How far along a row is between elimination passes. */
typedef enum
{
    TARGET_FREE,
    TARGET_IN_USE,
    TARGET_DONE,
} target_status_t;

typedef struct
{
    target_status_t status;
    /** Column of the row this one is waiting on before it can be eliminated. */
    uint64_t idx_src_needed;
} target_row_t;

/** Distinguishes a bound workspace from an arbitrary or stale buffer. */
#define HYBSOL_WORKSPACE_MAGIC UINT64_C(0x687962736f6c7731) /* "hybsolw1" */

/** Layout of a bound workspace. The arrays follow it in one contiguous block. */
typedef struct hybsol_workspace
{
    uint64_t magic;
    /** ``sys->n`` the buffer was sized for. */
    uint64_t n;
    /** Resolved thread count the buffer was sized for. */
    uint64_t n_threads;
    /** Bytes each per-thread scratch block holds. */
    size_t scratch_stride;
    /** Total the caller had to supply. */
    size_t total_bytes;
    /** Byte offsets of each array within the buffer. */
    size_t off_target_status;
    size_t off_ready;
    size_t off_results;
    size_t off_scratch;
} hybsol_workspace_t;

/** Record the layout of the arrays that follow the header in ``buffer``. */
void hybsol_workspace_bind(hybsol_workspace_t *ws, void *buffer, size_t buffer_bytes, const hybsol_system_t *sys,
                           uint64_t threads);

/** The scratch block belonging to the calling thread, or ``NULL`` if it has none. */
void *hybsol_workspace_scratch(const hybsol_workspace_t *ws);

/* ------------------------------------------------------------------------- */
/* Storage layout                                                             */
/* ------------------------------------------------------------------------- */

/** One stored block: its column index followed by row-major values. */
typedef struct hybsol_row_entry
{
    /** Block column index; strictly increasing within a row. */
    uint64_t col;
    /**
     * ``block_size(row) * block_size(col)`` elements of the owning system's
     * precision, sized at allocation time.
     *
     * Raw bytes rather than a flexible array of a specific type: the entry
     * does not know which precision it holds, so readers cast it using
     * :c:func:`hybsol_scalar_size`. The member starts at offset 8, which is
     * alignment enough for ``double``.
     */
    unsigned char vals[];
} hybsol_row_entry_t;

/**
 * The stored blocks of a single block row, kept sorted by column.
 *
 * No allocator is carried: everything that allocates or frees on a row's
 * behalf already holds the owning system, and takes the allocator from it via
 * :c:func:`hybsol_current_alloc`.
 */
typedef struct hybsol_row
{
    /** Number of entries in use. */
    uint64_t count;
    /** Number of entry slots allocated in ``entries``. */
    uint64_t capacity;
    /** Entry pointers; slots outside ``[0, count)`` are ``NULL``. */
    hybsol_row_entry_t **entries;
} hybsol_row_t;

struct hybsol_system
{
    /**
     * Allocator every allocation of this system goes through.
     *
     * Chosen at creation and never changed, so reading it is safe from any
     * thread. Always non-NULL for a system that exists.
     */
    const cutl_allocator_t *allocator;
    /** Number of blocks per dimension. */
    uint64_t n;
    /** ``n + 1`` offsets into the underlying matrix; ``block_offsets[0] == 0``. */
    uint64_t *block_offsets;
    /** One row per block row. */
    hybsol_row_t *rows;
    /** Per-row flag: the diagonal block currently holds LU factors. */
    uint8_t *diag_decomposed;
    /**
     * The block whose diagonal could not be factorized, or ``UINT64_MAX``, so a
     * failure can name the offending block.
     */
    uint64_t failing_block;
    /** Recorded operations, or ``NULL`` while the system is undecomposed. */
    hybsol_operation_t *ops;
    /** Number of recorded operations. */
    uint64_t n_ops;
    /** Slots allocated in ``ops``. */
    uint64_t ops_capacity;
    /** Non-zero once :c:func:`hybsol_system_decompose` succeeded. */
    uint8_t decomposed;
    /** Type every stored block is held in. */
    hybsol_precision_t precision;
    /**
     * Fill-in pool, owned by the system; ``NULL`` when no decomposition is
     * running, which is what routes every other phase back to ``allocator``.
     *
     * Armed before the first parallel region and disarmed after the last, but
     * the memory stays: the blocks carved out of it are the system's fill-in
     * and are released with the system rather than individually.
     */
    unsigned char *pool;
    /** Bytes of ``pool``. */
    size_t pool_size;
    /** Next free byte in ``pool``. Only ever touched by an atomic bump. */
    size_t pool_used;
    /** What to hand back to the allocator: ``pool`` is this rounded up. */
    unsigned char *pool_raw;
    /** Allocator handing out of the pool, with ``state`` pointing at the system. */
    cutl_allocator_t pool_alloc;
    /**
     * Non-zero only while a decomposition is running.
     *
     * Separate from ``pool`` staying non-NULL: the pool has to remain
     * recognisable for as long as the system lives, because that is what
     * :c:func:`hybsol_ptr_is_pooled` uses to tell fill-in from ordinary
     * allocations when the system is destroyed.
     */
    uint8_t pool_armed;
};

/* ------------------------------------------------------------------------- */
/* Allocation routing                                                          */
/* ------------------------------------------------------------------------- */

/**
 * Which allocator the calling thread should use for ``sys`` right now.
 *
 * While a decomposition is running that is the fill-in pool, which threads
 * share through one atomic bump; everywhere else it is the system's own
 * allocator.
 */
static inline const cutl_allocator_t *hybsol_current_alloc(const hybsol_system_t *const sys)
{
    return sys->pool_armed ? &sys->pool_alloc : sys->allocator;
}

/**
 * Whether ``ptr`` came out of ``sys``'s pool rather than its allocator.
 *
 * Pooled memory is owned by the system: it is released with the system, never
 * individually, so a caller holding a pooled pointer must skip the free.
 */
int hybsol_ptr_is_pooled(const hybsol_system_t *sys, const void *ptr);

/* ------------------------------------------------------------------------- */
/* Helpers                                                                    */
/* ------------------------------------------------------------------------- */

/** Number of rows/columns of block ``idx``. */
static inline uint64_t hybsol_block_size(const hybsol_system_t *const sys, const uint64_t idx)
{
    return sys->block_offsets[idx + 1] - sys->block_offsets[idx];
}

/** Bytes occupied by one stored element of ``precision``. */
static inline size_t hybsol_scalar_size(const hybsol_precision_t precision)
{
    return precision == HYBSOL_PRECISION_SINGLE ? sizeof(float) : sizeof(double);
}

/**
 * Find the first entry of a row whose column is greater than or equal to ``val``.
 *
 * :returns: The index of that entry, or ``row->count`` when there is none.
 */
uint64_t hybsol_row_find_geq(const hybsol_row_t *row, uint64_t val);

/**
 * Find the entry for an exact column index.
 *
 * :param row: The row to search.
 * :param col: The column to look for.
 * :param out: Receives the entry index when the function returns ``1``.
 * :returns: ``1`` if the column is present, ``0`` otherwise.
 */
int hybsol_row_find(const hybsol_row_t *row, uint64_t col, uint64_t *out);

/**
 * Make room for ``needed`` entries in a row.
 *
 * New slots are filled with ``NULL`` so that partially populated rows stay
 * safe to walk.
 *
 * :returns: :c:enumerator:`HYBSOL_SUCCESS` or
 *     :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY`.
 */
hybsol_result_t hybsol_row_reserve(hybsol_system_t *sys, hybsol_row_t *row, uint64_t needed);

/**
 * Reject mutations once the system has been decomposed.
 *
 * :returns: :c:enumerator:`HYBSOL_SUCCESS` or
 *     :c:enumerator:`HYBSOL_ERROR_ALREADY_DECOMPOSED`.
 */
hybsol_result_t hybsol_require_mutable(const hybsol_system_t *sys);

/**
 * Assert that the caller used the spelling matching how the system stores.
 *
 * Every value-carrying function has an unsuffixed double spelling and an
 * ``_f32`` single-precision twin, which size a block differently — the wrong
 * one would silently narrow the blocks and over-read the buffer.
 *
 * :param sys: The system.
 * :param precision: The spelling the caller chose.
 */
static inline void hybsol_require_precision(const hybsol_system_t *const sys, const hybsol_precision_t precision)
{
    CUTL_ASSERT(sys->precision == precision,
                "This is the %s spelling but the system stores %s; use the matching function.",
                precision == HYBSOL_PRECISION_DOUBLE ? "double" : "single precision",
                sys->precision == HYBSOL_PRECISION_DOUBLE ? "doubles" : "floats");
}

/**
 * Assert that a block index names one of the system's blocks.
 *
 * :param sys: The system.
 * :param idx: The block index to check.
 */
static inline void hybsol_require_index(const hybsol_system_t *const sys, const uint64_t idx)
{
    CUTL_ASSERT(idx < sys->n, "Block index %llu is outside [0, %llu).", (unsigned long long)idx,
                (unsigned long long)sys->n);
}

/** Mark row ``idx`` as needing a fresh LU factorization. */
void hybsol_invalidate_diagonal(hybsol_system_t *sys, uint64_t idx);

/**
 * Eliminate one row with another, using scratch the caller already owns.
 *
 * Called from inside a parallel region, where ``scratch`` is the calling
 * thread's workspace block: at least ``max_block_size^2`` elements in the
 * system's precision.
 */
hybsol_result_t hybsol_system_eliminate_row_scratch(hybsol_system_t *sys, uint64_t row_tgt, uint64_t row_src,
                                                    void *scratch);

/**
 * The most operations a decomposition of a system with ``n`` blocks can record.
 *
 * One ``INVERT_DIAGONAL`` per row plus at most one ``ELIMINATE`` per
 * ``(target, source)`` pair with ``source < target``. Dense systems reach this
 * exactly.
 */
static inline uint64_t hybsol_operation_bound(const uint64_t n)
{
    return n * (n + 1) / 2;
}

/**
 * Walk the elimination graph symbolically and size what it will need.
 *
 * Mirrors the pass order and the merge in
 * :c:func:`hybsol_system_decompose_with_workspace`, carrying only column
 * indices, so no block value is touched. `hybsol_fill_plan_t` is declared
 * in the public header, which this one reaches through the umbrella.
 *
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`,
 *     :c:enumerator:`HYBSOL_ERROR_ALREADY_DECOMPOSED`,
 *     :c:enumerator:`HYBSOL_ERROR_SYSTEM_INVALID`,
 *     :c:enumerator:`HYBSOL_ERROR_EMPTY_ROW` or
 *     :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY`.
 */
hybsol_result_t hybsol_fill_plan_compute(const hybsol_system_t *sys, hybsol_fill_plan_t *out);

/**
 * Size the fill-in and grow every row to its peak, on one thread.
 *
 * :c:func:`hybsol_fill_plan_compute` plus the row pre-grow that lets the
 * elimination's own reserves always find enough capacity, so nothing inside a
 * parallel region ever has to release an array.
 */
hybsol_result_t hybsol_fill_plan_prepare(hybsol_system_t *sys, hybsol_fill_plan_t *out);

/** Allocate ``sys``'s fill-in pool and arm it, sized by ``plan``. */
hybsol_result_t hybsol_pool_open(hybsol_system_t *sys, const hybsol_fill_plan_t *plan);

/** Stop routing allocations through the pool; the memory stays with the system. */
void hybsol_pool_close(hybsol_system_t *sys);

/** Release the pool. Only for a system being destroyed. */
void hybsol_pool_free(hybsol_system_t *sys);

/** Round ``value`` up to the pool's carve granularity. */
static inline size_t hybsol_pool_align(const size_t value)
{
    return (value + (HYBSOL_REGION_ALIGN - 1)) & ~(size_t)(HYBSOL_REGION_ALIGN - 1);
}

/**
 * Turn a thread request into the team size used by the OpenMP pragmas.
 *
 * ``0`` selects the OpenMP default (usually every core); ``1`` runs serially.
 */
static inline int hybsol_resolve_threads(const uint64_t n_threads)
{
    if (n_threads == 0)
        return HYBSOL_DEFAULT_NUM_THREADS;
    if (n_threads > (uint64_t)INT_MAX)
        return INT_MAX;
    return (int)n_threads;
}

#define HYBSOL_MARK_USED(var) ((void)(var))

#endif /* HYBSOL_CORE_INTERNAL_H */
