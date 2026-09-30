/**
 * @file core/internal.h
 * Private declarations shared by the hybsol core translation units.
 *
 * Nothing here is part of the public API: consumers only ever see the
 * headers under ``include/hybsol/``. In particular this file may pull in
 * whatever the implementation needs; it never pulls in Python.
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
 * The three allocation helpers every core file goes through. They take the
 * allocator explicitly rather than reading a global, which is what lets two
 * systems use different allocators and keeps concurrent use safe.
 *
 * A zero size yields NULL, matching what cutl's allocators already do, so the
 * `== NULL` checks callers make for out-of-memory stay correct.
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
/* Per-thread bump regions                                                     */
/* ------------------------------------------------------------------------- */

/*
 * A decomposition allocates from inside OpenMP parallel regions, so a caller
 * supplying an allocator that is not thread-safe would be raced. Each thread
 * therefore bumps against a region of its own instead: the fast path touches
 * only that thread's cursor, needs no lock and no atomic, and never calls the
 * system's allocator at all. Only growing a region does, which is why the
 * regions are what they are rather than a fixed arena.
 *
 * A region is presented as a `cutl_allocator_t`, so every existing
 * `hybsol_alloc(alloc, size)` call site keeps its shape and only the allocator
 * pointer changes between phases.
 */
typedef struct hybsol_region
{
    /** Start of the block owned by the system. */
    unsigned char *base;
    /** Bytes in the block. */
    size_t size;
    /** Bytes already handed out. Only this region's own thread writes it. */
    size_t used;
} hybsol_region_t;

/** One thread's slot: which region it is bumping against, and its allocator. */
typedef struct hybsol_thread_alloc
{
    /** The system being decomposed, for growing the region. */
    struct hybsol_system *sys;
    /** This thread's index, fixed for the run. */
    uint64_t thread;
    /** Index into ``sys->regions`` of the region currently being bumped. */
    uint64_t region;
    cutl_allocator_t alloc;
} hybsol_thread_alloc_t;

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

/** Granularity a bump region carves on; wide enough for any scalar stored. */
#define HYBSOL_REGION_ALIGN 64u

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
     * Kept as raw bytes rather than a flexible array of a specific type: the
     * entry does not know which precision it holds, so readers cast it using
     * :c:func:`hybsol_scalar_size` on the system they already hold. The member
     * starts at offset 8, which is alignment enough for ``double``.
     */
    unsigned char vals[];
} hybsol_row_entry_t;

/** The stored blocks of a single block row, kept sorted by column. */
typedef struct hybsol_row
{
    /**
     * Allocator this row's entries were allocated from.
     *
     * Stored per row rather than read from the system so the row helpers,
     * which only ever receive a `hybsol_row_t *`, can allocate. It is the
     * owning system's allocator, copied at creation and never changed.
     */
    const cutl_allocator_t *allocator;
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
     * One slot per thread, live only while a decomposition runs.
     *
     * ``NULL`` outside :c:func:`hybsol_system_decompose`, which is what routes
     * every other phase back to ``allocator``.
     */
    hybsol_thread_alloc_t *thread_allocs;
    /** Entries in ``thread_allocs``. */
    uint64_t n_thread_allocs;
    /**
     * Every region handed out this run, current and superseded alike.
     *
     * A region a thread outgrew stays alive because entries carved out of it
     * are still in use, so they cannot be tracked through ``thread_allocs``
     * alone. Both the range check in :c:func:`hybsol_ptr_is_pooled` and the
     * release walk this list.
     */
    hybsol_region_t *regions;
    /** Regions in use. */
    uint64_t n_regions;
    /** Slots allocated in ``regions``. */
    uint64_t regions_capacity;
};

/* ------------------------------------------------------------------------- */
/* Allocation routing                                                          */
/* ------------------------------------------------------------------------- */

/**
 * Which allocator the calling thread should use for ``sys`` right now.
 *
 * Inside :c:func:`hybsol_system_decompose` that is the calling thread's bump
 * region; everywhere else it is the system's own allocator. A thread OpenMP
 * handed that has no region -- a larger team than was sized for -- falls back
 * to the system allocator rather than writing out of bounds.
 */
static inline const cutl_allocator_t *hybsol_current_alloc(const hybsol_system_t *const sys)
{
    const uint64_t t = (uint64_t)HYBSOL_THREAD_NUM();
    if (sys->thread_allocs != NULL && t < sys->n_thread_allocs)
        return &sys->thread_allocs[t].alloc;
    return sys->allocator;
}

/** Whether ``ptr`` came out of one of ``sys``'s regions rather than the allocator. */
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
 * Give every thread a bump region, replacing any left from a previous run.
 * Regions are large enough that growth is rare; see
 * :c:func:`hybsol_system_decompose_with_workspace` for the teardown side.
 *
 * :returns: :c:enumerator:`HYBSOL_SUCCESS` or
 *     :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY`.
 */
hybsol_result_t hybsol_thread_allocs_open(hybsol_system_t *sys, uint64_t n_threads);

/**
 * Stop routing through the regions, keeping the memory they hold.
 *
 * Entries carved out of a region are the system's fill-in once the
 * decomposition is done, so the regions outlive the run that filled them; only
 * the per-thread slots go away, which puts later phases back on the system's
 * allocator.
 */
void hybsol_thread_allocs_done(hybsol_system_t *sys);

/** Release every region. Only for a system being destroyed. */
void hybsol_regions_free(hybsol_system_t *sys);

/**
 * Assert that the caller used the spelling matching how the system stores.
 *
 * Every value-carrying function has an unsuffixed double spelling and an
 * ``_f32`` single-precision twin. This is what makes handing the wrong one a
 * system fail loudly rather than silently narrow the blocks — and over-read
 * the buffer, since the two spellings size it differently.
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
 * The decomposition calls this from inside a parallel region, where it passes
 * the calling thread's block out of the workspace. ``scratch`` holds at least
 * ``max_block_size^2`` elements in the system's precision.
 */
hybsol_result_t hybsol_system_eliminate_row_scratch(hybsol_system_t *sys, uint64_t row_tgt, uint64_t row_src,
                                                    void *scratch);

/**
 * Append ``op`` to the recorded operation list.
 *
 * The list is sized to its proven upper bound before the parallel region that
 * fills it, so this never grows it and never fails: it is an atomic bump.
 * Operations recorded within a single elimination pass commute -- targets are
 * distinct and every source was finished before the pass began -- so the order
 * threads happen to interleave them in does not matter.
 */
void hybsol_ops_append(hybsol_system_t *sys, hybsol_operation_t op);

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
 * Give every thread a bump region, replacing any from a previous run.
 *
 * Regions are large enough that growth is rare; see
 * :c:func:`hybsol_system_decompose` for the teardown side.
 *
 * :returns: :c:enumerator:`HYBSOL_SUCCESS` or
 *     :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY`.
 */
hybsol_result_t hybsol_thread_allocs_open(hybsol_system_t *sys, uint64_t n_threads);

/** Release the bump regions, leaving ``sys`` usable for another decomposition. */
void hybsol_thread_allocs_close(hybsol_system_t *sys);

/**
 * Turn a thread request into the team size used by the OpenMP pragmas.
 *
 * A request of ``0`` selects the OpenMP default (usually every core);
 * ``1`` runs serially.
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
