/**
 * @file core/internal.h
 * Private declarations shared by the hybsol core translation units. Nothing
 * here is public: consumers see only ``include/hybsol/``.
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
 * Preconditions are checked with cutl's CUTL_ASSERT, which reports file, line,
 * function and the failed condition before aborting. Without
 * CUTL_ENABLE_ASSERTS it degrades to CUTL_ASSUME, which may delete the
 * expression outright -- so never put a side effect in an assert condition.
 */
#ifndef CUTL_ENABLE_ASSERTS
#define CUTL_ENABLE_ASSERTS 0
#endif

/* ------------------------------------------------------------------------- */
/* Allocation                                                                 */
/* ------------------------------------------------------------------------- */

/*
 * Every core file allocates through these: the allocator is an argument, not a
 * global, so concurrent use is safe. A zero size yields NULL, as cutl's
 * allocators do, which keeps the `== NULL` out-of-memory checks correct.
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
/* Alignment                                                                  */
/* ------------------------------------------------------------------------- */

/** Granularity every carved block starts at; wide enough for any scalar stored. */
#define HYBSOL_ALIGN 64u

/** Round ``value`` up to :c:enumerator:`HYBSOL_ALIGN`. */
static inline size_t hybsol_align_up(const size_t value)
{
    return (value + (HYBSOL_ALIGN - 1)) & ~(size_t)(HYBSOL_ALIGN - 1);
}

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
     * precision. Raw bytes because the entry does not know which precision it
     * holds; offset 8 is aligned for ``double``.
     */
    unsigned char vals[];
} hybsol_row_entry_t;

/** The stored blocks of a single block row, kept sorted by column. */
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
    /** Fixed at creation, so reading it is safe from any thread. */
    const cutl_allocator_t *allocator;
    uint64_t n;
    /** ``n + 1`` offsets into the underlying matrix; ``block_offsets[0] == 0``. */
    uint64_t *block_offsets;
    hybsol_row_t *rows;
    hybsol_precision_t precision;
    /** Arena of a copied system's entry payloads, or ``NULL`` when each owns its own. */
    unsigned char *values;
    size_t values_bytes;
};

/* ------------------------------------------------------------------------- */
/* The elimination graph                                                      */
/* ------------------------------------------------------------------------- */

/*
 * What the symbolic walk produces: the pattern every block row ends up with,
 * the passes the rows are processed in, and the exact storage a decomposition
 * needs. A function of the system's pattern alone -- no value is ever read.
 */
struct hybsol_elimination
{
    /** Allocator the walk allocated through; every pointer above is interior. */
    const cutl_allocator_t *allocator;
    uint64_t n;
    /** Sum of the final row lengths. */
    uint64_t n_columns;
    /** Exact number of operations a factorization records. */
    uint64_t n_operations;
    /** Number of passes; at least 1. */
    uint64_t n_levels;
    /** Number of (row, pass) pairs the passes hold in total. */
    uint64_t n_occupancy;
    /** Bytes the decomposition's value arena has to carve. */
    size_t value_bytes;
    /** FNV-1a over the system's ``n + 1`` block offsets. */
    uint64_t signature;
    /** Precision of the analyzed system. */
    hybsol_precision_t precision;
    /** The block whose diagonal is identically zero, or ``UINT64_MAX``. */
    uint64_t failing_block;
    /**
     * ``n + 1`` block sizes of the analyzed system, copied from it. A
     * decomposition's layout is a function of the graph alone, at any
     * precision: without the sizes the per-entry payload could not be
     * computed without the system in hand.
     */
    uint64_t *block_offsets;
    /** ``n + 1`` prefix into ``cols``. */
    uint64_t *row_offset;
    /** ``n_columns`` column indices, ascending within each row. */
    uint64_t *cols;
    /** ``n_levels + 1`` prefix into ``level_rows``. */
    uint64_t *level_offset;
    /** ``n_occupancy`` block row indices, ascending within each pass. */
    uint64_t *level_rows;
    /**
     * ``n_occupancy`` step indices paired with ``level_rows``: the row's ``k``-th
     * elimination below ``row_n_elim``, its diagonal factorization at it. A
     * row's passes are not consecutive -- it waits for its source -- so the step
     * travels with the row rather than being derived from the pass.
     */
    uint64_t *level_k;
    /** ``n``; the first pass a row is processed in. */
    uint64_t *row_first;
    /** ``n``; how many eliminations a row performs. */
    uint64_t *row_n_elim;
    /** ``n``; the last pass a row is processed in. */
    uint64_t *row_level;
    /** The allocator's pointer; everything above is interior to it. */
    unsigned char *raw;
    /** Size of ``raw``. */
    size_t raw_bytes;
};

/**
 * Whether ``sys``'s block order admits a factorization at all: the walk of
 * :c:func:`hybsol_elimination_create` with the result discarded.
 * ``failing_block`` may be ``NULL``.
 */
hybsol_result_t hybsol_elimination_check(const hybsol_system_t *sys, uint64_t *failing_block);

/**
 * Whether ``entry``'s payload is the caller's to free -- false for a system
 * copied by :c:func:`hybsol_system_copy`, whose payloads ride in one arena.
 */
static inline int hybsol_entry_is_owned(const hybsol_system_t *const sys, const hybsol_row_entry_t *const entry)
{
    const unsigned char *const p = (const unsigned char *)entry;
    return p < sys->values || p >= sys->values + sys->values_bytes;
}

/** Where row ``row``'s columns start in :c:func:`hybsol_elimination_row_columns`. */
static inline uint64_t hybsol_elimination_row_offset(const hybsol_elimination_t *const graph, const uint64_t row)
{
    return graph->row_offset[row];
}

/** FNV-1a over a system's block offsets, so a graph can recognize its system. */
uint64_t hybsol_elimination_signature(const hybsol_system_t *sys);

/* ------------------------------------------------------------------------- */
/* The decomposition's kernels                                               */
/* ------------------------------------------------------------------------- */

/*
 * The arithmetic, instantiated once per precision in
 * ``decomposition_numeric.c`` and dispatched here, so the factorization and the
 * solve share one implementation of each step.
 */

/** Replace the diagonal block of ``idx`` with its LU factorization. */
hybsol_result_t hybsol_decomposition_diagonal_lu(hybsol_decomposition_t *dec, uint64_t idx);

/** Scale everything above the diagonal of ``idx`` by the inverse of its factors. */
void hybsol_decomposition_diagonal_inverse(hybsol_decomposition_t *dec, uint64_t idx);

/**
 * Subtract the ``k``-th source row of ``row_tgt``'s chain from what is left of
 * that row, using the caller's per-thread scratch.
 */
void hybsol_decomposition_eliminate(hybsol_decomposition_t *dec, uint64_t row_tgt, uint64_t k, void *scratch);

/** ``vec[row_tgt] -= block(row_tgt, source) @ vec[source]`` for the ``k``-th source. */
void hybsol_decomposition_forward_eliminate(const hybsol_decomposition_t *dec, uint64_t row, uint64_t k, double *vec);

/** ``vec[row] = L[row]^{-1} vec[row]``. */
void hybsol_decomposition_forward_solve_diagonal(const hybsol_decomposition_t *dec, uint64_t row, double *vec);

/** Replay a recorded operation list, front to back. */
void hybsol_decomposition_replay(const hybsol_decomposition_t *dec, uint64_t n_ops, const hybsol_operation_t *ops,
                                 double *vec);

/** Solve ``U x = y`` by block back-substitution. */
void hybsol_decomposition_back_substitute(const hybsol_decomposition_t *dec, double *y);

/* ------------------------------------------------------------------------- */
/* The decomposition                                                          */
/* ------------------------------------------------------------------------- */

/**
 * The system's payload for one slot of the graph's pattern, or ``NULL`` when
 * the pattern holds a column the system does not -- fill-in, which starts at
 * zero. Backends staging their own copy of the blocks read through this so
 * they cannot drift from the copy :c:func:`hybsol_decomposition_init` makes.
 */
const void *hybsol_decomposition_entry_source(const hybsol_system_t *sys, const hybsol_elimination_t *graph,
                                              uint64_t row, uint64_t slot);

/**
 * Bytes of a decomposition frame *without* its value arena: the schedule, the
 * pattern and the headers, for a backend that keeps the blocks somewhere else
 * and fills :c:func:`hybsol_decomposition_init_frame` itself.
 */
size_t hybsol_decomposition_frame_bytes(const hybsol_elimination_t *graph, hybsol_precision_t precision);

/**
 * Lay out that arena-less frame in caller-owned storage and copy the schedule
 * into it.
 *
 * Everything :c:func:`hybsol_decomposition_init` does *except* the blocks:
 * ``rows`` and ``entries`` are carved but hold no entry pointers, ``values`` is
 * ``NULL``, and ``backend``/``backend_state`` are ``NULL`` for the backend to
 * fill in. A frame laid out this way is never factored on the CPU.
 */
hybsol_result_t hybsol_decomposition_init_frame(const hybsol_system_t *sys, const hybsol_elimination_t *graph,
                                                hybsol_precision_t precision, void *storage,
                                                hybsol_decomposition_t **out);

/*
 * A copy of every block the elimination will ever touch, in one allocation
 * whose layout the graph fixes: no row grows and no entry moves.
 */
struct hybsol_decomposition
{
    /** Allocator every allocation of the decomposition goes through. */
    const cutl_allocator_t *allocator;
    /** Number of blocks per dimension. */
    uint64_t n;
    /** Number of passes, copied from the graph. */
    uint64_t n_levels;
    /** Number of (row, pass) pairs the passes hold in total. */
    uint64_t n_occupancy;
    /** Exact number of operations a factorization records. */
    uint64_t n_operations;
    /** The block whose diagonal could not be factorized, or ``UINT64_MAX``. */
    uint64_t failing_block;
    /** Type every stored block is held in. */
    hybsol_precision_t precision;
    /** Non-zero once :c:func:`hybsol_decomposition_factorize` succeeded. */
    uint8_t factorized;
    /** The allocator's pointer; everything below is interior to it. */
    unsigned char *raw;
    /** ``n + 1`` offsets, copied from the system. */
    const uint64_t *block_offsets;
    /** ``n_levels + 1`` prefix into ``level_rows``, copied from the graph. */
    uint64_t *level_offset;
    /** ``n_occupancy`` block row indices, ascending within each pass. */
    uint64_t *level_rows;
    /**
     * ``n_occupancy`` step indices paired with ``level_rows``: the row's ``k``-th
     * elimination below ``row_n_elim``, its diagonal factorization at it. A
     * row's passes are not consecutive -- it waits for its source -- so the step
     * travels with the row rather than being derived from the pass.
     */
    uint64_t *level_k;
    /** ``n``; how many eliminations a row performs. */
    uint64_t *row_n_elim;
    /** ``n``; the last pass a row is processed in. */
    uint64_t *row_level;
    /** One row per block row, holding the final pattern. */
    hybsol_row_t *rows;
    /** ``n_columns`` entry pointers, one flat array behind every row. */
    hybsol_row_entry_t **entries;
    /** The block storage itself. */
    unsigned char *values;

    /** The backend computing over this decomposition's factors, or ``NULL``
     * for the CPU ones in ``values``. Set once, by the backend's own create. */
    const hybsol_backend_t *backend;
    /** Whatever the backend attached; its ``destroy`` takes it back. */
    void *backend_state;
};

/** Number of rows/columns of block ``idx`` of a decomposition. */
static inline uint64_t hybsol_decomposition_block_size(const hybsol_decomposition_t *const dec, const uint64_t idx)
{
    return dec->block_offsets[idx + 1] - dec->block_offsets[idx];
}

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

/** The first entry whose column is ``>= val``, or ``row->count`` when there is none. */
uint64_t hybsol_row_find_geq(const hybsol_row_t *row, uint64_t val);

/** The entry for an exact column: ``1`` if present, with its index in ``*out``. */
int hybsol_row_find(const hybsol_row_t *row, uint64_t col, uint64_t *out);

/**
 * Make room for ``needed`` entries, filling new slots with ``NULL``.
 *
 * :returns: :c:enumerator:`HYBSOL_SUCCESS` or
 *     :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY`.
 */
hybsol_result_t hybsol_row_reserve(hybsol_system_t *sys, hybsol_row_t *row, uint64_t needed);

/**
 * Assert that the caller used the spelling matching how the system stores: the
 * ``_f32`` twin sizes a block differently, and the wrong one would silently
 * narrow the blocks and over-read the buffer.
 */
static inline void hybsol_require_precision(const hybsol_system_t *const sys, const hybsol_precision_t precision)
{
    CUTL_ASSERT(sys->precision == precision,
                "This is the %s spelling but the system stores %s; use the matching function.",
                precision == HYBSOL_PRECISION_DOUBLE ? "double" : "single precision",
                sys->precision == HYBSOL_PRECISION_DOUBLE ? "doubles" : "floats");
}

/** Assert that a block index names one of the system's blocks. */
static inline void hybsol_require_index(const hybsol_system_t *const sys, const uint64_t idx)
{
    CUTL_ASSERT(idx < sys->n, "Block index %llu is outside [0, %llu).", (unsigned long long)idx,
                (unsigned long long)sys->n);
}

/**
 * The team size for the OpenMP pragmas; ``0`` is the OpenMP default, ``1`` serial.
 *
 * A request larger than the machine has cores is honoured rather than capped:
 * over-subscribing is the caller's decision, and an absurd one is the caller's
 * to live with.
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
