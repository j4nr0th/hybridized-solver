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
 * objects use different allocators and keeps concurrent use safe.
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
 * behalf already holds the owning system, and takes the allocator from it.
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
    /**
     * The block whose diagonal could not be factorized, or ``UINT64_MAX``, so a
     * failure can name the offending block.
     */
    uint64_t failing_block;
    /** Type every stored block is held in. */
    hybsol_precision_t precision;
};

/* ------------------------------------------------------------------------- */
/* The elimination graph                                                      */
/* ------------------------------------------------------------------------- */

/*
 * What the symbolic walk produces: the pattern every block row ends up with,
 * the passes the rows are processed in, and the exact size of the storage a
 * decomposition needs. Nothing here depends on a block value, so the whole
 * structure is a function of the system's pattern alone.
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
    /** ``n + 1`` prefix into ``cols``. */
    uint64_t *row_offset;
    /** ``n_columns`` column indices, ascending within each row. */
    uint64_t *cols;
    /** ``n_levels + 1`` prefix into ``level_rows``. */
    uint64_t *level_offset;
    /** ``n_occupancy`` block row indices, ascending within each pass. */
    uint64_t *level_rows;
    /**
     * ``n_occupancy`` step indices, paired with ``level_rows``.
     *
     * The step is the row's ``k``-th elimination when it is below
     * ``row_n_elim``, and the diagonal factorization when it equals it. A row's
     * passes are not consecutive -- it waits for its source -- so the step has
     * to be carried alongside the row rather than derived from the pass.
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
 * Whether ``sys``'s current block order admits a factorization at all.
 *
 * Runs the same walk as :c:func:`hybsol_elimination_create` and throws the
 * result away, so a caller reordering blocks can reject a permutation before
 * anything is factorized. Sets ``sys->failing_block`` when it rejects one.
 */
hybsol_result_t hybsol_elimination_check(hybsol_system_t *sys);

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
 * ``decomposition_numeric.c`` and dispatched here, so the factorization and
 * the solve share one implementation of each step.
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

/*
 * A copy of every block the elimination will ever touch, in one allocation
 * whose layout the graph fixes. The pattern never changes, so no row grows and
 * no entry moves: the factorization writes into slots that already exist.
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
    /** Size of ``raw``. */
    size_t raw_bytes;
    /** ``n + 1`` offsets, copied from the system. */
    const uint64_t *block_offsets;
    /** ``n_levels + 1`` prefix into ``level_rows``, copied from the graph. */
    uint64_t *level_offset;
    /** ``n_occupancy`` block row indices, ascending within each pass. */
    uint64_t *level_rows;
    /**
     * ``n_occupancy`` step indices, paired with ``level_rows``.
     *
     * The step is the row's ``k``-th elimination when it is below
     * ``row_n_elim``, and the diagonal factorization when it equals it. A row's
     * passes are not consecutive -- it waits for its source -- so the step has
     * to be carried alongside the row rather than derived from the pass.
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
