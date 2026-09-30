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

#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
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

/* ------------------------------------------------------------------------- */
/* Assertions                                                                 */
/* ------------------------------------------------------------------------- */

#ifndef HYBSOL_ENABLE_ASSERTS
#define HYBSOL_ENABLE_ASSERTS 0
#endif

#ifdef __GNUC__
#define HYBSOL_NORETURN __attribute__((noreturn))
#define HYBSOL_PRINTF(fmt_idx, arg_idx) __attribute__((format(printf, fmt_idx, arg_idx)))
#else
#define HYBSOL_NORETURN
#define HYBSOL_PRINTF(fmt_idx, arg_idx)
#endif

#if HYBSOL_ENABLE_ASSERTS
/**
 * Abort the process after printing an internal invariant failure.
 *
 * Only called through :c:macro:`HYBSOL_ASSERT`; never directly.
 */
HYBSOL_NORETURN HYBSOL_PRINTF(5, 6) void hybsol_assert_fail(const char *file, int line, const char *func,
                                                            const char *cond, const char *fmt, ...);

#define HYBSOL_ASSERT(cond, fmt, ...)                                                                                  \
    ((cond) ? (void)0 : hybsol_assert_fail(__FILE__, __LINE__, __func__, #cond, fmt, ##__VA_ARGS__))
#else
#define HYBSOL_ASSERT(cond, fmt, ...) ((void)0)
#endif

/* ------------------------------------------------------------------------- */
/* Allocation                                                                 */
/* ------------------------------------------------------------------------- */

/** The active allocator; replaced by :c:func:`hybsol_set_allocator`. */
extern hybsol_allocator_t hybsol_current_allocator;

static inline void *hybsol_alloc(const size_t size)
{
    return size == 0 ? NULL : hybsol_current_allocator.malloc_fn(size);
}

static inline void *hybsol_grow(void *const ptr, const size_t size)
{
    return size == 0 ? NULL : hybsol_current_allocator.realloc_fn(ptr, size);
}

static inline void hybsol_free(void *const ptr)
{
    hybsol_current_allocator.free_fn(ptr);
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
    /** Number of entries in use. */
    uint64_t count;
    /** Number of entry slots allocated in ``entries``. */
    uint64_t capacity;
    /** Entry pointers; slots outside ``[0, count)`` are ``NULL``. */
    hybsol_row_entry_t **entries;
} hybsol_row_t;

struct hybsol_system
{
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
};

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
hybsol_result_t hybsol_row_reserve(hybsol_row_t *row, uint64_t needed);

/**
 * Reject mutations once the system has been decomposed.
 *
 * :returns: :c:enumerator:`HYBSOL_SUCCESS` or
 *     :c:enumerator:`HYBSOL_ERROR_ALREADY_DECOMPOSED`.
 */
hybsol_result_t hybsol_require_mutable(const hybsol_system_t *sys);

/**
 * Reject a call whose operand type does not match how the system stores blocks.
 *
 * Every value-carrying function has an unsuffixed double spelling and an
 * ``_f32`` single-precision twin. This is what makes handing the wrong one a
 * system a clean error instead of a silent narrowing conversion — and a
 * buffer over-read, since the two spellings size their buffers differently.
 *
 * :param sys: The system.
 * :param precision: The spelling the caller chose.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS` or
 *     :c:enumerator:`HYBSOL_ERROR_INVALID_ARGUMENT`.
 */
static inline hybsol_result_t hybsol_check_precision(const hybsol_system_t *const sys,
                                                     const hybsol_precision_t precision)
{
    return sys->precision == precision ? HYBSOL_SUCCESS : HYBSOL_ERROR_INVALID_ARGUMENT;
}

/** Validate a block row/column index pair. */
hybsol_result_t hybsol_check_index(const hybsol_system_t *sys, uint64_t idx);

/** Mark row ``idx`` as needing a fresh LU factorization. */
void hybsol_invalidate_diagonal(hybsol_system_t *sys, uint64_t idx);

/** Grow the recorded operation list by one, appending ``op``. */
hybsol_result_t hybsol_ops_append(hybsol_system_t *sys, hybsol_operation_t op);

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

#ifdef _OPENMP
#define HYBSOL_THREAD_NUM() omp_get_thread_num()
#else
#define HYBSOL_THREAD_NUM() 0
#endif

#define HYBSOL_MARK_USED(var) ((void)(var))

#endif /* HYBSOL_CORE_INTERNAL_H */
