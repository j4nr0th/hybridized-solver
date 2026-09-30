/**
 * @file hybsol/types.h
 * Shared types, error handling and allocation for the hybsol C API.
 *
 * This header (and everything else under ``hybsol/``) is independent of
 * Python, NumPy and any other external dependency.
 */

#ifndef HYBSOL_TYPES_H
#define HYBSOL_TYPES_H

#include <stddef.h>
#include <stdint.h>

/**
 * Error codes returned by the fallible functions of the hybsol API.
 *
 * These cover the conditions that are outcomes of the data, not caller
 * mistakes: allocation failure, a singular matrix, a system that does not
 * decompose, and the solver's other runtime states. Conditions the caller
 * can be asked to guarantee — index ranges, non-``NULL`` pointers, matching
 * shapes — are preconditions instead, checked with ``HYBSOL_ASSERT``; see
 * :c:file:`hybsol/hybsol.h`.
 *
 * The enumerator names are spelled out in full so that they can be used
 * safely from other translation units; :c:func:`hybsol_result_str` turns a
 * code into a short human-readable description.
 */
typedef enum hybsol_result
{
    /** The operation completed successfully. */
    HYBSOL_SUCCESS = 0,
    /** A memory allocation failed. */
    HYBSOL_ERROR_OUT_OF_MEMORY,
    /** The row contains no blocks at all. */
    HYBSOL_ERROR_EMPTY_ROW,
    /** The row contains no block with a column index greater than the given one. */
    HYBSOL_ERROR_NO_MORE_COLUMNS,
    /** The requested coloring needed more colors than ``max_colors`` allowed. */
    HYBSOL_ERROR_MAX_COLORS,
    /** A zero pivot was hit by the (unpivoted) LU decomposition. */
    HYBSOL_ERROR_SINGULAR,
    /** The system violates the assumptions of the solver (see :c:func:`hybsol_system_is_valid`). */
    HYBSOL_ERROR_SYSTEM_INVALID,
    /** The operation requires a decomposed system, but none has been computed yet. */
    HYBSOL_ERROR_NOT_DECOMPOSED,
    /** The system has already been decomposed and can no longer be modified. */
    HYBSOL_ERROR_ALREADY_DECOMPOSED,
    /** An invariant inside the library was violated; this is a bug in hybsol. */
    HYBSOL_ERROR_INTERNAL,
} hybsol_result_t;

/**
 * Get a short, stable, human-readable description of a result code.
 *
 * The returned pointer is to a string literal and never needs to be freed.
 *
 * :param result: The code to describe.
 * :returns: A NUL-terminated description; ``"unknown result code"`` for
 *     values that are not part of the enumeration.
 */
const char *hybsol_result_str(hybsol_result_t result);

/**
 * The floating-point type a system stores its blocks in.
 *
 * Precision is a property of a whole system, chosen when it is created: the
 * blocks, the factors produced from them and every operation that touches
 * them use this type. Vectors passed to :c:func:`hybsol_system_solve` and the
 * ordering helpers are always doubles regardless of it.
 *
 * The two flavours of every value-carrying function are spelled out by
 * suffix: the unsuffixed spelling is the double one and takes
 * :c:type:`hybsol_matrix_t`, while ``_f32`` is the single-precision one and
 * takes :c:type:`hybsol_fmatrix_t`. Mixing them up asserts rather than
 * converting silently.
 */
typedef enum hybsol_precision
{
    /** IEEE-754 binary64. The default, and what :c:func:`hybsol_system_create` builds. */
    HYBSOL_PRECISION_DOUBLE,
    /** IEEE-754 binary32: half the memory, and a solve only good to about ``cond * 1e-7``. */
    HYBSOL_PRECISION_SINGLE,
} hybsol_precision_t;

/**
 * The documented spelling of a length-qualified array parameter.
 *
 * ``HYBSOL_IN(uint64_t, rows, n_entries)`` expands to
 * ``const uint64_t rows[static n_entries]``, which promises the compiler a
 * non-null pointer to at least ``n_entries`` elements -- exactly the contract
 * these parameters have, and worth keeping in the real headers.
 *
 * libclang cannot spell a variably modified array type, though: it comes back
 * as ``<recovery-expr>()`` and the rendered signature is unsalvageable. So the
 * documentation build (``-DHYBSOL_DOCS``, set in ``docs/conf.py``) sees a plain
 * pointer instead, and the docs say ``const uint64_t *rows`` -- which is what
 * the parameter is to a caller anyway.
 */
#ifdef HYBSOL_DOCS
#define HYBSOL_IN(type, name, n) const type *name
#else
#define HYBSOL_IN(type, name, n) const type name[static n]
#endif

/**
 * Allocator used by the library.
 *
 * All three members must be set. The default allocator is plain
 * ``malloc``/``realloc``/``free``.
 */
typedef struct hybsol_allocator
{
    /** Allocate ``size`` bytes, or return ``NULL``. Must not return ``NULL`` for ``size == 0``. */
    void *(*malloc_fn)(size_t size);
    /** Grow or shrink a previous allocation; may return ``NULL`` and leave ``ptr`` intact. */
    void *(*realloc_fn)(void *ptr, size_t size);
    /** Release memory obtained from ``malloc_fn``/``realloc_fn``. ``NULL`` must be a no-op. */
    void (*free_fn)(void *ptr);
} hybsol_allocator_t;

/**
 * Install a custom allocator for all hybsol allocations.
 *
 * This must be called before any system is created, and the allocator must
 * remain valid for as long as any hybsol object is alive. It is not
 * thread-safe with respect to concurrent allocation.
 *
 * :param allocator: The allocator to install, or ``NULL`` to restore the
 *     default ``malloc``-based allocator.
 */
void hybsol_set_allocator(const hybsol_allocator_t *allocator);

/**
 * Get the allocator currently in use.
 *
 * :returns: A pointer to the active allocator; never ``NULL``.
 */
const hybsol_allocator_t *hybsol_get_allocator(void);

#endif /* HYBSOL_TYPES_H */
