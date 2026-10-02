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

#include <cutl/allocators.h>

/**
 * Error codes returned by the fallible functions of the hybsol API.
 *
 * These cover outcomes of the data, not caller mistakes: allocation failure, a
 * singular matrix, a system that does not decompose. Conditions the caller can
 * be asked to guarantee — index ranges, non-``NULL`` pointers, matching shapes
 * — are preconditions instead, checked with ``CUTL_ASSERT``; see
 * :c:file:`hybsol/hybsol.h`.
 *
 * :c:func:`hybsol_result_str` turns a code into a short human-readable
 * description.
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
    /** A diagonal block hit an exactly zero pivot and cannot be factorized. */
    HYBSOL_ERROR_SINGULAR,
    /** The system violates the assumptions of the solver (see :c:func:`hybsol_system_is_valid`). */
    HYBSOL_ERROR_SYSTEM_INVALID,
    /** The operation requires a decomposed system, but none has been computed yet. */
    HYBSOL_ERROR_NOT_DECOMPOSED,
    /** The system has already been decomposed and can no longer be modified. */
    HYBSOL_ERROR_ALREADY_DECOMPOSED,
    /** The block order does not admit any factorization of this system. */
    HYBSOL_ERROR_INVALID_ORDERING,
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
 * them use this type. Vectors passed to :c:func:`hybsol_decomposition_solve` and
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
 * ``const uint64_t rows[static n_entries]`` -- the contract these parameters
 * have, worth keeping in the real headers. libclang cannot spell a variably
 * modified array type, though: it returns ``<recovery-expr>()``. The docs
 * build (``-DHYBSOL_DOCS``, set in ``docs/conf.py``) therefore sees a plain
 * pointer, which is what the parameter is to a caller anyway.
 */
#ifdef HYBSOL_DOCS
#define HYBSOL_IN(type, name, n) const type *name
#else
#define HYBSOL_IN(type, name, n) const type name[static n]
#endif

/*
 * Allocations go through a `cutl` allocator, chosen per system rather than
 * globally: `hybsol_system_create` takes the one to use and the system keeps it
 * for its whole lifetime, so the spelling is re-exported rather than duplicated.
 * cutl ships an arena, a fixed-size pool and a validating allocator alongside
 * the default. Pass `&CUTL_STD_ALLOCATOR` for plain malloc/realloc/free.
 */
typedef cutl_allocator_t hybsol_allocator_t;

#endif /* HYBSOL_TYPES_H */
