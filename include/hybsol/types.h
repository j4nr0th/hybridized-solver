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
 * The outcomes of the fallible functions of the hybsol API.
 *
 * Every code here is something the caller cannot know up front: an allocation
 * that failed, a diagonal with no pivot, a block order that admits no
 * factorization, a coloring that ran out of colors, an iterator that ran off
 * the end of a row. Conditions the caller *can* be asked to guarantee — index
 * ranges, non-``NULL`` pointers, matching shapes, an unfactorized
 * decomposition — are preconditions instead, checked with ``CUTL_ASSERT``; see
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
    /** The row stores no blocks. */
    HYBSOL_ERROR_EMPTY_ROW,
    /** The row stores no block beyond the given column. */
    HYBSOL_ERROR_NO_MORE_COLUMNS,
    /** The requested coloring needed more colors than ``max_colors`` allowed. */
    HYBSOL_ERROR_MAX_COLORS,
    /** A diagonal block hit an exactly zero pivot and cannot be factorized. */
    HYBSOL_ERROR_SINGULAR,
    /** The block order does not admit any factorization of this system. */
    HYBSOL_ERROR_INVALID_ORDERING,
} hybsol_result_t;

/**
 * Describe a result code.
 *
 * Returns a NUL-terminated string literal, never to be freed;
 * ``"unknown result code"`` for values outside the enumeration.
 */
const char *hybsol_result_str(hybsol_result_t result);

/**
 * The floating-point type a system stores its blocks in.
 *
 * Precision is a property of a whole system, chosen when it is created, and
 * cannot be changed afterwards. Vectors passed to
 * :c:func:`hybsol_decomposition_solve` and the ordering helpers are always
 * doubles regardless of it.
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
