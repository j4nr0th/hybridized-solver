/**
 * @file hybsol/matrix.h
 * Dense, row-major matrix views and the small set of operations the solver
 * needs on them.
 *
 * A :c:struct:`hybsol_matrix_t` never owns its storage: it is a view onto
 * caller-provided memory, or onto a block inside a :c:struct:`hybsol_system_t`.
 * Keeping ``data`` aligned with ``rows`` is the caller's responsibility.
 *
 * Everything comes in a single-precision twin for
 * :c:enumerator:`HYBSOL_PRECISION_SINGLE` systems. The two spellings are
 * written once and instantiated together, so their semantics are the same
 * apart from the type they store.
 */

#ifndef HYBSOL_MATRIX_H
#define HYBSOL_MATRIX_H

#include <hybsol/types.h>

/** A dense row-major matrix of doubles. */
typedef struct hybsol_matrix
{
    uint64_t rows;
    uint64_t cols;
    /** Row-major storage of ``rows * cols`` doubles. */
    double *data;
} hybsol_matrix_t;

/** Create a matrix view over ``rows * cols`` doubles of existing memory. */
hybsol_matrix_t hybsol_matrix_view(uint64_t rows, uint64_t cols, double *data);

/**
 * Compute ``out := a @ b``.
 *
 * Preconditions: ``a->cols == b->rows``; ``out`` is ``a->rows x b->cols`` and
 * must not overlap ``a`` or ``b``. ``out`` is zeroed before the product is
 * accumulated into it, so even a view that starts out identical to an operand
 * reads zeros rather than its own values.
 */
void hybsol_matrix_multiply(const hybsol_matrix_t *a, const hybsol_matrix_t *b, const hybsol_matrix_t *out);

/**
 * Compute ``out -= a @ b``.
 *
 * Preconditions: ``a->cols == b->rows``; ``out`` is ``a->rows x b->cols``.
 */
void hybsol_matrix_multiply_sub_inplace(const hybsol_matrix_t *a, const hybsol_matrix_t *b, const hybsol_matrix_t *out);

/** Subtract ``b`` from ``a`` in place. ``a`` and ``b`` must have the same shape. */
void hybsol_matrix_subtract_inplace(const hybsol_matrix_t *a, const hybsol_matrix_t *b);

/**
 * Overwrite ``m`` with its LU factorization (Doolittle, unit-lower ``L``).
 * Asserted square; returns :c:enumerator:`HYBSOL_ERROR_SINGULAR` on a zero pivot.
 *
 * No pivoting is performed, so the caller must make the matrix safe to factor
 * without it. The requirement is on the *leading principal minors*, not on the
 * matrix as a whole: every one of the determinants of ``m[0:k, 0:k]`` for
 * ``k = 1 .. rows`` must be nonzero. A nonsingular matrix can fail this, and
 * then a step divides by a pivot that is numerically zero.
 *
 * .. warning::
 *    A *numerically* vanishing leading minor is not detected and produces an
 *    inaccurate factorization rather than an error. This is inherent to
 *    factorizing without pivoting; check the condition yourself if the matrix
 *    is not well behaved by construction.
 */
hybsol_result_t hybsol_matrix_lu_decompose(const hybsol_matrix_t *m);

/**
 * Solve ``m @ x = b`` using a factorization from
 * :c:func:`hybsol_matrix_lu_decompose`.
 *
 * Preconditions: ``m`` asserted square, ``m->rows == b->rows``, ``out`` the
 * shape of ``b``. ``out`` may be ``b`` itself, which solves in place.
 */
hybsol_result_t hybsol_matrix_lu_solve(const hybsol_matrix_t *m, const hybsol_matrix_t *b, const hybsol_matrix_t *out);

/* ----------------------------------------------------------------------- */
/* Single-precision twins                                                  */
/* ----------------------------------------------------------------------- */

/**
 * A dense row-major matrix of floats: the same shape of type as
 * :c:struct:`hybsol_matrix_t`, spelling the storage a
 * :c:enumerator:`HYBSOL_PRECISION_SINGLE` system uses. Mixing the two is not
 * resolved for you: pass the one that matches the system.
 */
typedef struct hybsol_fmatrix
{
    uint64_t rows;
    uint64_t cols;
    /** Row-major storage of ``rows * cols`` floats. */
    float *data;
} hybsol_fmatrix_t;

/** Create a single-precision matrix view over existing memory. */
hybsol_fmatrix_t hybsol_fmatrix_view(uint64_t rows, uint64_t cols, float *data);

/**
 * Compute ``out := a @ b`` in single precision.
 *
 * Preconditions: ``a->cols == b->rows``; ``out`` is ``a->rows x b->cols`` and
 * must not overlap ``a`` or ``b``, for the same reason as the double spelling.
 */
void hybsol_fmatrix_multiply(const hybsol_fmatrix_t *a, const hybsol_fmatrix_t *b, const hybsol_fmatrix_t *out);

/**
 * Compute ``out -= a @ b`` in single precision.
 *
 * Preconditions: ``a->cols == b->rows``; ``out`` is ``a->rows x b->cols``.
 */
void hybsol_fmatrix_multiply_sub_inplace(const hybsol_fmatrix_t *a, const hybsol_fmatrix_t *b,
                                         const hybsol_fmatrix_t *out);

/** Subtract ``b`` from ``a`` in place. ``a`` and ``b`` must have the same shape. */
void hybsol_fmatrix_subtract_inplace(const hybsol_fmatrix_t *a, const hybsol_fmatrix_t *b);

/**
 * Overwrite ``m`` with its LU factorization (Doolittle, unit-lower ``L``),
 * with ``float`` pivots -- so a matrix that survives the double spelling can
 * still come out singular here.
 *
 * See :c:func:`hybsol_matrix_lu_decompose` for what is and is not detected.
 */
hybsol_result_t hybsol_fmatrix_lu_decompose(const hybsol_fmatrix_t *m);

/**
 * Solve ``m @ x = b`` using a factorization from
 * :c:func:`hybsol_fmatrix_lu_decompose`.
 *
 * Preconditions: ``m`` asserted square, ``m->rows == b->rows``, ``out`` the
 * shape of ``b``. ``out`` may be ``b`` itself, which solves in place.
 */
hybsol_result_t hybsol_fmatrix_lu_solve(const hybsol_fmatrix_t *m, const hybsol_fmatrix_t *b,
                                        const hybsol_fmatrix_t *out);

#endif /* HYBSOL_MATRIX_H */
