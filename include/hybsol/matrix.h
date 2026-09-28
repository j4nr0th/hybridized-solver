/**
 * @file hybsol/matrix.h
 * Dense, row-major matrix views and the small set of operations the solver
 * needs on them.
 *
 * A :c:struct:`hybsol_matrix_t` never owns its storage: it is a view onto
 * caller-provided memory (or onto a block stored inside a
 * :c:struct:`hybsol_system_t`). Keeping ``data`` aligned with ``rows`` is the
 * caller's responsibility.
 */

#ifndef HYBSOL_MATRIX_H
#define HYBSOL_MATRIX_H

#include <hybsol/types.h>

/**
 * A dense row-major matrix of doubles.
 */
typedef struct hybsol_matrix
{
    /** Number of rows. */
    uint64_t rows;
    /** Number of columns. */
    uint64_t cols;
    /** Row-major storage of ``rows * cols`` doubles. */
    double *data;
} hybsol_matrix_t;

/**
 * Create a matrix view over existing memory.
 *
 * :param rows: Number of rows.
 * :param cols: Number of columns.
 * :param data: Storage of at least ``rows * cols`` doubles.
 * :returns: A view referring to ``data``.
 */
hybsol_matrix_t hybsol_matrix_view(uint64_t rows, uint64_t cols, double *data);

/**
 * Compute ``out := a @ b``.
 *
 * :param a: Left operand, ``a->cols`` must equal ``b->rows``.
 * :param b: Right operand.
 * :param out: Destination; must be ``a->rows`` by ``b->cols``. Its storage
 *     must not overlap ``a`` or ``b`` unless it is identical to one of them.
 */
void hybsol_matrix_multiply(const hybsol_matrix_t *a, const hybsol_matrix_t *b, const hybsol_matrix_t *out);

/**
 * Compute ``out -= a @ b``.
 *
 * :param a: Left operand, ``a->cols`` must equal ``b->rows``.
 * :param b: Right operand.
 * :param out: Accumulator; must be ``a->rows`` by ``b->cols``.
 */
void hybsol_matrix_multiply_sub_inplace(const hybsol_matrix_t *a, const hybsol_matrix_t *b, const hybsol_matrix_t *out);

/**
 * Subtract ``b`` from ``a`` in place.
 *
 * :param a: Minuend, modified in place.
 * :param b: Subtrahend; must have the same shape as ``a``.
 */
void hybsol_matrix_subtract_inplace(const hybsol_matrix_t *a, const hybsol_matrix_t *b);

/**
 * Overwrite ``m`` with its LU factorization (Doolittle, unit-lower ``L``).
 *
 * .. warning::
 *    No pivoting is performed. A zero pivot aborts the factorization with
 *    :c:enumerator:`HYBSOL_ERROR_SINGULAR`; matrices that are merely
 *    ill-conditioned are *not* detected and will produce inaccurate results.
 *
 * :param m: Square matrix to factor in place.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`, :c:enumerator:`HYBSOL_ERROR_SINGULAR`
 *     if a zero pivot was encountered, or :c:enumerator:`HYBSOL_ERROR_INVALID_ARGUMENT`
 *     if the matrix is not square.
 */
hybsol_result_t hybsol_matrix_lu_decompose(const hybsol_matrix_t *m);

/**
 * Solve ``m @ x = b`` using a factorization produced by
 * :c:func:`hybsol_matrix_lu_decompose`.
 *
 * :param m: The LU factors; square.
 * :param b: Right-hand side with ``m->rows`` rows.
 * :param out: Destination for ``x`` with the same shape as ``b``. May alias ``b``.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS` or :c:enumerator:`HYBSOL_ERROR_INVALID_ARGUMENT`
 *     for incompatible shapes.
 */
hybsol_result_t hybsol_matrix_lu_solve(const hybsol_matrix_t *m, const hybsol_matrix_t *b, const hybsol_matrix_t *out);

#endif /* HYBSOL_MATRIX_H */
