/**
 * @file hybsol/matrix.h
 * Dense, row-major matrix views and the small set of operations the solver
 * needs on them.
 *
 * A :c:struct:`hybsol_matrix_t` never owns its storage: it is a view onto
 * caller-provided memory (or onto a block stored inside a
 * :c:struct:`hybsol_system_t`). Keeping ``data`` aligned with ``rows`` is the
 * caller's responsibility.
 *
 * Everything below comes in a single-precision twin for systems created with
 * :c:enumerator:`HYBSOL_PRECISION_SINGLE`: :c:struct:`hybsol_matrix_t` and the
 * unsuffixed names carry doubles, :c:struct:`hybsol_fmatrix_t` and the
 * ``_f32``-suffixed names carry floats. The two spellings are written once and
 * instantiated together, so their semantics are the same apart from the type
 * they store.
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
 * :param m: Square matrix to factor in place. Asserted square.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS` or
 *     :c:enumerator:`HYBSOL_ERROR_SINGULAR` if a zero pivot was encountered.
 */
hybsol_result_t hybsol_matrix_lu_decompose(const hybsol_matrix_t *m);

/**
 * Solve ``m @ x = b`` using a factorization produced by
 * :c:func:`hybsol_matrix_lu_decompose`.
 *
 * :param m: The LU factors; asserted square.
 * :param b: Right-hand side with ``m->rows`` rows.
 * :param out: Destination for ``x`` with the same shape as ``b``. May alias ``b``.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`.
 */
hybsol_result_t hybsol_matrix_lu_solve(const hybsol_matrix_t *m, const hybsol_matrix_t *b, const hybsol_matrix_t *out);

/* ----------------------------------------------------------------------- */
/* Single-precision twins                                                  */
/* ----------------------------------------------------------------------- */

/**
 * A dense row-major matrix of floats.
 *
 * The same shape of type as :c:struct:`hybsol_matrix_t`, spelling the
 * storage a :c:enumerator:`HYBSOL_PRECISION_SINGLE` system uses. Mixing the
 * two is not resolved for you: pass the one that matches the system.
 */
typedef struct hybsol_fmatrix
{
    /** Number of rows. */
    uint64_t rows;
    /** Number of columns. */
    uint64_t cols;
    /** Row-major storage of ``rows * cols`` floats. */
    float *data;
} hybsol_fmatrix_t;

/**
 * Create a single-precision matrix view over existing memory.
 *
 * :param rows: Number of rows.
 * :param cols: Number of columns.
 * :param data: Storage of at least ``rows * cols`` floats.
 * :returns: A view referring to ``data``.
 */
hybsol_fmatrix_t hybsol_fmatrix_view(uint64_t rows, uint64_t cols, float *data);

/**
 * Compute ``out := a @ b`` in single precision.
 *
 * :param a: Left operand, ``a->cols`` must equal ``b->rows``.
 * :param b: Right operand.
 * :param out: Destination; must be ``a->rows`` by ``b->cols``. Its storage
 *     must not overlap ``a`` or ``b`` unless it is identical to one of them.
 */
void hybsol_fmatrix_multiply(const hybsol_fmatrix_t *a, const hybsol_fmatrix_t *b, const hybsol_fmatrix_t *out);

/**
 * Compute ``out -= a @ b`` in single precision.
 *
 * :param a: Left operand, ``a->cols`` must equal ``b->rows``.
 * :param b: Right operand.
 * :param out: Accumulator; must be ``a->rows`` by ``b->cols``.
 */
void hybsol_fmatrix_multiply_sub_inplace(const hybsol_fmatrix_t *a, const hybsol_fmatrix_t *b,
                                         const hybsol_fmatrix_t *out);

/**
 * Subtract ``b`` from ``a`` in place.
 *
 * :param a: Minuend, modified in place.
 * :param b: Subtrahend; must have the same shape as ``a``.
 */
void hybsol_fmatrix_subtract_inplace(const hybsol_fmatrix_t *a, const hybsol_fmatrix_t *b);

/**
 * Overwrite ``m`` with its LU factorization (Doolittle, unit-lower ``L``).
 *
 * The same unpivoted, unchecked factorization as
 * :c:func:`hybsol_matrix_lu_decompose` — with ``float`` pivots, so a matrix
 * that survives the double spelling can still come out singular here.
 *
 * :param m: Square matrix to factor in place. Asserted square.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS` or
 *     :c:enumerator:`HYBSOL_ERROR_SINGULAR` if a zero pivot was encountered.
 */
hybsol_result_t hybsol_fmatrix_lu_decompose(const hybsol_fmatrix_t *m);

/**
 * Solve ``m @ x = b`` using a factorization produced by
 * :c:func:`hybsol_fmatrix_lu_decompose`.
 *
 * :param m: The LU factors; asserted square.
 * :param b: Right-hand side with ``m->rows`` rows.
 * :param out: Destination for ``x`` with the same shape as ``b``. May alias ``b``.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`.
 */
hybsol_result_t hybsol_fmatrix_lu_solve(const hybsol_fmatrix_t *m, const hybsol_fmatrix_t *b,
                                        const hybsol_fmatrix_t *out);

#endif /* HYBSOL_MATRIX_H */
