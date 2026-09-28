/**
 * @file core/matrix.c
 * Dense matrix kernels: products, differences and unpivoted LU.
 */

#include "internal.h"

hybsol_matrix_t hybsol_matrix_view(const uint64_t rows, const uint64_t cols, double *const data)
{
    return (hybsol_matrix_t){.rows = rows, .cols = cols, .data = data};
}

void hybsol_matrix_multiply(const hybsol_matrix_t *const a, const hybsol_matrix_t *const b,
                            const hybsol_matrix_t *const out)
{
    HYBSOL_ASSERT(a->cols == b->rows,
                  "Matrix multiplication: number of columns of A (%llu) does not match number of rows of B (%llu).",
                  (unsigned long long)a->cols, (unsigned long long)b->rows);
    HYBSOL_ASSERT(out->rows == a->rows && out->cols == b->cols, "Output matrix has incorrect dimensions.");

    const double *const restrict ptr_a = a->data;
    const double *const restrict ptr_b = b->data;
    double *const restrict ptr_out = out->data;
    for (uint64_t i = 0; i < a->rows; ++i)
    {
        for (uint64_t j = 0; j < b->cols; ++j)
        {
            double v = 0;
            for (uint64_t k = 0; k < a->cols; ++k)
                v += ptr_a[i * a->cols + k] * ptr_b[k * b->cols + j];

            ptr_out[i * out->cols + j] = v;
        }
    }
}

void hybsol_matrix_multiply_sub_inplace(const hybsol_matrix_t *const a, const hybsol_matrix_t *const b,
                                        const hybsol_matrix_t *const out)
{
    HYBSOL_ASSERT(a->cols == b->rows, "Columns of A must match the rows of B (%llu vs %llu).",
                  (unsigned long long)a->cols, (unsigned long long)b->rows);
    HYBSOL_ASSERT(out->rows == a->rows && out->cols == b->cols, "Output matrix has incorrect dimensions.");

    // Compute A @ B and subtract the result from whatever is in OUT
    const double *const restrict ptr_a = a->data;
    const double *const restrict ptr_b = b->data;
    double *const restrict ptr_out = out->data;
    for (uint64_t i = 0; i < a->rows; ++i)
    {
        for (uint64_t j = 0; j < b->cols; ++j)
        {
            double v = 0;
            for (uint64_t k = 0; k < a->cols; ++k)
                v += ptr_a[i * a->cols + k] * ptr_b[k * b->cols + j];

            ptr_out[i * out->cols + j] -= v;
        }
    }
}

void hybsol_matrix_subtract_inplace(const hybsol_matrix_t *const a, const hybsol_matrix_t *const b)
{
    HYBSOL_ASSERT(a->rows == b->rows && a->cols == b->cols, "Matrices have different dimensions.");
    for (uint64_t i = 0; i < a->rows * a->cols; ++i)
        a->data[i] -= b->data[i];
}

hybsol_result_t hybsol_matrix_lu_decompose(const hybsol_matrix_t *const m)
{
    if (m->rows != m->cols)
        return HYBSOL_ERROR_INVALID_ARGUMENT;

    for (uint64_t i = 0; i < m->rows; ++i)
    {
        // Compute a row of U
        for (uint64_t j = i; j < m->rows; ++j)
        {
            double u_ij = m->data[i * m->cols + j];
            for (uint64_t k = 0; k < i; ++k)
            {
                u_ij -= m->data[i * m->cols + k] * m->data[k * m->cols + j];
            }
            m->data[i * m->cols + j] = u_ij;
        }

        // No pivoting: an exactly zero pivot can not be divided by.
        const double pivot = m->data[i * m->cols + i];
        if (pivot == 0.0)
            return HYBSOL_ERROR_SINGULAR;

        // Compute a column of L (the first element is always implicitly 1, so skip it)
        for (uint64_t j = i + 1; j < m->cols; ++j)
        {
            double l_ij = m->data[j * m->cols + i];
            for (uint64_t k = 0; k < i; ++k)
            {
                l_ij -= m->data[k * m->cols + i] * m->data[j * m->cols + k];
            }
            m->data[j * m->cols + i] = l_ij / pivot;
        }
    }

    return HYBSOL_SUCCESS;
}

hybsol_result_t hybsol_matrix_lu_solve(const hybsol_matrix_t *const m, const hybsol_matrix_t *const b,
                                       const hybsol_matrix_t *const out)
{
    if (m->rows != m->cols || m->rows != b->rows)
        return HYBSOL_ERROR_INVALID_ARGUMENT;
    if (b->rows != out->rows || b->cols != out->cols)
        return HYBSOL_ERROR_INVALID_ARGUMENT;

    // Have to deal with every column in the same way
    for (uint64_t i_col = 0; i_col < b->cols; ++i_col)
    {
        // Forward substitution to solve the L part (diagonal is 1)
        for (uint64_t i = 0; i < m->rows; ++i)
        {
            double v = b->data[i * b->cols + i_col];
            for (uint64_t j = 0; j < i; ++j)
            {
                v -= m->data[i * m->cols + j] * out->data[j * out->cols + i_col];
            }
            out->data[i * out->cols + i_col] = v;
        }

        // Backward substitution to solve the U part
        for (uint64_t i = m->rows; i > 0; --i)
        {
            double v = out->data[(i - 1) * out->cols + i_col];
            for (uint64_t j = i; j < m->rows; ++j)
            {
                v -= m->data[(i - 1) * m->cols + j] * out->data[j * out->cols + i_col];
            }
            out->data[(i - 1) * out->cols + i_col] = v / m->data[(i - 1) * m->cols + (i - 1)];
        }
    }

    return HYBSOL_SUCCESS;
}
