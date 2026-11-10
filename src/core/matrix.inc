/*
 * Dense matrix kernels, instantiated once per floating-point type. ``matrix.c``
 * includes this file twice, deliberately unguarded, once per public spelling,
 * after defining
 *
 *   HYBSOL_SCALAR   the type the elements are stored in
 *   HYBSOL_ACC      the type the product loops accumulate in
 *   HYBSOL_VIEW     the view type spelling those elements
 *   HYBSOL_FN(name) the name the resulting function gets
 *
 * ``HYBSOL_ACC`` is separate from ``HYBSOL_SCALAR`` so that accumulating in double while storing
 * single needs no restructuring: casts to it are no-ops for the double instantiation. Every macro
 * is undefined at the bottom, so a second inclusion is well-defined.
 */

HYBSOL_VIEW HYBSOL_FN(view)(const uint64_t rows, const uint64_t cols, HYBSOL_SCALAR *const data)
{
    return (HYBSOL_VIEW){.rows = rows, .cols = cols, .data = data};
}

void HYBSOL_FN(multiply)(const HYBSOL_VIEW *const a, const HYBSOL_VIEW *const b, const HYBSOL_VIEW *const out)
{
    CUTL_ASSERT(a->cols == b->rows,
                "Matrix multiplication: number of columns of A (%llu) does not match number of rows of B (%llu).",
                (unsigned long long)a->cols, (unsigned long long)b->rows);
    CUTL_ASSERT(out->rows == a->rows && out->cols == b->cols, "Output matrix has incorrect dimensions.");

    const HYBSOL_SCALAR *const restrict ptr_a = a->data;
    const HYBSOL_SCALAR *const restrict ptr_b = b->data;
    HYBSOL_SCALAR *const restrict ptr_out = out->data;

    // Ordered i-k-j rather than i-j-k: the inner loop then runs along a row of B and of out, both
    // unit stride, so it vectorizes without being asked. Every out[i][j] still accumulates over k in
    // ascending order and the j-loop is elementwise, so nothing is reassociated and the result is
    // bit-for-bit what i-j-k produced.
    for (uint64_t i = 0; i < a->rows; ++i)
    {
        for (uint64_t j = 0; j < out->cols; ++j)
            ptr_out[i * out->cols + j] = 0;

        for (uint64_t k = 0; k < a->cols; ++k)
        {
            const HYBSOL_ACC aik = (HYBSOL_ACC)ptr_a[i * a->cols + k];
            const HYBSOL_SCALAR *const restrict row_b = ptr_b + k * b->cols;
            HYBSOL_SCALAR *const restrict row_out = ptr_out + i * out->cols;
            for (uint64_t j = 0; j < out->cols; ++j)
                row_out[j] = (HYBSOL_SCALAR)((HYBSOL_ACC)row_out[j] + aik * (HYBSOL_ACC)row_b[j]);
        }
    }
}

void HYBSOL_FN(multiply_sub_inplace)(const HYBSOL_VIEW *const a, const HYBSOL_VIEW *const b,
                                     const HYBSOL_VIEW *const out)
{
    CUTL_ASSERT(a->cols == b->rows, "Columns of A must match the rows of B (%llu vs %llu).",
                (unsigned long long)a->cols, (unsigned long long)b->rows);
    CUTL_ASSERT(out->rows == a->rows && out->cols == b->cols, "Output matrix has incorrect dimensions.");

    // Compute A @ B and subtract it from whatever is in OUT
    const HYBSOL_SCALAR *const restrict ptr_a = a->data;
    const HYBSOL_SCALAR *const restrict ptr_b = b->data;
    HYBSOL_SCALAR *const restrict ptr_out = out->data;
    for (uint64_t i = 0; i < a->rows; ++i)
    {
        for (uint64_t j = 0; j < b->cols; ++j)
        {
            HYBSOL_ACC v = 0;
            for (uint64_t k = 0; k < a->cols; ++k)
                v += (HYBSOL_ACC)ptr_a[i * a->cols + k] * (HYBSOL_ACC)ptr_b[k * b->cols + j];

            ptr_out[i * out->cols + j] -= v;
        }
    }
}

void HYBSOL_FN(subtract_inplace)(const HYBSOL_VIEW *const a, const HYBSOL_VIEW *const b)
{
    CUTL_ASSERT(a->rows == b->rows && a->cols == b->cols, "Matrices have different dimensions.");
    for (uint64_t i = 0; i < a->rows * a->cols; ++i)
        a->data[i] -= b->data[i];
}

hybsol_result_t HYBSOL_FN(lu_decompose)(const HYBSOL_VIEW *const m)
{
    CUTL_ASSERT(m->rows == m->cols, "LU needs a square matrix, but this one is %llu x %llu.",
                (unsigned long long)m->rows, (unsigned long long)m->cols);

    for (uint64_t i = 0; i < m->rows; ++i)
    {
        // One row of U
        for (uint64_t j = i; j < m->rows; ++j)
        {
            HYBSOL_ACC u_ij = (HYBSOL_ACC)m->data[i * m->cols + j];
            for (uint64_t k = 0; k < i; ++k)
            {
                u_ij -= (HYBSOL_ACC)m->data[i * m->cols + k] * (HYBSOL_ACC)m->data[k * m->cols + j];
            }
            m->data[i * m->cols + j] = (HYBSOL_SCALAR)u_ij;
        }

        // No pivoting: an exactly zero pivot can not be divided by.
        const HYBSOL_SCALAR pivot = m->data[i * m->cols + i];
        if (pivot == 0)
            return HYBSOL_ERROR_SINGULAR;

        // One column of L; the first element is implicitly 1, so skip it
        for (uint64_t j = i + 1; j < m->cols; ++j)
        {
            HYBSOL_ACC l_ij = (HYBSOL_ACC)m->data[j * m->cols + i];
            for (uint64_t k = 0; k < i; ++k)
            {
                l_ij -= (HYBSOL_ACC)m->data[k * m->cols + i] * (HYBSOL_ACC)m->data[j * m->cols + k];
            }
            m->data[j * m->cols + i] = (HYBSOL_SCALAR)(l_ij / (HYBSOL_ACC)pivot);
        }
    }

    return HYBSOL_SUCCESS;
}

hybsol_result_t HYBSOL_FN(lu_solve)(const HYBSOL_VIEW *const m, const HYBSOL_VIEW *const b,
                                    const HYBSOL_VIEW *const out)
{
    CUTL_ASSERT(m->rows == m->cols, "LU needs a square matrix, but this one is %llu x %llu.",
                (unsigned long long)m->rows, (unsigned long long)m->cols);
    CUTL_ASSERT(m->rows == b->rows, "The right-hand side has %llu rows, but the factors have %llu.",
                (unsigned long long)b->rows, (unsigned long long)m->rows);
    CUTL_ASSERT(b->rows == out->rows && b->cols == out->cols,
                "The destination is %llu x %llu, but the right-hand side is %llu x %llu.",
                (unsigned long long)out->rows, (unsigned long long)out->cols, (unsigned long long)b->rows,
                (unsigned long long)b->cols);

    // Every column is solved the same way
    for (uint64_t i_col = 0; i_col < b->cols; ++i_col)
    {
        // Forward substitution to solve the L part (diagonal is 1)
        for (uint64_t i = 0; i < m->rows; ++i)
        {
            HYBSOL_ACC v = (HYBSOL_ACC)b->data[i * b->cols + i_col];
            for (uint64_t j = 0; j < i; ++j)
            {
                v -= (HYBSOL_ACC)m->data[i * m->cols + j] * (HYBSOL_ACC)out->data[j * out->cols + i_col];
            }
            out->data[i * out->cols + i_col] = (HYBSOL_SCALAR)v;
        }

        // Backward substitution to solve the U part
        for (uint64_t i = m->rows; i > 0; --i)
        {
            HYBSOL_ACC v = (HYBSOL_ACC)out->data[(i - 1) * out->cols + i_col];
            for (uint64_t j = i; j < m->rows; ++j)
            {
                v -= (HYBSOL_ACC)m->data[(i - 1) * m->cols + j] * (HYBSOL_ACC)out->data[j * out->cols + i_col];
            }
            out->data[(i - 1) * out->cols + i_col] =
                (HYBSOL_SCALAR)(v / (HYBSOL_ACC)m->data[(i - 1) * m->cols + (i - 1)]);
        }
    }

    return HYBSOL_SUCCESS;
}

#undef HYBSOL_SCALAR
#undef HYBSOL_ACC
#undef HYBSOL_VIEW
#undef HYBSOL_FN
