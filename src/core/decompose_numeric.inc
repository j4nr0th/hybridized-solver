/*
 * The value-carrying half of the decomposition, instantiated once per
 * floating-point type.
 *
 * Unguarded on purpose: ``decompose.c`` includes this file twice, after
 * defining
 *
 *   HYBSOL_SCALAR    the type the system stores its blocks in
 *   HYBSOL_ACC       the type products accumulate in
 *   HYBSOL_VIEW      the view type spelling that storage
 *   HYBSOL_KERNEL(n) the dense-kernel name matching ``HYBSOL_VIEW``
 *   HYBSOL_FN(name)  the name the resulting function gets
 *   HYBSOL_NARROWED  1 when ``HYBSOL_SCALAR`` is narrower than the doubles the
 *                    solve protocol hands around
 *
 * Vectors are *not* templated: :c:func:`hybsol_system_solve` takes doubles for
 * both precisions, so the two kernels at the top take ``HYBSOL_SCALAR`` factors
 * against a ``hybsol_matrix_t``. Only the factors narrow; the vector
 * arithmetic stays in double, so a single-precision solve loses no accuracy it
 * does not have to lose.
 *
 * Everything here is ``static``: the instantiations are private to
 * ``decompose.c`` and reached through its dispatchers. Every macro is
 * undefined at the bottom so a second inclusion is well defined.
 */

#if HYBSOL_NARROWED

/** ``out -= a @ b`` with ``a`` in system storage and ``b``/``out`` doubles. */
static void HYBSOL_FN(vec_multiply_sub)(const HYBSOL_VIEW *const a, const hybsol_matrix_t *const b,
                                        const hybsol_matrix_t *const out)
{
    CUTL_ASSERT(a->cols == b->rows,
                "Matrix multiplication: number of columns of A (%llu) does not match number of rows of B (%llu).",
                (unsigned long long)a->cols, (unsigned long long)b->rows);
    CUTL_ASSERT(out->rows == a->rows && out->cols == b->cols, "Output matrix has incorrect dimensions.");

    const HYBSOL_SCALAR *const restrict ptr_a = a->data;
    const double *const restrict ptr_b = b->data;
    double *const restrict ptr_out = out->data;
    for (uint64_t i = 0; i < a->rows; ++i)
    {
        for (uint64_t j = 0; j < b->cols; ++j)
        {
            HYBSOL_ACC v = 0;
            for (uint64_t k = 0; k < a->cols; ++k)
                v += (HYBSOL_ACC)ptr_a[i * a->cols + k] * ptr_b[k * b->cols + j];

            ptr_out[i * out->cols + j] -= (double)v;
        }
    }
}

/** Solve ``m @ x = b`` with ``m`` in system storage and ``b``/``out`` doubles. */
static hybsol_result_t HYBSOL_FN(vec_lu_solve)(const HYBSOL_VIEW *const m, const hybsol_matrix_t *const b,
                                               const hybsol_matrix_t *const out)
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
            double v = b->data[i * b->cols + i_col];
            for (uint64_t j = 0; j < i; ++j)
                v -= (double)m->data[i * m->cols + j] * out->data[j * out->cols + i_col];
            out->data[i * out->cols + i_col] = v;
        }

        // Backward substitution to solve the U part
        for (uint64_t i = m->rows; i > 0; --i)
        {
            double v = out->data[(i - 1) * out->cols + i_col];
            for (uint64_t j = i; j < m->rows; ++j)
                v -= (double)m->data[(i - 1) * m->cols + j] * out->data[j * out->cols + i_col];
            out->data[(i - 1) * out->cols + i_col] = v / (double)m->data[(i - 1) * m->cols + (i - 1)];
        }
    }

    return HYBSOL_SUCCESS;
}

#else

/* The factors are already doubles, so these are the plain kernels. */

static void HYBSOL_FN(vec_multiply_sub)(const HYBSOL_VIEW *const a, const hybsol_matrix_t *const b,
                                        const hybsol_matrix_t *const out)
{
    hybsol_matrix_multiply_sub_inplace(a, b, out);
}

static hybsol_result_t HYBSOL_FN(vec_lu_solve)(const HYBSOL_VIEW *const m, const hybsol_matrix_t *const b,
                                               const hybsol_matrix_t *const out)
{
    return hybsol_matrix_lu_solve(m, b, out);
}

#endif

static hybsol_result_t HYBSOL_FN(decompose_diagonal)(hybsol_system_t *const sys, const uint64_t idx)
{
    hybsol_result_t res = hybsol_require_mutable(sys);
    if (res != HYBSOL_SUCCESS)
        return res;

    hybsol_row_entry_t *diag;
    find_diagonal(sys, idx, &diag);

    const uint64_t block_size = hybsol_block_size(sys, idx);
    const HYBSOL_VIEW mat = HYBSOL_KERNEL(view)(block_size, block_size, (HYBSOL_SCALAR *)diag->vals);
    res = HYBSOL_KERNEL(lu_decompose)(&mat);
    if (res != HYBSOL_SUCCESS)
    {
        sys->failing_block = idx;
        return res;
    }

    sys->diag_decomposed[idx] = 1;
    return HYBSOL_SUCCESS;
}

static hybsol_result_t HYBSOL_FN(apply_diagonal_inverse)(hybsol_system_t *const sys, const uint64_t idx)
{
    hybsol_result_t res = hybsol_require_mutable(sys);
    if (res != HYBSOL_SUCCESS)
        return res;

    hybsol_row_entry_t *diag;
    find_diagonal(sys, idx, &diag);

    if (!sys->diag_decomposed[idx])
        return HYBSOL_ERROR_NOT_DECOMPOSED;

    hybsol_row_t *const row = sys->rows + idx;
    uint64_t i_diag;
    (void)hybsol_row_find(row, idx, &i_diag);

    const uint64_t block_size = hybsol_block_size(sys, idx);
    const HYBSOL_VIEW mat_diag = HYBSOL_KERNEL(view)(block_size, block_size, (HYBSOL_SCALAR *)diag->vals);
    for (uint64_t i = i_diag + 1; i < row->count; ++i)
    {
        hybsol_row_entry_t *const entry = row->entries[i];
        const HYBSOL_VIEW mat_entry =
            HYBSOL_KERNEL(view)(block_size, hybsol_block_size(sys, entry->col), (HYBSOL_SCALAR *)entry->vals);
        HYBSOL_KERNEL(lu_solve)(&mat_diag, &mat_entry, &mat_entry);
    }

    return HYBSOL_SUCCESS;
}

static hybsol_result_t HYBSOL_FN(solve_diagonal)(hybsol_system_t *const sys, const uint64_t idx,
                                                 const HYBSOL_VIEW *const b, const HYBSOL_VIEW *const x)
{
    hybsol_row_entry_t *diag;
    find_diagonal(sys, idx, &diag);

    if (!sys->diag_decomposed[idx])
        return HYBSOL_ERROR_NOT_DECOMPOSED;

    const uint64_t block_size = hybsol_block_size(sys, idx);
    CUTL_ASSERT(b->rows == block_size, "Block row %llu is %llu rows, but the right-hand side has %llu.",
                (unsigned long long)idx, (unsigned long long)block_size, (unsigned long long)b->rows);
    CUTL_ASSERT(x->rows == b->rows && x->cols == b->cols,
                "The destination is %llu x %llu, but the right-hand side is %llu x %llu.", (unsigned long long)x->rows,
                (unsigned long long)x->cols, (unsigned long long)b->rows, (unsigned long long)b->cols);

    const HYBSOL_VIEW mat_diag = HYBSOL_KERNEL(view)(block_size, block_size, (HYBSOL_SCALAR *)diag->vals);
    return HYBSOL_KERNEL(lu_solve)(&mat_diag, b, x);
}

static void HYBSOL_FN(apply_operations)(const hybsol_system_t *const sys, const uint64_t n_ops,
                                        const hybsol_operation_t *const ops, double *const vec)
{
    for (uint64_t i = 0; i < n_ops; ++i)
    {
        const hybsol_operation_t *const op = ops + i;

        switch (op->type)
        {
        case HYBSOL_OPERATION_ELIMINATE:
        {
            uint64_t i_block;
            (void)hybsol_row_find(sys->rows + op->idx_row, op->idx_col, &i_block);
            const HYBSOL_VIEW block_mat =
                HYBSOL_KERNEL(view)(hybsol_block_size(sys, op->idx_row), hybsol_block_size(sys, op->idx_col),
                                    (HYBSOL_SCALAR *)sys->rows[op->idx_row].entries[i_block]->vals);
            const hybsol_matrix_t in_vec =
                hybsol_matrix_view(hybsol_block_size(sys, op->idx_col), 1, vec + sys->block_offsets[op->idx_col]);
            const hybsol_matrix_t out_vec =
                hybsol_matrix_view(hybsol_block_size(sys, op->idx_row), 1, vec + sys->block_offsets[op->idx_row]);
            HYBSOL_FN(vec_multiply_sub)(&block_mat, &in_vec, &out_vec);
        }
        break;

        case HYBSOL_OPERATION_INVERT_DIAGONAL:
        {
            uint64_t i_block;
            (void)hybsol_row_find(sys->rows + op->idx_row, op->idx_row, &i_block);
            const HYBSOL_VIEW block_mat =
                HYBSOL_KERNEL(view)(hybsol_block_size(sys, op->idx_row), hybsol_block_size(sys, op->idx_row),
                                    (HYBSOL_SCALAR *)sys->rows[op->idx_row].entries[i_block]->vals);
            const hybsol_matrix_t inout_vec =
                hybsol_matrix_view(hybsol_block_size(sys, op->idx_row), 1, vec + sys->block_offsets[op->idx_row]);
            HYBSOL_FN(vec_lu_solve)(&block_mat, &inout_vec, &inout_vec);
        }
        break;
        }
    }
}

static void HYBSOL_FN(solve_upper)(const hybsol_system_t *const sys, double *const y)
{
    // Solve the U x = y system, where the diagonal blocks are the identity
    for (uint64_t i = sys->n; i > 0; --i)
    {
        const uint64_t idx_row = i - 1;
        const hybsol_row_t *const row = sys->rows + idx_row;
        if (row->count == 0)
            continue;

        const uint64_t block_size = hybsol_block_size(sys, idx_row);
        const hybsol_matrix_t vec = hybsol_matrix_view(block_size, 1, y + sys->block_offsets[idx_row]);
        for (uint64_t j = row->count; j > 0 && idx_row < row->entries[j - 1]->col; --j)
        {
            hybsol_row_entry_t *const entry = row->entries[j - 1];
            const HYBSOL_VIEW mat_entry =
                HYBSOL_KERNEL(view)(block_size, hybsol_block_size(sys, entry->col), (HYBSOL_SCALAR *)entry->vals);
            const hybsol_matrix_t vec_entry =
                hybsol_matrix_view(hybsol_block_size(sys, entry->col), 1, y + sys->block_offsets[entry->col]);
            HYBSOL_FN(vec_multiply_sub)(&mat_entry, &vec_entry, &vec);
        }
    }
}

#undef HYBSOL_SCALAR
#undef HYBSOL_ACC
#undef HYBSOL_VIEW
#undef HYBSOL_KERNEL
#undef HYBSOL_FN
#undef HYBSOL_NARROWED
