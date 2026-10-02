/*
 * The value-carrying half of the decomposition, instantiated once per
 * floating-point type. ``decomposition_numeric.c`` includes this file twice,
 * unguarded on purpose, after defining
 *
 *   HYBSOL_SCALAR    the type the decomposition stores its blocks in
 *   HYBSOL_ACC       the type products accumulate in
 *   HYBSOL_VIEW      the view type spelling that storage
 *   HYBSOL_KERNEL(n) the dense-kernel name matching ``HYBSOL_VIEW``
 *   HYBSOL_FN(name)  the name the resulting function gets
 *   HYBSOL_NARROWED  1 when ``HYBSOL_SCALAR`` is narrower than the doubles the
 *                    solve protocol hands around
 *
 * Vectors are *not* templated: the solve takes doubles for both precisions, so the two kernels at
 * the top take ``HYBSOL_SCALAR`` factors against a ``hybsol_matrix_t``. Everything is ``static``
 * and every macro is undefined at the bottom, so a second inclusion is well defined.
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

/** The diagonal block of ``idx``, which the graph places at ``row_n_elim``. */
static hybsol_row_entry_t *HYBSOL_FN(diagonal)(const hybsol_decomposition_t *const dec, const uint64_t idx)
{
    hybsol_row_entry_t *const entry = dec->rows[idx].entries[dec->row_n_elim[idx]];
    CUTL_ASSERT(entry->col == idx, "Block row %llu has its diagonal at slot %llu, which holds column %llu.",
                (unsigned long long)idx, (unsigned long long)dec->row_n_elim[idx], (unsigned long long)entry->col);
    return entry;
}

static hybsol_result_t HYBSOL_FN(factorize_diagonal)(hybsol_decomposition_t *const dec, const uint64_t idx)
{
    hybsol_row_entry_t *const diag = HYBSOL_FN(diagonal)(dec, idx);
    const uint64_t block_size = hybsol_decomposition_block_size(dec, idx);
    const HYBSOL_VIEW mat = HYBSOL_KERNEL(view)(block_size, block_size, (HYBSOL_SCALAR *)diag->vals);

    const hybsol_result_t res = HYBSOL_KERNEL(lu_decompose)(&mat);
    if (res != HYBSOL_SUCCESS)
        dec->failing_block = idx;
    return res;
}

/** Scale everything above the diagonal by the inverse of the factors. */
static void HYBSOL_FN(invert_diagonal_into_row)(const hybsol_decomposition_t *const dec, const uint64_t idx)
{
    hybsol_row_t *const row = dec->rows + idx;
    const uint64_t i_diag = dec->row_n_elim[idx];
    const hybsol_row_entry_t *const diag = row->entries[i_diag];

    const uint64_t block_size = hybsol_decomposition_block_size(dec, idx);
    const HYBSOL_VIEW mat_diag = HYBSOL_KERNEL(view)(block_size, block_size, (HYBSOL_SCALAR *)diag->vals);
    for (uint64_t i = i_diag + 1; i < row->count; ++i)
    {
        hybsol_row_entry_t *const entry = row->entries[i];
        const HYBSOL_VIEW mat_entry = HYBSOL_KERNEL(view)(block_size, hybsol_decomposition_block_size(dec, entry->col),
                                                          (HYBSOL_SCALAR *)entry->vals);
        HYBSOL_KERNEL(lu_solve)(&mat_diag, &mat_entry, &mat_entry);
    }
}

/**
 * One step of the elimination: subtract the ``k``-th source row from what is
 * left of ``row_tgt``.
 *
 * The graph already fixed both rows' final layout, so this is a merge of two
 * sorted index ranges writing into slots that already exist: nothing is
 * allocated and no entry moves.
 */
static void HYBSOL_FN(eliminate_step)(hybsol_decomposition_t *const dec, const uint64_t row_tgt, const uint64_t k,
                                      HYBSOL_SCALAR *const scratch)
{
    hybsol_row_t *const row_t = dec->rows + row_tgt;
    hybsol_row_entry_t *const mat = row_t->entries[k];
    hybsol_row_t *const row_s = dec->rows + mat->col;

    const uint64_t size_tgt = hybsol_decomposition_block_size(dec, row_tgt);
    const uint64_t size_src = hybsol_decomposition_block_size(dec, mat->col);
    const HYBSOL_VIEW mat_view = HYBSOL_KERNEL(view)(size_tgt, size_src, (HYBSOL_SCALAR *)mat->vals);

    // Past the multiplier in the target, past the diagonal in the source: both
    // are exactly the columns greater than the source's.
    uint64_t a = k + 1;
    for (uint64_t b = dec->row_n_elim[mat->col] + 1; b < row_s->count; ++b)
    {
        const hybsol_row_entry_t *const src = row_s->entries[b];
        while (a < row_t->count && row_t->entries[a]->col < src->col)
            ++a;

        CUTL_ASSERT(a < row_t->count, "Row %llu gained column %llu only at step %llu, which the walk missed.",
                    (unsigned long long)row_tgt, (unsigned long long)src->col, (unsigned long long)k);

        const uint64_t n_cols = hybsol_decomposition_block_size(dec, src->col);
        const HYBSOL_VIEW mat_src = HYBSOL_KERNEL(view)(size_src, n_cols, (HYBSOL_SCALAR *)src->vals);
        const HYBSOL_VIEW out = HYBSOL_KERNEL(view)(size_tgt, n_cols, scratch);
        HYBSOL_KERNEL(multiply)(&mat_view, &mat_src, &out);

        // Always a subtraction, never an assignment. A column the target did not
        // have until this step sits at zero, so subtracting the product writes
        // exactly the negated product the fill-in needs; a column it already had
        // has its own value subtracted, as before. No branch is needed.
        hybsol_row_entry_t *const dst = row_t->entries[a];
        const HYBSOL_VIEW mat_dst = HYBSOL_KERNEL(view)(size_tgt, n_cols, (HYBSOL_SCALAR *)dst->vals);
        HYBSOL_KERNEL(subtract_inplace)(&mat_dst, &out);
        ++a;
    }
}

/** ``vec[row_tgt] -= block(row_tgt, source) @ vec[source]``. */
static void HYBSOL_FN(forward_step)(const hybsol_decomposition_t *const dec, const uint64_t row_tgt, const uint64_t k,
                                    double *const vec)
{
    const hybsol_row_entry_t *const block = dec->rows[row_tgt].entries[k];
    const hybsol_matrix_t in_vec =
        hybsol_matrix_view(hybsol_decomposition_block_size(dec, block->col), 1, vec + dec->block_offsets[block->col]);
    const hybsol_matrix_t out_vec =
        hybsol_matrix_view(hybsol_decomposition_block_size(dec, row_tgt), 1, vec + dec->block_offsets[row_tgt]);
    const HYBSOL_VIEW mat =
        HYBSOL_KERNEL(view)(hybsol_decomposition_block_size(dec, row_tgt),
                            hybsol_decomposition_block_size(dec, block->col), (HYBSOL_SCALAR *)block->vals);
    HYBSOL_FN(vec_multiply_sub)(&mat, &in_vec, &out_vec);
}

/** ``vec[row_tgt] = L[row_tgt]^{-1} vec[row_tgt]``. */
static void HYBSOL_FN(forward_diagonal)(const hybsol_decomposition_t *const dec, const uint64_t row_tgt,
                                        double *const vec)
{
    const hybsol_row_entry_t *const diag = HYBSOL_FN(diagonal)(dec, row_tgt);
    const uint64_t block_size = hybsol_decomposition_block_size(dec, row_tgt);
    const HYBSOL_VIEW mat = HYBSOL_KERNEL(view)(block_size, block_size, (HYBSOL_SCALAR *)diag->vals);
    const hybsol_matrix_t inout = hybsol_matrix_view(block_size, 1, vec + dec->block_offsets[row_tgt]);
    HYBSOL_FN(vec_lu_solve)(&mat, &inout, &inout);
}

static void HYBSOL_FN(apply_operations)(const hybsol_decomposition_t *const dec, const uint64_t n_ops,
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
            (void)hybsol_row_find(dec->rows + op->idx_row, op->idx_col, &i_block);
            const HYBSOL_VIEW block_mat = HYBSOL_KERNEL(view)(
                hybsol_decomposition_block_size(dec, op->idx_row), hybsol_decomposition_block_size(dec, op->idx_col),
                (HYBSOL_SCALAR *)dec->rows[op->idx_row].entries[i_block]->vals);
            const hybsol_matrix_t in_vec = hybsol_matrix_view(hybsol_decomposition_block_size(dec, op->idx_col), 1,
                                                              vec + dec->block_offsets[op->idx_col]);
            const hybsol_matrix_t out_vec = hybsol_matrix_view(hybsol_decomposition_block_size(dec, op->idx_row), 1,
                                                               vec + dec->block_offsets[op->idx_row]);
            HYBSOL_FN(vec_multiply_sub)(&block_mat, &in_vec, &out_vec);
        }
        break;

        case HYBSOL_OPERATION_INVERT_DIAGONAL:
        {
            uint64_t i_block;
            (void)hybsol_row_find(dec->rows + op->idx_row, op->idx_row, &i_block);
            const HYBSOL_VIEW block_mat = HYBSOL_KERNEL(view)(
                hybsol_decomposition_block_size(dec, op->idx_row), hybsol_decomposition_block_size(dec, op->idx_row),
                (HYBSOL_SCALAR *)dec->rows[op->idx_row].entries[i_block]->vals);
            const hybsol_matrix_t inout_vec = hybsol_matrix_view(hybsol_decomposition_block_size(dec, op->idx_row), 1,
                                                                 vec + dec->block_offsets[op->idx_row]);
            HYBSOL_FN(vec_lu_solve)(&block_mat, &inout_vec, &inout_vec);
        }
        break;
        }
    }
}

static void HYBSOL_FN(solve_upper)(const hybsol_decomposition_t *const dec, double *const y)
{
    // Solve the U x = y system, where the diagonal blocks are the identity
    for (uint64_t i = dec->n; i > 0; --i)
    {
        const uint64_t idx_row = i - 1;
        const hybsol_row_t *const row = dec->rows + idx_row;
        if (row->count == 0)
            continue;

        const uint64_t block_size = hybsol_decomposition_block_size(dec, idx_row);
        const hybsol_matrix_t vec = hybsol_matrix_view(block_size, 1, y + dec->block_offsets[idx_row]);
        for (uint64_t j = row->count; j > 0 && idx_row < row->entries[j - 1]->col; --j)
        {
            const hybsol_row_entry_t *const entry = row->entries[j - 1];
            const HYBSOL_VIEW mat_entry = HYBSOL_KERNEL(view)(
                block_size, hybsol_decomposition_block_size(dec, entry->col), (HYBSOL_SCALAR *)entry->vals);
            const hybsol_matrix_t vec_entry = hybsol_matrix_view(hybsol_decomposition_block_size(dec, entry->col), 1,
                                                                 y + dec->block_offsets[entry->col]);
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
