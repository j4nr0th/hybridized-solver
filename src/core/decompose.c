/** @file core/decompose.c
 * Block LU decomposition, replay of the recorded operations and the solves
 * that build on them.
 */

#include "internal.h"

hybsol_result_t hybsol_ops_append(hybsol_system_t *const sys, const hybsol_operation_t op)
{
    hybsol_result_t res = HYBSOL_SUCCESS;

#pragma omp critical(hybsol_ops_append)
    {
        if (sys->n_ops == sys->ops_capacity)
        {
            const uint64_t new_capacity = sys->ops_capacity ? sys->ops_capacity * 2 : 8;
            hybsol_operation_t *const ptr = hybsol_grow(sys->ops, (size_t)new_capacity * sizeof(*ptr));
            if (ptr == NULL)
            {
                res = HYBSOL_ERROR_OUT_OF_MEMORY;
            }
            else
            {
                sys->ops = ptr;
                sys->ops_capacity = new_capacity;
            }
        }
        if (res == HYBSOL_SUCCESS)
        {
            sys->ops[sys->n_ops] = op;
            sys->n_ops += 1;
        }
    }

    return res;
}

int hybsol_system_is_decomposed(const hybsol_system_t *const sys)
{
    return sys->decomposed ? 1 : 0;
}

uint64_t hybsol_system_n_operations(const hybsol_system_t *const sys)
{
    return sys->n_ops;
}

const hybsol_operation_t *hybsol_system_operations(const hybsol_system_t *const sys)
{
    return sys->ops;
}

/* ------------------------------------------------------------------------- */
/* Diagonal block work                                                       */
/* ------------------------------------------------------------------------- */

static hybsol_result_t find_diagonal(hybsol_system_t *const sys, const uint64_t idx, hybsol_row_entry_t **const out)
{
    hybsol_result_t res = hybsol_check_index(sys, idx);
    if (res != HYBSOL_SUCCESS)
        return res;

    hybsol_row_t *const row = sys->rows + idx;
    uint64_t i_diag;
    if (!hybsol_row_find(row, idx, &i_diag))
        return HYBSOL_ERROR_MISSING_DIAGONAL;

    *out = row->entries[i_diag];
    return HYBSOL_SUCCESS;
}

hybsol_result_t hybsol_system_decompose_diagonal(hybsol_system_t *const sys, const uint64_t idx)
{
    hybsol_result_t res = hybsol_require_mutable(sys);
    if (res != HYBSOL_SUCCESS)
        return res;

    hybsol_row_entry_t *diag;
    res = find_diagonal(sys, idx, &diag);
    if (res != HYBSOL_SUCCESS)
        return res;

    const uint64_t block_size = hybsol_block_size(sys, idx);
    const hybsol_matrix_t mat = hybsol_matrix_view(block_size, block_size, diag->vals);
    res = hybsol_matrix_lu_decompose(&mat);
    if (res != HYBSOL_SUCCESS)
        return res;

    sys->diag_decomposed[idx] = 1;
    return HYBSOL_SUCCESS;
}

hybsol_result_t hybsol_system_apply_diagonal_inverse(hybsol_system_t *const sys, const uint64_t idx)
{
    hybsol_result_t res = hybsol_require_mutable(sys);
    if (res != HYBSOL_SUCCESS)
        return res;

    hybsol_row_entry_t *diag;
    res = find_diagonal(sys, idx, &diag);
    if (res != HYBSOL_SUCCESS)
        return res;

    if (!sys->diag_decomposed[idx])
        return HYBSOL_ERROR_NOT_DECOMPOSED;

    hybsol_row_t *const row = sys->rows + idx;
    uint64_t i_diag;
    (void)hybsol_row_find(row, idx, &i_diag);

    const uint64_t block_size = hybsol_block_size(sys, idx);
    const hybsol_matrix_t mat_diag = hybsol_matrix_view(block_size, block_size, diag->vals);
    for (uint64_t i = i_diag + 1; i < row->count; ++i)
    {
        hybsol_row_entry_t *const entry = row->entries[i];
        const hybsol_matrix_t mat_entry =
            hybsol_matrix_view(block_size, hybsol_block_size(sys, entry->col), entry->vals);
        hybsol_matrix_lu_solve(&mat_diag, &mat_entry, &mat_entry);
    }

    return HYBSOL_SUCCESS;
}

hybsol_result_t hybsol_system_solve_diagonal(hybsol_system_t *const sys, const uint64_t idx,
                                             const hybsol_matrix_t *const b, const hybsol_matrix_t *const x)
{
    if (b == NULL || x == NULL)
        return HYBSOL_ERROR_INVALID_ARGUMENT;

    hybsol_row_entry_t *diag;
    hybsol_result_t res = find_diagonal(sys, idx, &diag);
    if (res != HYBSOL_SUCCESS)
        return res;

    if (!sys->diag_decomposed[idx])
        return HYBSOL_ERROR_NOT_DECOMPOSED;

    const uint64_t block_size = hybsol_block_size(sys, idx);
    if (b->rows != block_size || x->rows != b->rows || x->cols != b->cols)
        return HYBSOL_ERROR_INVALID_ARGUMENT;

    const hybsol_matrix_t mat_diag = hybsol_matrix_view(block_size, block_size, diag->vals);
    return hybsol_matrix_lu_solve(&mat_diag, b, x);
}

/* ------------------------------------------------------------------------- */
/* The decomposition itself                                                   */
/* ------------------------------------------------------------------------- */

typedef enum
{
    TARGET_FREE,
    TARGET_IN_USE,
    TARGET_DONE,
} target_status_t;

typedef struct
{
    target_status_t status;
    uint64_t idx_src_needed;
} target_row_t;

hybsol_result_t hybsol_system_decompose(hybsol_system_t *const sys, const uint64_t n_threads)
{
    if (sys->decomposed)
        return HYBSOL_ERROR_ALREADY_DECOMPOSED;
    if (!hybsol_system_is_valid(sys))
        return HYBSOL_ERROR_SYSTEM_INVALID;

    const int threads = hybsol_resolve_threads(n_threads);
    HYBSOL_MARK_USED(threads);
    sys->n_ops = 0;

    target_row_t *const target_status = hybsol_alloc((size_t)sys->n * sizeof(*target_status));
    if (target_status == NULL)
        return HYBSOL_ERROR_OUT_OF_MEMORY;

    hybsol_result_t shared_res = HYBSOL_SUCCESS;
    uint64_t n_tgt = 0;

    // Rows that already start with their diagonal block can be finished right
    // away; everything else waits for the elimination pass below.
#pragma omp parallel for schedule(dynamic) reduction(+ : n_tgt) default(none)                                          \
    shared(sys, target_status, shared_res) if (threads > 1) num_threads(threads)
    for (uint64_t i_row = 0; i_row < sys->n; ++i_row)
    {
        if (shared_res != HYBSOL_SUCCESS)
            continue;

        const hybsol_row_t *const row = sys->rows + i_row;
        if (row->entries[0]->col == i_row)
        {
            hybsol_result_t res = hybsol_ops_append(
                sys, (hybsol_operation_t){.type = HYBSOL_OPERATION_INVERT_DIAGONAL, .idx_row = i_row});
            if (res == HYBSOL_SUCCESS)
                res = hybsol_system_decompose_diagonal(sys, i_row);
            if (res == HYBSOL_SUCCESS)
                res = hybsol_system_apply_diagonal_inverse(sys, i_row);

            if (res != HYBSOL_SUCCESS)
            {
                shared_res = res;
                continue;
            }
            target_status[i_row].status = TARGET_DONE;
        }
        else
        {
            target_status[i_row].status = TARGET_FREE;
            target_status[i_row].idx_src_needed = row->entries[0]->col;
            n_tgt += 1;
        }
    }

    if (shared_res != HYBSOL_SUCCESS)
    {
        hybsol_free(target_status);
        sys->n_ops = 0;
        return shared_res;
    }

    // The search for work can start at the first row that is not done yet
    uint64_t i_first = 0;
    while (i_first < sys->n && target_status[i_first].status == TARGET_DONE)
        i_first += 1;

    // Eliminating a row only makes another row's diagonal available, so every
    // thread scans the same queue and takes ownership of whatever it can
    // complete next.
#pragma omp parallel default(none) shared(n_tgt, sys, target_status, shared_res, i_first) if (threads > 1)             \
    num_threads(threads)
    while (n_tgt && shared_res == HYBSOL_SUCCESS)
    {
        uint64_t i_tgt = sys->n;
        while (i_tgt == sys->n && n_tgt > 0 && shared_res == HYBSOL_SUCCESS)
        {
            for (i_tgt = i_first; i_tgt < sys->n; ++i_tgt)
            {
                target_row_t *const status = target_status + i_tgt;
                if (status->status == TARGET_FREE && target_status[status->idx_src_needed].status == TARGET_DONE)
                {
                    target_status_t old_status;
#pragma omp atomic capture
                    {
                        old_status = status->status;
                        status->status = (status->status == TARGET_FREE) ? TARGET_IN_USE : status->status;
                    }

                    // The atomic capture resolves races between threads
                    if (old_status != TARGET_FREE)
                        continue;

                    break;
                }
            }
        }
        if (n_tgt == 0 || shared_res != HYBSOL_SUCCESS)
            break;

        target_row_t *const status = target_status + i_tgt;
        hybsol_result_t res = hybsol_system_eliminate_row(sys, i_tgt, status->idx_src_needed);
        if (res != HYBSOL_SUCCESS)
        {
            shared_res = res;
            break;
        }

        res = hybsol_ops_append(sys, (hybsol_operation_t){.type = HYBSOL_OPERATION_ELIMINATE,
                                                          .idx_row = i_tgt,
                                                          .idx_col = status->idx_src_needed});
        if (res != HYBSOL_SUCCESS)
        {
            shared_res = res;
            break;
        }

        const hybsol_row_t *const row = sys->rows + i_tgt;
        const uint64_t next = hybsol_row_find_geq(row, status->idx_src_needed + 1);
        if (next >= row->count)
        {
            shared_res = HYBSOL_ERROR_INTERNAL;
            break;
        }
        const hybsol_row_entry_t *const entry = row->entries[next];
        if (entry->col == i_tgt)
        {
            // The diagonal is now first in the row, so the row is finished
            res = hybsol_system_decompose_diagonal(sys, i_tgt);
            if (res == HYBSOL_SUCCESS)
                res = hybsol_system_apply_diagonal_inverse(sys, i_tgt);
            if (res == HYBSOL_SUCCESS)
                res = hybsol_ops_append(
                    sys, (hybsol_operation_t){.type = HYBSOL_OPERATION_INVERT_DIAGONAL, .idx_row = i_tgt});
            if (res != HYBSOL_SUCCESS)
            {
                shared_res = res;
                break;
            }
            status->status = TARGET_DONE;

#pragma omp atomic update
            n_tgt -= 1;

#pragma omp critical(updating_first)
            if (i_tgt == i_first)
            {
                while (i_first < sys->n && target_status[i_first].status == TARGET_DONE)
                    i_first += 1;
            }
        }
        else
        {
            // This row still needs another elimination
            status->idx_src_needed = entry->col;
            status->status = TARGET_FREE;
        }
    }

    hybsol_free(target_status);

    if (shared_res != HYBSOL_SUCCESS)
    {
        sys->n_ops = 0;
        return shared_res;
    }

    sys->decomposed = 1;
    return HYBSOL_SUCCESS;
}

/* ------------------------------------------------------------------------- */
/* Solving                                                                    */
/* ------------------------------------------------------------------------- */

void hybsol_system_apply_operations(const hybsol_system_t *const sys, const uint64_t n_ops,
                                    const hybsol_operation_t *const ops, double *const vec)
{
    for (uint64_t i = 0; i < n_ops; ++i)
    {
        const hybsol_operation_t *const op = ops + i;
        hybsol_matrix_t block_mat;

        switch (op->type)
        {
        case HYBSOL_OPERATION_ELIMINATE: {
            (void)hybsol_lookup_block(sys, op->idx_row, op->idx_col, &block_mat);
            const hybsol_matrix_t in_vec =
                hybsol_matrix_view(hybsol_block_size(sys, op->idx_col), 1, vec + sys->block_offsets[op->idx_col]);
            const hybsol_matrix_t out_vec =
                hybsol_matrix_view(hybsol_block_size(sys, op->idx_row), 1, vec + sys->block_offsets[op->idx_row]);
            hybsol_matrix_multiply_sub_inplace(&block_mat, &in_vec, &out_vec);
        }
        break;

        case HYBSOL_OPERATION_INVERT_DIAGONAL: {
            (void)hybsol_lookup_block(sys, op->idx_row, op->idx_row, &block_mat);
            const hybsol_matrix_t inout_vec =
                hybsol_matrix_view(hybsol_block_size(sys, op->idx_row), 1, vec + sys->block_offsets[op->idx_row]);
            hybsol_matrix_lu_solve(&block_mat, &inout_vec, &inout_vec);
        }
        break;
        }
    }
}

void hybsol_system_solve_upper(const hybsol_system_t *const sys, double *const y)
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
            const hybsol_matrix_t mat_entry =
                hybsol_matrix_view(block_size, hybsol_block_size(sys, entry->col), entry->vals);
            const hybsol_matrix_t vec_entry =
                hybsol_matrix_view(hybsol_block_size(sys, entry->col), 1, y + sys->block_offsets[entry->col]);
            hybsol_matrix_multiply_sub_inplace(&mat_entry, &vec_entry, &vec);
        }
    }
}

hybsol_result_t hybsol_system_solve(hybsol_system_t *const sys, double *const vec)
{
    if (!sys->decomposed)
        return HYBSOL_ERROR_NOT_DECOMPOSED;
    if (vec == NULL)
        return HYBSOL_ERROR_INVALID_ARGUMENT;

    hybsol_system_apply_operations(sys, sys->n_ops, sys->ops, vec);
    hybsol_system_solve_upper(sys, vec);
    return HYBSOL_SUCCESS;
}
