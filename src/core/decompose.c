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
    return sys->decomposed ? sys->ops : NULL;
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

/* ------------------------------------------------------------------------- */
/* The value-carrying half, instantiated once per precision                   */
/* ------------------------------------------------------------------------- */

/*
 * Everything that touches block values is written once in
 * ``decompose_numeric.inc`` and instantiated here. Both instantiations are
 * private to this file: callers go through the spellings below, which pick
 * the right one for the system they were handed.
 */
#define HYBSOL_SCALAR double
#define HYBSOL_ACC double
#define HYBSOL_VIEW hybsol_matrix_t
#define HYBSOL_KERNEL(name) hybsol_matrix_##name
#define HYBSOL_FN(name) hybsol_f64_##name
#define HYBSOL_NARROWED 0
#include "decompose_numeric.inc"

#define HYBSOL_SCALAR float
#define HYBSOL_ACC float
#define HYBSOL_VIEW hybsol_fmatrix_t
#define HYBSOL_KERNEL(name) hybsol_fmatrix_##name
#define HYBSOL_FN(name) hybsol_f32_##name
#define HYBSOL_NARROWED 1
#include "decompose_numeric.inc"

/* ------------------------------------------------------------------------- */
/* Public spellings                                                           */
/* ------------------------------------------------------------------------- */

hybsol_result_t hybsol_system_decompose_diagonal(hybsol_system_t *const sys, const uint64_t idx)
{
    if (sys->precision == HYBSOL_PRECISION_SINGLE)
        return hybsol_f32_decompose_diagonal(sys, idx);
    return hybsol_f64_decompose_diagonal(sys, idx);
}

hybsol_result_t hybsol_system_apply_diagonal_inverse(hybsol_system_t *const sys, const uint64_t idx)
{
    if (sys->precision == HYBSOL_PRECISION_SINGLE)
        return hybsol_f32_apply_diagonal_inverse(sys, idx);
    return hybsol_f64_apply_diagonal_inverse(sys, idx);
}

hybsol_result_t hybsol_system_solve_diagonal(hybsol_system_t *const sys, const uint64_t idx,
                                             const hybsol_matrix_t *const b, const hybsol_matrix_t *const x)
{
    if (b == NULL || x == NULL)
        return HYBSOL_ERROR_INVALID_ARGUMENT;
    if (hybsol_check_precision(sys, HYBSOL_PRECISION_DOUBLE) != HYBSOL_SUCCESS)
        return HYBSOL_ERROR_INVALID_ARGUMENT;

    return hybsol_f64_solve_diagonal(sys, idx, b, x);
}

hybsol_result_t hybsol_system_solve_diagonal_f32(hybsol_system_t *const sys, const uint64_t idx,
                                                 const hybsol_fmatrix_t *const b, const hybsol_fmatrix_t *const x)
{
    if (b == NULL || x == NULL)
        return HYBSOL_ERROR_INVALID_ARGUMENT;
    if (hybsol_check_precision(sys, HYBSOL_PRECISION_SINGLE) != HYBSOL_SUCCESS)
        return HYBSOL_ERROR_INVALID_ARGUMENT;

    return hybsol_f32_solve_diagonal(sys, idx, b, x);
}

void hybsol_system_apply_operations(const hybsol_system_t *const sys, const uint64_t n_ops,
                                    const hybsol_operation_t *const ops, double *const vec)
{
    // The vector stays double for both precisions; only the factors narrow.
    if (sys->precision == HYBSOL_PRECISION_SINGLE)
        hybsol_f32_apply_operations(sys, n_ops, ops, vec);
    else
        hybsol_f64_apply_operations(sys, n_ops, ops, vec);
}

void hybsol_system_solve_upper(const hybsol_system_t *const sys, double *const y)
{
    if (sys->precision == HYBSOL_PRECISION_SINGLE)
        hybsol_f32_solve_upper(sys, y);
    else
        hybsol_f64_solve_upper(sys, y);
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
    uint64_t *const ready = hybsol_alloc((size_t)sys->n * sizeof(*ready));
    hybsol_result_t *const results = hybsol_alloc((size_t)sys->n * sizeof(*results));
    if (target_status == NULL || ready == NULL || results == NULL)
    {
        hybsol_free(target_status);
        hybsol_free(ready);
        hybsol_free(results);
        return HYBSOL_ERROR_OUT_OF_MEMORY;
    }

    hybsol_result_t shared_res = HYBSOL_SUCCESS;
    uint64_t n_tgt = 0;

    // Rows that already start with their diagonal block can be finished right
    // away; everything else waits for the elimination passes below.
#pragma omp parallel for schedule(dynamic) reduction(+ : n_tgt) default(none)                                          \
    shared(sys, target_status, shared_res) if (threads > 1) num_threads(threads)
    for (uint64_t i_row = 0; i_row < sys->n; ++i_row)
    {
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
#pragma omp critical(hybsol_decompose_error)
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

    // Every pass eliminates the rows whose blocking row is finished. Rows that
    // are not done yet simply need another pass, and since the row a row waits
    // for always has a strictly smaller index, the passes always make progress.
    while (shared_res == HYBSOL_SUCCESS)
    {
        uint64_t n_ready = 0;
        for (uint64_t i = 0; i < sys->n; ++i)
        {
            target_row_t *const status = target_status + i;
            if (status->status == TARGET_FREE && target_status[status->idx_src_needed].status == TARGET_DONE)
            {
                status->status = TARGET_IN_USE;
                ready[n_ready++] = i;
            }
        }
        if (n_ready == 0)
            break;

#pragma omp parallel for schedule(dynamic) default(none)                                                               \
    shared(sys, target_status, ready, results, n_ready) if (threads > 1) num_threads(threads)
        for (uint64_t k = 0; k < n_ready; ++k)
        {
            const uint64_t i_tgt = ready[k];
            target_row_t *const status = target_status + i_tgt;

            hybsol_result_t res = hybsol_system_eliminate_row(sys, i_tgt, status->idx_src_needed);
            if (res == HYBSOL_SUCCESS)
                res = hybsol_ops_append(sys, (hybsol_operation_t){.type = HYBSOL_OPERATION_ELIMINATE,
                                                                  .idx_row = i_tgt,
                                                                  .idx_col = status->idx_src_needed});

            if (res == HYBSOL_SUCCESS)
            {
                const hybsol_row_t *const row = sys->rows + i_tgt;
                const uint64_t next = hybsol_row_find_geq(row, status->idx_src_needed + 1);
                if (next >= row->count)
                    res = HYBSOL_ERROR_INTERNAL;
                else if (row->entries[next]->col == i_tgt)
                {
                    // The diagonal is now first in the row, so the row is
                    // finished and can act as a source in the next pass.
                    res = hybsol_system_decompose_diagonal(sys, i_tgt);
                    if (res == HYBSOL_SUCCESS)
                        res = hybsol_system_apply_diagonal_inverse(sys, i_tgt);
                    if (res == HYBSOL_SUCCESS)
                        res = hybsol_ops_append(
                            sys, (hybsol_operation_t){.type = HYBSOL_OPERATION_INVERT_DIAGONAL, .idx_row = i_tgt});
                }
            }

            results[k] = res;
        }

        for (uint64_t k = 0; k < n_ready; ++k)
        {
            const uint64_t i_tgt = ready[k];
            if (results[k] != HYBSOL_SUCCESS)
            {
                shared_res = results[k];
                break;
            }

            target_row_t *const status = target_status + i_tgt;
            const hybsol_row_t *const row = sys->rows + i_tgt;
            const uint64_t next = hybsol_row_find_geq(row, status->idx_src_needed + 1);
            if (row->entries[next]->col == i_tgt)
            {
                status->status = TARGET_DONE;
                n_tgt -= 1;
            }
            else
            {
                status->idx_src_needed = row->entries[next]->col;
                status->status = TARGET_FREE;
            }
        }
    }

    hybsol_free(target_status);
    hybsol_free(ready);
    hybsol_free(results);

    if (shared_res != HYBSOL_SUCCESS)
    {
        sys->n_ops = 0;
        return shared_res;
    }
    if (n_tgt != 0)
        return HYBSOL_ERROR_INTERNAL;

    sys->decomposed = 1;
    return HYBSOL_SUCCESS;
}
