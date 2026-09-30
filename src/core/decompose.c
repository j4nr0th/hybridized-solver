/** @file core/decompose.c
 * Block LU decomposition, replay of the recorded operations and the solves
 * that build on them.
 */

#include "internal.h"

void hybsol_ops_append(hybsol_system_t *const sys, const hybsol_operation_t op)
{
    // The list is sized to its proven upper bound before any thread reaches
    // here, so there is nothing to grow and nothing to fail on. Within a pass
    // the recorded operations commute, so the order threads interleave them in
    // does not matter; across passes it is the loop order.
    //
    // `omp atomic capture` is deliberately not used: GCC does not treat its
    // structured-block form as a capture and silently loses updates. A relaxed
    // fetch-add is the same lock-free bump; the slot contents are published by
    // the release that ends the loop.
#ifdef __GNUC__
    const uint64_t pos = __atomic_fetch_add(&sys->n_ops, 1, __ATOMIC_RELAXED);
#else
    uint64_t pos;
#pragma omp critical(hybsol_ops_append)
    pos = sys->n_ops++;
#endif
    sys->ops[pos] = op;
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

static void find_diagonal(hybsol_system_t *const sys, const uint64_t idx, hybsol_row_entry_t **const out)
{
    hybsol_require_index(sys, idx);

    hybsol_row_t *const row = sys->rows + idx;
    uint64_t i_diag;
    // The lookup must run even with asserts compiled out, where CUTL_ASSERT
    // degrades to an assumption the optimizer may delete.
    const int has_diag = hybsol_row_find(row, idx, &i_diag);
    CUTL_ASSERT(has_diag, "Row %llu has no diagonal block; every row must contain one.", (unsigned long long)idx);

    *out = row->entries[i_diag];
}
/* ------------------------------------------------------------------------- */
/* The value-carrying half, instantiated once per precision                   */
/* ------------------------------------------------------------------------- */

/*
 * Everything that touches block values is written once in
 * ``decompose_numeric.inc`` and instantiated here. Both instantiations are
 * private: callers go through the spellings below, which pick the right one
 * for the system they were handed.
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
    CUTL_ASSERT(b != NULL && x != NULL, "The right-hand side and destination must not be NULL.");
    hybsol_require_precision(sys, HYBSOL_PRECISION_DOUBLE);

    return hybsol_f64_solve_diagonal(sys, idx, b, x);
}

hybsol_result_t hybsol_system_solve_diagonal_f32(hybsol_system_t *const sys, const uint64_t idx,
                                                 const hybsol_fmatrix_t *const b, const hybsol_fmatrix_t *const x)
{
    CUTL_ASSERT(b != NULL && x != NULL, "The right-hand side and destination must not be NULL.");
    hybsol_require_precision(sys, HYBSOL_PRECISION_SINGLE);

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
    CUTL_ASSERT(vec != NULL, "The solution vector must not be NULL.");

    hybsol_system_apply_operations(sys, sys->n_ops, sys->ops, vec);
    hybsol_system_solve_upper(sys, vec);
    return HYBSOL_SUCCESS;
}

/* ------------------------------------------------------------------------- */
/* The decomposition itself                                                   */
/* ------------------------------------------------------------------------- */

hybsol_result_t hybsol_system_decompose_with_workspace(hybsol_system_t *const sys, void *const workspace,
                                                       const size_t workspace_bytes, const uint64_t n_threads)
{
    if (sys->decomposed)
        return HYBSOL_ERROR_ALREADY_DECOMPOSED;
    if (!hybsol_system_is_valid(sys))
        return HYBSOL_ERROR_SYSTEM_INVALID;
    CUTL_ASSERT(workspace != NULL, "The workspace pointer must not be NULL.");

    const uint64_t threads = (uint64_t)hybsol_resolve_threads(n_threads);

    // Everything the parallel regions touch is in place before they start: the
    // operation list at its proven bound, and one bump region per thread.
    // Nothing inside a region calls the allocator from here on.
    const uint64_t bound = hybsol_operation_bound(sys->n);
    if (sys->ops_capacity < bound)
    {
        hybsol_operation_t *const ops = hybsol_grow(sys->allocator, sys->ops, (size_t)bound * sizeof(*ops));
        if (ops == NULL)
            return HYBSOL_ERROR_OUT_OF_MEMORY;
        sys->ops = ops;
        sys->ops_capacity = bound;
    }
    sys->n_ops = 0;

    hybsol_result_t res = hybsol_thread_allocs_open(sys, threads);
    if (res != HYBSOL_SUCCESS)
        return res;

    hybsol_workspace_bind((hybsol_workspace_t *)workspace, workspace, workspace_bytes, sys, threads);
    const hybsol_workspace_t *const ws = (const hybsol_workspace_t *)workspace;
    unsigned char *const base = (unsigned char *)workspace;
    target_row_t *const target_status = (target_row_t *)(base + ws->off_target_status);
    uint64_t *const ready = (uint64_t *)(base + ws->off_ready);
    hybsol_result_t *const results = (hybsol_result_t *)(base + ws->off_results);

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
            hybsol_ops_append(sys, (hybsol_operation_t){.type = HYBSOL_OPERATION_INVERT_DIAGONAL, .idx_row = i_row});
            hybsol_result_t res = hybsol_system_decompose_diagonal(sys, i_row);
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

    // Each pass eliminates the rows whose blocking row is finished; the rest
    // need another pass. A row always waits on a strictly smaller index, so
    // the passes make progress.
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
    shared(sys, ws, target_status, ready, results, n_ready) if (threads > 1) num_threads(threads)
        for (uint64_t k = 0; k < n_ready; ++k)
        {
            const uint64_t i_tgt = ready[k];
            target_row_t *const status = target_status + i_tgt;

            void *const scratch = hybsol_workspace_scratch(ws);
            hybsol_result_t res = hybsol_system_eliminate_row_scratch(sys, i_tgt, status->idx_src_needed, scratch);
            if (res == HYBSOL_SUCCESS)
                hybsol_ops_append(sys, (hybsol_operation_t){.type = HYBSOL_OPERATION_ELIMINATE,
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
                    // Diagonal is now first in the row: finished, usable as a
                    // source in the next pass.
                    res = hybsol_system_decompose_diagonal(sys, i_tgt);
                    if (res == HYBSOL_SUCCESS)
                        res = hybsol_system_apply_diagonal_inverse(sys, i_tgt);
                    if (res == HYBSOL_SUCCESS)
                        hybsol_ops_append(
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

    // The regions stay alive: their entries are the system's fill-in now and
    // are released with the system rather than individually.
    hybsol_thread_allocs_done(sys);

    if (shared_res != HYBSOL_SUCCESS)
    {
        sys->n_ops = 0;
        return shared_res;
    }
    if (n_tgt != 0)
        return HYBSOL_ERROR_INTERNAL;

    CUTL_ASSERT(sys->n_ops <= bound, "Recorded %llu operations but the bound is %llu!", (unsigned long long)sys->n_ops,
                (unsigned long long)bound);

    sys->decomposed = 1;
    return HYBSOL_SUCCESS;
}

hybsol_result_t hybsol_system_decompose(hybsol_system_t *const sys, const uint64_t n_threads)
{
    const size_t bytes = hybsol_workspace_bytes(sys, n_threads);
    if (bytes == 0)
        return HYBSOL_ERROR_OUT_OF_MEMORY;

    void *const workspace = hybsol_alloc(sys->allocator, bytes);
    if (workspace == NULL)
        return HYBSOL_ERROR_OUT_OF_MEMORY;

    const hybsol_result_t res = hybsol_system_decompose_with_workspace(sys, workspace, bytes, n_threads);
    hybsol_free(sys->allocator, workspace);
    return res;
}

uint64_t hybsol_system_operation_bound(const hybsol_system_t *const sys)
{
    CUTL_ASSERT(sys != NULL, "The system must not be NULL.");
    return hybsol_operation_bound(sys->n);
}
