/**
 * @file core/decomposition_numeric.c
 * The per-precision instantiations of the decomposition's kernels, and the
 * dispatchers that pick between them.
 *
 * Kept apart from the code that schedules the work so the arithmetic has one
 * home: the factorization and the solve both reach it through these.
 */

#include "internal.h"

/*
 * Everything that touches block values is written once in
 * ``decomposition_numeric.inc`` and instantiated here. Both instantiations are
 * private: callers go through the dispatchers below, which pick the right one
 * for the decomposition they were handed.
 */
#define HYBSOL_SCALAR double
#define HYBSOL_ACC double
#define HYBSOL_VIEW hybsol_matrix_t
#define HYBSOL_KERNEL(name) hybsol_matrix_##name
#define HYBSOL_FN(name) hybsol_f64_##name
#define HYBSOL_NARROWED 0
#include "decomposition_numeric.inc"

#define HYBSOL_SCALAR float
#define HYBSOL_ACC float
#define HYBSOL_VIEW hybsol_fmatrix_t
#define HYBSOL_KERNEL(name) hybsol_fmatrix_##name
#define HYBSOL_FN(name) hybsol_f32_##name
#define HYBSOL_NARROWED 1
#include "decomposition_numeric.inc"

/* ------------------------------------------------------------------------- */
/* Dispatchers                                                                */
/* ------------------------------------------------------------------------- */

hybsol_result_t hybsol_decomposition_diagonal_lu(hybsol_decomposition_t *dec, const uint64_t idx)
{
    if (dec->precision == HYBSOL_PRECISION_SINGLE)
        return hybsol_f32_factorize_diagonal(dec, idx);
    return hybsol_f64_factorize_diagonal(dec, idx);
}

void hybsol_decomposition_diagonal_inverse(hybsol_decomposition_t *dec, const uint64_t idx)
{
    if (dec->precision == HYBSOL_PRECISION_SINGLE)
        hybsol_f32_invert_diagonal_into_row(dec, idx);
    else
        hybsol_f64_invert_diagonal_into_row(dec, idx);
}

void hybsol_decomposition_eliminate(hybsol_decomposition_t *dec, const uint64_t row_tgt, const uint64_t k,
                                    void *scratch)
{
    if (dec->precision == HYBSOL_PRECISION_SINGLE)
        hybsol_f32_eliminate_step(dec, row_tgt, k, (float *)scratch);
    else
        hybsol_f64_eliminate_step(dec, row_tgt, k, (double *)scratch);
}

void hybsol_decomposition_forward_eliminate(const hybsol_decomposition_t *dec, const uint64_t row, const uint64_t k,
                                            double *vec)
{
    if (dec->precision == HYBSOL_PRECISION_SINGLE)
        hybsol_f32_forward_step(dec, row, k, vec);
    else
        hybsol_f64_forward_step(dec, row, k, vec);
}

void hybsol_decomposition_forward_solve_diagonal(const hybsol_decomposition_t *dec, const uint64_t row, double *vec)
{
    if (dec->precision == HYBSOL_PRECISION_SINGLE)
        hybsol_f32_forward_diagonal(dec, row, vec);
    else
        hybsol_f64_forward_diagonal(dec, row, vec);
}

void hybsol_decomposition_replay(const hybsol_decomposition_t *dec, const uint64_t n_ops, const hybsol_operation_t *ops,
                                 double *vec)
{
    // The vector stays double for both precisions; only the factors narrow.
    if (dec->precision == HYBSOL_PRECISION_SINGLE)
        hybsol_f32_apply_operations(dec, n_ops, ops, vec);
    else
        hybsol_f64_apply_operations(dec, n_ops, ops, vec);
}

void hybsol_decomposition_back_substitute(const hybsol_decomposition_t *dec, double *y)
{
    if (dec->precision == HYBSOL_PRECISION_SINGLE)
        hybsol_f32_solve_upper(dec, y);
    else
        hybsol_f64_solve_upper(dec, y);
}
