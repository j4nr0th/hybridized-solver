/**
 * @file core/solve.c
 * Level-synchronous forward substitution, then a serial back-substitution.
 */

#include "internal.h"

/*
 * The forward substitution walks the same passes the factorization did. A row in pass `p` writes its
 * own slice and reads the slice of the row it is eliminated with, which the walk put in a strictly
 * earlier pass -- so every read is already final, no two rows in a pass write the same element, and
 * each row's slice sees the same operations in the same order at any thread count.
 */
static void forward_substitution(const hybsol_decomposition_t *const dec, double *const vec, const int threads)
{
    HYBSOL_MARK_USED(threads);

    for (uint64_t pass = 0; pass < dec->n_levels; ++pass)
    {
        const uint64_t from = dec->level_offset[pass];
        const uint64_t to = dec->level_offset[pass + 1];

#pragma omp parallel for schedule(static) default(none) shared(dec, vec, from, to, pass) if (threads > 1)              \
    num_threads(threads)
        for (uint64_t j = from; j < to; ++j)
        {
            const uint64_t row = dec->level_rows[j];
            const uint64_t step = dec->level_k[j];

            if (step < dec->row_n_elim[row])
                hybsol_decomposition_forward_eliminate(dec, row, step, vec);
            if (pass == dec->row_level[row])
                hybsol_decomposition_forward_solve_diagonal(dec, row, vec);
        }
    }
}

hybsol_result_t hybsol_decomposition_solve(const hybsol_decomposition_t *const dec, double *const vec,
                                           const uint64_t n_threads)
{
    CUTL_ASSERT(dec != NULL, "The decomposition must not be NULL.");
    CUTL_ASSERT(vec != NULL, "The solution vector must not be NULL.");
    CUTL_ASSERT(dec->factorized, "This decomposition has not been factorized.");

    // A backend's factors are not in this process's memory; it solves them.
    if (dec->backend != NULL)
        return dec->backend->solve(dec->backend_state, vec, n_threads);

    forward_substitution(dec, vec, hybsol_resolve_threads(n_threads));

    // Serial: row i needs every higher column it holds, a different dependency from the elimination's passes.
    hybsol_decomposition_back_substitute(dec, vec);
    return HYBSOL_SUCCESS;
}
