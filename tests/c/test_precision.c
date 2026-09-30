/** @file test_precision.c
 * The single-precision spelling: creation, assembly, readback and a small
 * end-to-end decomposition and solve.
 */

#include "test_util.h"

#include <hybsol/hybsol.h>

#include <string.h>

#define MAX_DIM 6

/** Build a well-conditioned ``n_block x n_block`` block system in `precision`. */
static hybsol_system_t *build_system(const hybsol_precision_t precision, const uint64_t n_block, const uint64_t *sizes,
                                     double out[MAX_DIM][MAX_DIM])
{
    hybsol_system_t *sys = NULL;
    if (hybsol_system_create_with_precision(n_block, sizes, precision, &sys) != HYBSOL_SUCCESS)
    {
        return NULL;
    }

    uint64_t offsets[MAX_DIM + 1] = {0};
    for (uint64_t i = 0; i < n_block; ++i)
        offsets[i + 1] = offsets[i] + sizes[i];
    const uint64_t total = offsets[n_block];

    // Symmetric, diagonally dominant, so the unpivoted LU has an easy run.
    uint64_t k = 0;
    for (uint64_t i = 0; i < total; ++i)
    {
        for (uint64_t j = 0; j < total; ++j)
            out[i][j] = (double)(k++ % 5) * 0.125 - 0.25;
        out[i][i] += (double)total + 1.0;
        for (uint64_t j = 0; j < i; ++j)
            out[j][i] = out[i][j];
    }

    double block[MAX_DIM * MAX_DIM];
    float block_f[MAX_DIM * MAX_DIM];
    for (uint64_t i = 0; i < n_block; ++i)
    {
        for (uint64_t j = 0; j < n_block; ++j)
        {
            for (uint64_t a = 0; a < sizes[i]; ++a)
            {
                for (uint64_t b = 0; b < sizes[j]; ++b)
                {
                    block[a * sizes[j] + b] = out[offsets[i] + a][offsets[j] + b];
                    block_f[a * sizes[j] + b] = (float)out[offsets[i] + a][offsets[j] + b];
                }
            }

            const hybsol_result_t res = precision == HYBSOL_PRECISION_SINGLE
                                            ? hybsol_system_add_block_f32(sys, i, j, sizes[i], sizes[j], block_f)
                                            : hybsol_system_add_block(sys, i, j, sizes[i], sizes[j], block);
            if (res != HYBSOL_SUCCESS)
            {
                hybsol_system_destroy(sys);
                return NULL;
            }
        }
    }

    return sys;
}

static void test_create_precision(void)
{
    const uint64_t sizes[2] = {2, 3};

    hybsol_system_t *sys = NULL;
    CHECK_OK(hybsol_system_create_with_precision(2, sizes, HYBSOL_PRECISION_SINGLE, &sys));
    CHECK(hybsol_system_precision(sys) == HYBSOL_PRECISION_SINGLE);
    hybsol_system_destroy(sys);

    sys = NULL;
    CHECK_OK(hybsol_system_create_with_precision(2, sizes, HYBSOL_PRECISION_DOUBLE, &sys));
    CHECK(hybsol_system_precision(sys) == HYBSOL_PRECISION_DOUBLE);
    hybsol_system_destroy(sys);

    // The default spelling is the double one.
    hybsol_system_t *plain = NULL;
    CHECK_OK(hybsol_system_create(2, sizes, &plain));
    CHECK(hybsol_system_precision(plain) == HYBSOL_PRECISION_DOUBLE);
    hybsol_system_destroy(plain);
}

static void test_single_assembly_and_readback(void)
{
    hybsol_system_t *sys = NULL;
    const uint64_t sizes[2] = {2, 3};
    CHECK_OK(hybsol_system_create_with_precision(2, sizes, HYBSOL_PRECISION_SINGLE, &sys));

    const float vals[6] = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f};
    CHECK_OK(hybsol_system_add_block_f32(sys, 0, 1, 2, 3, vals));

    hybsol_fmatrix_t block;
    CHECK_OK(hybsol_system_get_block_f32(sys, 0, 1, &block));
    CHECK(block.rows == 2 && block.cols == 3);
    for (uint64_t i = 0; i < 6; ++i)
        CHECK(block.data[i] == vals[i]);

    // In-place storage hands back a zeroed float view and keeps it stable.
    hybsol_fmatrix_t view;
    CHECK_OK(hybsol_system_block_storage_f32(sys, 1, 1, &view));
    CHECK(view.rows == 3 && view.cols == 3);
    for (uint64_t i = 0; i < 9; ++i)
        CHECK(view.data[i] == 0.0f);
    for (uint64_t i = 0; i < 9; ++i)
        view.data[i] = (float)i + 0.5f;
    hybsol_fmatrix_t again;
    CHECK_OK(hybsol_system_block_storage_f32(sys, 1, 1, &again));
    CHECK(again.data == view.data);
    CHECK(again.data[4] == 4.5f);

    float dense[MAX_DIM * MAX_DIM] = {0};
    CHECK_OK(hybsol_system_to_dense_f32(sys, dense));
    // Block (0, 1) sits at rows 0..1, columns 2..4 of the 5 x 5 dense matrix.
    CHECK(dense[0 * 5 + 2] == 1.0f && dense[1 * 5 + 4] == -6.0f);
    CHECK(dense[2 * 5 + 2] == 0.5f && dense[4 * 5 + 4] == 8.5f);

    hybsol_system_destroy(sys);
}

static void test_copy_preserves_precision(void)
{
    hybsol_system_t *sys = NULL;
    const uint64_t sizes[2] = {2, 2};
    const float vals[4] = {4.0f, 1.0f, 1.0f, 4.0f};
    CHECK_OK(hybsol_system_create_with_precision(2, sizes, HYBSOL_PRECISION_SINGLE, &sys));
    CHECK_OK(hybsol_system_add_block_f32(sys, 0, 0, 2, 2, vals));
    CHECK_OK(hybsol_system_add_block_f32(sys, 1, 1, 2, 2, vals));

    hybsol_system_t *copy = NULL;
    CHECK_OK(hybsol_system_copy(sys, &copy));
    CHECK(hybsol_system_precision(copy) == HYBSOL_PRECISION_SINGLE);

    hybsol_fmatrix_t block;
    CHECK_OK(hybsol_system_get_block_f32(copy, 0, 0, &block));
    CHECK(block.data[0] == 4.0f && block.data[3] == 4.0f);

    hybsol_system_destroy(copy);
    hybsol_system_destroy(sys);
}

static void test_single_decompose_and_solve(void)
{
    double full[MAX_DIM][MAX_DIM];
    hybsol_system_t *sys = build_system(HYBSOL_PRECISION_SINGLE, 2, (const uint64_t[2]){2, 3}, full);
    CHECK(sys != NULL);
    if (sys == NULL)
    {
        return;
    }

    CHECK(hybsol_system_is_valid(sys));
    CHECK_OK(hybsol_system_decompose(sys, 1));

    // Solving takes and returns doubles whatever the storage is. The right
    // hand side is the image of x_j = j + 1, so that is what the solve has to
    // come back with.
    double rhs_vec[5];
    for (uint64_t i = 0; i < 5; ++i)
    {
        rhs_vec[i] = 0.0;
        for (uint64_t j = 0; j < 5; ++j)
            rhs_vec[i] += full[i][j] * (double)(j + 1);
    }

    double vec[5];
    memcpy(vec, rhs_vec, sizeof(vec));

    CHECK_OK(hybsol_system_solve(sys, vec));

    // A float solve cannot do better than cond * 1e-7 or so; ask only for
    // that much before comparing against the exact solution.
    double exact[5];
    for (uint64_t i = 0; i < 5; ++i)
        exact[i] = (double)(i + 1);

    // `vec` currently holds x; check A @ x against the right-hand side it
    // was built from by reconstructing it.
    double residual[5] = {0};
    for (uint64_t i = 0; i < 5; ++i)
        for (uint64_t j = 0; j < 5; ++j)
            residual[i] += full[i][j] * vec[j];

    for (uint64_t i = 0; i < 5; ++i)
    {
        double expected = 0;
        for (uint64_t j = 0; j < 5; ++j)
            expected += full[i][j] * (double)(j + 1);
        CHECK_NEAR(residual[i], expected, 1e-4 + 1e-4 * fabs(expected));
        CHECK_NEAR(vec[i], exact[i], 1e-3);
    }

    // The recorded operations are replayable on the double vector directly.
    double replay[5];
    for (uint64_t i = 0; i < 5; ++i)
        replay[i] = rhs_vec[i];
    hybsol_system_apply_operations(sys, hybsol_system_n_operations(sys), hybsol_system_operations(sys), replay);
    hybsol_system_solve_upper(sys, replay);
    for (uint64_t i = 0; i < 5; ++i)
        CHECK_NEAR(replay[i], vec[i], 1e-12);

    // The diagonal solve has a float spelling too.
    hybsol_system_t *diag = NULL;
    const uint64_t one[1] = {2};
    const float entries[4] = {4.0f, 0.0f, 0.0f, 4.0f};
    CHECK_OK(hybsol_system_create_with_precision(1, one, HYBSOL_PRECISION_SINGLE, &diag));
    CHECK_OK(hybsol_system_add_block_f32(diag, 0, 0, 2, 2, entries));
    CHECK_OK(hybsol_system_decompose_diagonal(diag, 0));

    float rhs_f[2] = {8.0f, 4.0f};
    float x_f[2] = {0.0f, 0.0f};
    const hybsol_fmatrix_t b = hybsol_fmatrix_view(2, 1, rhs_f);
    const hybsol_fmatrix_t x = hybsol_fmatrix_view(2, 1, x_f);
    CHECK_OK(hybsol_system_solve_diagonal_f32(diag, 0, &b, &x));
    CHECK_NEAR(x_f[0], 2.0f, 1e-6);
    CHECK_NEAR(x_f[1], 1.0f, 1e-6);

    hybsol_system_destroy(diag);
    hybsol_system_destroy(sys);
}

/** The double spelling must not have moved: same system, same bytes out. */
static void test_double_is_unchanged(void)
{
    double full[MAX_DIM][MAX_DIM];
    hybsol_system_t *sys = build_system(HYBSOL_PRECISION_DOUBLE, 2, (const uint64_t[2]){2, 3}, full);
    CHECK(sys != NULL);
    if (sys == NULL)
    {
        return;
    }

    CHECK_OK(hybsol_system_decompose(sys, 1));

    double vec[5] = {0};
    for (uint64_t i = 0; i < 5; ++i)
        for (uint64_t j = 0; j < 5; ++j)
            vec[i] += full[i][j] * (double)(j + 1);
    CHECK_OK(hybsol_system_solve(sys, vec));

    for (uint64_t i = 0; i < 5; ++i)
        CHECK_NEAR(vec[i], (double)(i + 1), 1e-12);

    hybsol_system_destroy(sys);
}

int main(void)
{
    test_create_precision();
    test_single_assembly_and_readback();
    test_copy_preserves_precision();
    test_single_decompose_and_solve();
    test_double_is_unchanged();
    return test_report("test_precision");
}
