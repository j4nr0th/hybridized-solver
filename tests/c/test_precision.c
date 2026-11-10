/** @file test_precision.c
 * The single-precision spelling: creation, assembly, readback and a small
 * end-to-end decomposition and solve.
 */

#include "test_util.h"

#include <hybsol/hybsol.h>

#include <string.h>

/** Walk the graph, lay out the destination and factorize it. */
static hybsol_decomposition_t *factorize(hybsol_system_t *const sys, const uint64_t n_threads)
{
    hybsol_elimination_t *graph = NULL;
    if (hybsol_elimination_create(sys, &graph, NULL) != HYBSOL_SUCCESS)
        return NULL;

    hybsol_decomposition_t *dec = NULL;
    if (hybsol_decomposition_create(sys, graph, &dec) != HYBSOL_SUCCESS)
    {
        hybsol_elimination_destroy(graph);
        return NULL;
    }
    hybsol_elimination_destroy(graph);

    if (hybsol_decomposition_factorize(dec, n_threads) != HYBSOL_SUCCESS)
    {
        hybsol_decomposition_destroy(dec);
        return NULL;
    }
    return dec;
}

#define MAX_DIM 6

/** Build a well-conditioned ``n_block x n_block`` block system in `precision`. */
static hybsol_system_t *build_system(const hybsol_precision_t precision, const uint64_t n_block, const uint64_t *sizes,
                                     double out[MAX_DIM][MAX_DIM])
{
    hybsol_system_t *sys = NULL;
    if (hybsol_system_create_with_precision(n_block, sizes, precision, &sys, &CUTL_STD_ALLOCATOR) != HYBSOL_SUCCESS)
    {
        return NULL;
    }

    uint64_t offsets[MAX_DIM + 1] = {0};
    for (uint64_t i = 0; i < n_block; ++i)
        offsets[i + 1] = offsets[i] + sizes[i];
    const uint64_t total = offsets[n_block];

    // Symmetric, diagonally dominant, so the LU has an easy run.
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
    CHECK_OK(hybsol_system_create_with_precision(2, sizes, HYBSOL_PRECISION_SINGLE, &sys, &CUTL_STD_ALLOCATOR));
    CHECK(hybsol_system_precision(sys) == HYBSOL_PRECISION_SINGLE);
    hybsol_system_destroy(sys);

    sys = NULL;
    CHECK_OK(hybsol_system_create_with_precision(2, sizes, HYBSOL_PRECISION_DOUBLE, &sys, &CUTL_STD_ALLOCATOR));
    CHECK(hybsol_system_precision(sys) == HYBSOL_PRECISION_DOUBLE);
    hybsol_system_destroy(sys);

    // The default spelling is the double one.
    hybsol_system_t *plain = NULL;
    CHECK_OK(hybsol_system_create(2, sizes, &plain, &CUTL_STD_ALLOCATOR));
    CHECK(hybsol_system_precision(plain) == HYBSOL_PRECISION_DOUBLE);
    hybsol_system_destroy(plain);
}

static void test_single_assembly_and_readback(void)
{
    hybsol_system_t *sys = NULL;
    const uint64_t sizes[2] = {2, 3};
    CHECK_OK(hybsol_system_create_with_precision(2, sizes, HYBSOL_PRECISION_SINGLE, &sys, &CUTL_STD_ALLOCATOR));

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
    CHECK_OK(hybsol_system_create_with_precision(2, sizes, HYBSOL_PRECISION_SINGLE, &sys, &CUTL_STD_ALLOCATOR));
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
    hybsol_decomposition_t *const dec = factorize(sys, 1);
    CHECK(dec != NULL);
    if (dec == NULL)
    {
        hybsol_system_destroy(sys);
        return;
    }

    // Solving takes and returns doubles whatever the storage is. The right-hand
    // side is the image of x_j = j + 1, so that is what the solve must return.
    double rhs_vec[5];
    for (uint64_t i = 0; i < 5; ++i)
    {
        rhs_vec[i] = 0.0;
        for (uint64_t j = 0; j < 5; ++j)
            rhs_vec[i] += full[i][j] * (double)(j + 1);
    }

    double vec[5];
    memcpy(vec, rhs_vec, sizeof(vec));

    CHECK_OK(hybsol_decomposition_solve(dec, vec, 1));

    // A float solve cannot do better than cond * 1e-7 or so.
    double exact[5];
    for (uint64_t i = 0; i < 5; ++i)
        exact[i] = (double)(i + 1);

    // Residual: A @ x against the right-hand side it was built from.
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
    const uint64_t n_ops = hybsol_decomposition_n_operations(dec);
    hybsol_operation_t *const ops = malloc(sizeof(*ops) * n_ops);
    CHECK(ops != NULL);
    if (ops != NULL)
    {
        uint64_t written = 0;
        CHECK_OK(hybsol_decomposition_operations(dec, ops, n_ops, &written));
        CHECK(written == n_ops);
        hybsol_decomposition_apply_operations(dec, written, ops, replay);
        free(ops);
    }
    hybsol_decomposition_solve_upper(dec, replay);
    for (uint64_t i = 0; i < 5; ++i)
        CHECK_NEAR(replay[i], vec[i], 1e-12);

    hybsol_decomposition_destroy(dec);
    hybsol_system_destroy(sys);
}

/** Walk the graph, lay the destination out in ``precision`` and factorize it. */
static hybsol_decomposition_t *factorize_in(const hybsol_system_t *const sys, const hybsol_precision_t precision,
                                            const uint64_t n_threads)
{
    hybsol_elimination_t *graph = NULL;
    if (hybsol_elimination_create(sys, &graph, NULL) != HYBSOL_SUCCESS)
        return NULL;

    hybsol_decomposition_t *dec = NULL;
    if (hybsol_decomposition_create_with_precision(sys, graph, precision, &dec) != HYBSOL_SUCCESS)
    {
        hybsol_elimination_destroy(graph);
        return NULL;
    }
    hybsol_elimination_destroy(graph);

    if (hybsol_decomposition_factorize(dec, n_threads) != HYBSOL_SUCCESS)
    {
        hybsol_decomposition_destroy(dec);
        return NULL;
    }
    return dec;
}

/** The image of ``x_j = j + 1`` under ``full``, which the solves must return. */
static void rhs_of(const double full[MAX_DIM][MAX_DIM], double out[5])
{
    for (uint64_t i = 0; i < 5; ++i)
    {
        out[i] = 0.0;
        for (uint64_t j = 0; j < 5; ++j)
            out[i] += full[i][j] * (double)(j + 1);
    }
}

/**
 * A double system can produce a decomposition of single factors. The factors
 * are exact for the values that went in, so the solve is as good as one from a
 * single system holding the same numbers, and no better.
 */
static void test_narrowed_decomposition_of_double_system(void)
{
    double full[MAX_DIM][MAX_DIM];
    hybsol_system_t *const sys = build_system(HYBSOL_PRECISION_DOUBLE, 2, (const uint64_t[2]){2, 3}, full);
    CHECK(sys != NULL);
    if (sys == NULL)
        return;

    hybsol_decomposition_t *const narrow = factorize_in(sys, HYBSOL_PRECISION_SINGLE, 1);
    CHECK(narrow != NULL);
    if (narrow == NULL)
    {
        hybsol_system_destroy(sys);
        return;
    }

    double rhs[5];
    rhs_of(full, rhs);
    double vec[5];
    memcpy(vec, rhs, sizeof(vec));
    CHECK_OK(hybsol_decomposition_solve(narrow, vec, 1));

    for (uint64_t i = 0; i < 5; ++i)
    {
        CHECK_NEAR(vec[i], (double)(i + 1), 1e-3);
        double residual = 0.0;
        for (uint64_t j = 0; j < 5; ++j)
            residual += full[i][j] * vec[j];
        CHECK_NEAR(residual, rhs[i], 1e-4 + 1e-4 * fabs(rhs[i]));
    }

    // The same values narrowed to single factors before the decomposition are
    // the same decomposition: both start from identical float blocks.
    hybsol_system_t *const single = build_system(HYBSOL_PRECISION_SINGLE, 2, (const uint64_t[2]){2, 3}, full);
    CHECK(single != NULL);
    if (single != NULL)
    {
        hybsol_decomposition_t *const dec = factorize_in(single, HYBSOL_PRECISION_SINGLE, 1);
        CHECK(dec != NULL);
        if (dec != NULL)
        {
            double other[5];
            memcpy(other, rhs, sizeof(other));
            CHECK_OK(hybsol_decomposition_solve(dec, other, 1));
            for (uint64_t i = 0; i < 5; ++i)
                CHECK_NEAR(vec[i], other[i], 1e-12 + 1e-6 * fabs(vec[i]));
            hybsol_decomposition_destroy(dec);
        }
        hybsol_system_destroy(single);
    }

    // A narrower factorization needs less scratch: the whole point of it.
    CHECK(hybsol_workspace_bytes(sys, HYBSOL_PRECISION_SINGLE, 1) <
          hybsol_workspace_bytes(sys, HYBSOL_PRECISION_DOUBLE, 1));

    hybsol_decomposition_destroy(narrow);
    hybsol_system_destroy(sys);
}

/**
 * And a single system can produce a decomposition of double factors: the
 * factorization is then exact, but the values it started from are still the
 * single ones the system rounded, so the solution cannot be better than those.
 */
static void test_widened_decomposition_of_single_system(void)
{
    double full[MAX_DIM][MAX_DIM];
    hybsol_system_t *const sys = build_system(HYBSOL_PRECISION_SINGLE, 2, (const uint64_t[2]){2, 3}, full);
    CHECK(sys != NULL);
    if (sys == NULL)
    {
        return;
    }

    hybsol_decomposition_t *const wide = factorize_in(sys, HYBSOL_PRECISION_DOUBLE, 1);
    CHECK(wide != NULL);
    if (wide == NULL)
    {
        hybsol_system_destroy(sys);
        return;
    }

    double rhs[5];
    rhs_of(full, rhs);
    double vec[5];
    memcpy(vec, rhs, sizeof(vec));
    CHECK_OK(hybsol_decomposition_solve(wide, vec, 1));

    // Bounded by how the system rounded its own blocks, roughly cond * 1e-7,
    // not by the factorization, which runs in double.
    for (uint64_t i = 0; i < 5; ++i)
    {
        CHECK_NEAR(vec[i], (double)(i + 1), 1e-6);
        double residual = 0.0;
        for (uint64_t j = 0; j < 5; ++j)
            residual += full[i][j] * vec[j];
        CHECK_NEAR(residual, rhs[i], 1e-6 + 1e-6 * fabs(rhs[i]));
    }

    hybsol_decomposition_destroy(wide);
    hybsol_system_destroy(sys);
}

/**
 * A single-precision factorization leaves its system untouched and agrees with
 * itself across thread counts, exactly as the double spelling does.
 */
static void test_single_system_survives_its_decomposition(void)
{
    const uint64_t thread_counts[] = {1, 2, 4};
    double full[MAX_DIM][MAX_DIM];
    hybsol_system_t *const sys = build_system(HYBSOL_PRECISION_SINGLE, 2, (const uint64_t[2]){2, 3}, full);
    CHECK(sys != NULL);
    if (sys == NULL)
    {
        return;
    }

    const uint64_t dim = hybsol_system_total_size(sys);
    float before[MAX_DIM * MAX_DIM] = {0};
    CHECK_OK(hybsol_system_to_dense_f32(sys, before));

    double rhs[5], baseline[5];
    for (uint64_t i = 0; i < 5; ++i)
    {
        rhs[i] = 0.0;
        for (uint64_t j = 0; j < 5; ++j)
            rhs[i] += full[i][j] * (double)(j + 1);
    }

    for (size_t t = 0; t < sizeof(thread_counts) / sizeof(thread_counts[0]); ++t)
    {
        hybsol_decomposition_t *const dec = factorize(sys, thread_counts[t]);
        CHECK(dec != NULL);
        if (dec == NULL)
        {
            continue;
        }

        float after[MAX_DIM * MAX_DIM] = {0};
        CHECK_OK(hybsol_system_to_dense_f32(sys, after));
        CHECK_MSG(memcmp(before, after, sizeof(float) * dim * dim) == 0,
                  "the single-precision factorization rewrote the system at %llu threads",
                  (unsigned long long)thread_counts[t]);

        double solution[5];
        memcpy(solution, rhs, sizeof(rhs));
        CHECK_OK(hybsol_decomposition_solve(dec, solution, thread_counts[t]));
        if (t == 0)
        {
            memcpy(baseline, solution, sizeof(baseline));
        }
        else
        {
            CHECK_MSG(memcmp(baseline, solution, sizeof(baseline)) == 0,
                      "thread count %llu produced a different single-precision solution",
                      (unsigned long long)thread_counts[t]);
        }

        for (uint64_t i = 0; i < 5; ++i)
            CHECK_NEAR(solution[i], (double)(i + 1), 1e-3);

        hybsol_decomposition_destroy(dec);
    }

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

    hybsol_decomposition_t *const dec = factorize(sys, 1);
    CHECK(dec != NULL);
    if (dec == NULL)
    {
        hybsol_system_destroy(sys);
        return;
    }

    double vec[5] = {0};
    for (uint64_t i = 0; i < 5; ++i)
        for (uint64_t j = 0; j < 5; ++j)
            vec[i] += full[i][j] * (double)(j + 1);
    CHECK_OK(hybsol_decomposition_solve(dec, vec, 1));

    for (uint64_t i = 0; i < 5; ++i)
        CHECK_NEAR(vec[i], (double)(i + 1), 1e-12);

    hybsol_decomposition_destroy(dec);
    hybsol_system_destroy(sys);
}

int main(void)
{
    test_create_precision();
    test_single_assembly_and_readback();
    test_copy_preserves_precision();
    test_single_decompose_and_solve();
    test_single_system_survives_its_decomposition();
    test_double_is_unchanged();
    test_narrowed_decomposition_of_double_system();
    test_widened_decomposition_of_single_system();
    return test_report("test_precision");
}
