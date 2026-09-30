/** @file test_matrix.c
 * Dense matrix views, products and the (unpivoted) LU factorization.
 */

#include "test_util.h"

#include <hybsol/hybsol.h>

#include <string.h>

/** Deterministic pseudo-random double in ``[0, 1)``. */
static double rand01(uint64_t *const state)
{
    *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(*state >> 11) * (1.0 / 9007199254740992.0);
}

static void test_view(void)
{
    double storage[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const hybsol_matrix_t m = hybsol_matrix_view(2, 3, storage);

    CHECK(m.rows == 2);
    CHECK(m.cols == 3);
    CHECK(m.data[0] == 1.0 && m.data[5] == 6.0);
}

static void test_multiply(void)
{
    double a_storage[] = {1.0, 2.0, 3.0, 4.0};
    double b_storage[] = {5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    double out_storage[6] = {0};
    const hybsol_matrix_t a = hybsol_matrix_view(2, 2, a_storage);
    const hybsol_matrix_t b = hybsol_matrix_view(2, 3, b_storage);
    const hybsol_matrix_t out = hybsol_matrix_view(2, 3, out_storage);

    hybsol_matrix_multiply(&a, &b, &out);

    // [[1,2],[3,4]] @ [[5,6,7],[8,9,10]] = [[21,24,27],[47,54,61]]
    CHECK(out_storage[0] == 21.0 && out_storage[1] == 24.0 && out_storage[2] == 27.0);
    CHECK(out_storage[3] == 47.0 && out_storage[4] == 54.0 && out_storage[5] == 61.0);
}

static void test_subtract(void)
{
    double a_storage[] = {5.0, 6.0, 7.0, 8.0};
    double b_storage[] = {1.0, 1.0, 2.0, 2.0};
    hybsol_matrix_t a = hybsol_matrix_view(2, 2, a_storage);
    const hybsol_matrix_t b = hybsol_matrix_view(2, 2, b_storage);

    hybsol_matrix_subtract_inplace(&a, &b);
    CHECK(a_storage[0] == 4.0 && a_storage[1] == 5.0 && a_storage[2] == 5.0 && a_storage[3] == 6.0);
}

static void test_lu_roundtrip(void)
{
    uint64_t state = 42;
    for (uint64_t n = 1; n <= 12; ++n)
    {
        double a[12 * 12], lu[12 * 12], b[12], x[12], expected[12];

        for (uint64_t i = 0; i < n * n; ++i)
            a[i] = rand01(&state) - 0.5;
        // Shift the matrix away from singularity.
        for (uint64_t i = 0; i < n; ++i)
            a[i * n + i] += 2.0 * n;

        for (uint64_t i = 0; i < n; ++i)
        {
            b[i] = rand01(&state);
            expected[i] = b[i];
        }

        memcpy(lu, a, sizeof(double) * (size_t)(n * n));
        const hybsol_matrix_t m = hybsol_matrix_view(n, n, lu);
        CHECK_OK(hybsol_matrix_lu_decompose(&m));

        const hybsol_matrix_t rhs = hybsol_matrix_view(n, 1, b);
        const hybsol_matrix_t sol = hybsol_matrix_view(n, 1, x);
        CHECK_OK(hybsol_matrix_lu_solve(&m, &rhs, &sol));

        // The solution has to reproduce the right-hand side of the original.
        for (uint64_t i = 0; i < n; ++i)
        {
            double acc = 0.0;
            for (uint64_t j = 0; j < n; ++j)
                acc += a[i * n + j] * x[j];
            CHECK_NEAR(acc, expected[i], 1e-9);
        }
    }
}

static void test_lu_singular(void)
{
    double singular[] = {1.0, 2.0, 2.0, 4.0};
    hybsol_matrix_t m = hybsol_matrix_view(2, 2, singular);

    CHECK_RESULT(hybsol_matrix_lu_decompose(&m), HYBSOL_ERROR_SINGULAR);
}

static void test_lu_solve(void)
{
    double lu[] = {1.0, 0.0, 0.0, 1.0};
    double b[4] = {0};
    hybsol_matrix_t m = hybsol_matrix_view(2, 2, lu);
    hybsol_matrix_t b2 = hybsol_matrix_view(2, 2, b);
    hybsol_matrix_t x2 = hybsol_matrix_view(2, 2, b);

    CHECK_OK(hybsol_matrix_lu_decompose(&m));
    CHECK_OK(hybsol_matrix_lu_solve(&m, &b2, &x2));
}

int main(void)
{
    test_view();
    test_multiply();
    test_subtract();
    test_lu_roundtrip();
    test_lu_singular();
    test_lu_solve();
    return test_report("test_matrix");
}
