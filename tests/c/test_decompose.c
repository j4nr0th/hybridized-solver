/** @file test_decompose.c
 * Block LU decomposition and solves, checked against an in-test reference
 * Gaussian elimination with partial pivoting.
 */

#include "test_util.h"

#include <hybsol/hybsol.h>

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#define MAX_BLOCKS 8
#define MAX_SIZE 3
#define MAX_ENTRIES 64
#define MAX_DIM (MAX_BLOCKS * MAX_SIZE)

typedef struct
{
    uint64_t state;
} rng_t;

static double rng_next(rng_t *const r)
{
    r->state = r->state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(r->state >> 11) * (1.0 / 9007199254740992.0);
}

static uint64_t rng_below(rng_t *const r, const uint64_t bound)
{
    return (uint64_t)(rng_next(r) * (double)bound) % bound;
}

/** A random, structurally valid system together with its dense matrix. */
typedef struct
{
    hybsol_system_t *sys;
    uint64_t n_blocks;
    uint64_t dim;
    double dense[MAX_DIM][MAX_DIM];
} random_system_t;

static void random_system_destroy(random_system_t *const s)
{
    hybsol_system_destroy(s->sys);
    s->sys = NULL;
}

/** Build ``s``; the dense matrix is filled first, then assembled as blocks. */
static void build_random_system(rng_t *const r, random_system_t *const s)
{
    s->n_blocks = 1 + rng_below(r, MAX_BLOCKS);
    uint64_t sizes[MAX_BLOCKS], offsets[MAX_BLOCKS + 1] = {0};
    bool present[MAX_BLOCKS][MAX_BLOCKS] = {{false}};
    for (uint64_t i = 0; i < s->n_blocks; ++i)
    {
        sizes[i] = 1 + rng_below(r, MAX_SIZE);
        offsets[i + 1] = offsets[i] + sizes[i];
    }
    s->dim = offsets[s->n_blocks];

    memset(s->dense, 0, sizeof(s->dense));
    for (uint64_t i = 0; i < s->n_blocks; ++i)
        present[i][i] = true;

    for (uint64_t i = 0; i < s->n_blocks; ++i)
    {
        for (uint64_t j = i + 1; j < s->n_blocks; ++j)
        {
            if (rng_next(r) < 0.4)
                continue;

            present[i][j] = present[j][i] = true;
            for (uint64_t a = 0; a < sizes[i]; ++a)
            {
                for (uint64_t b = 0; b < sizes[j]; ++b)
                {
                    const double value = rng_next(r) - 0.5;
                    s->dense[offsets[i] + a][offsets[j] + b] = value;
                    s->dense[offsets[j] + b][offsets[i] + a] = value;
                }
            }
        }
    }

    // Strict diagonal dominance keeps the unpivoted factorization stable enough for the tolerances below.
    for (uint64_t i = 0; i < s->dim; ++i)
    {
        double sum = 1.0;
        for (uint64_t j = 0; j < s->dim; ++j)
            sum += fabs(s->dense[i][j]);
        s->dense[i][i] += sum;
    }

    hybsol_system_t *sys = NULL;
    CHECK_OK(hybsol_system_create(s->n_blocks, sizes, &sys, &CUTL_STD_ALLOCATOR));
    s->sys = sys;

    for (uint64_t i = 0; i < s->n_blocks; ++i)
    {
        for (uint64_t j = 0; j < s->n_blocks; ++j)
        {
            if (!present[i][j])
                continue;

            double block[MAX_SIZE * MAX_SIZE];
            for (uint64_t a = 0; a < sizes[i]; ++a)
            {
                for (uint64_t b = 0; b < sizes[j]; ++b)
                    block[a * sizes[j] + b] = s->dense[offsets[i] + a][offsets[j] + b];
            }
            CHECK_OK(hybsol_system_add_block(sys, i, j, sizes[i], sizes[j], block));
        }
    }

    CHECK(hybsol_system_is_valid(sys));
}

/** Gaussian elimination with partial pivoting; overwrites ``a`` and ``b``. */
static void reference_solve(const uint64_t n, double a[MAX_DIM][MAX_DIM], double *const b)
{
    for (uint64_t k = 0; k < n; ++k)
    {
        uint64_t pivot = k;
        for (uint64_t i = k + 1; i < n; ++i)
        {
            if (fabs(a[i][k]) > fabs(a[pivot][k]))
                pivot = i;
        }
        if (pivot != k)
        {
            for (uint64_t j = 0; j < n; ++j)
            {
                const double tmp = a[k][j];
                a[k][j] = a[pivot][j];
                a[pivot][j] = tmp;
            }
            const double tmp = b[k];
            b[k] = b[pivot];
            b[pivot] = tmp;
        }

        for (uint64_t i = k + 1; i < n; ++i)
        {
            const double factor = a[i][k] / a[k][k];
            for (uint64_t j = k; j < n; ++j)
                a[i][j] -= factor * a[k][j];
            b[i] -= factor * b[k];
        }
    }

    for (uint64_t i = n; i > 0; --i)
    {
        double acc = b[i - 1];
        for (uint64_t j = i; j < n; ++j)
            acc -= a[i - 1][j] * b[j];
        b[i - 1] = acc / a[i - 1][i - 1];
    }
}

static double max_abs_difference(const uint64_t n, const double *const a, const double *const b)
{
    double worst = 0.0;
    for (uint64_t i = 0; i < n; ++i)
    {
        const double d = fabs(a[i] - b[i]);
        if (d > worst)
            worst = d;
    }
    return worst;
}

static void test_solve_matches_reference(void)
{
    rng_t r = {.state = 2024};
    for (int trial = 0; trial < 30; ++trial)
    {
        random_system_t s = {0};
        build_random_system(&r, &s);

        double a[MAX_DIM][MAX_DIM], b[MAX_DIM], expected[MAX_DIM];
        memcpy(a, s.dense, sizeof(a));
        for (uint64_t i = 0; i < s.dim; ++i)
            expected[i] = b[i] = rng_next(&r);

        reference_solve(s.dim, a, expected);

        hybsol_system_t *serial = NULL;
        hybsol_system_t *parallel = NULL;
        CHECK_OK(hybsol_system_copy(s.sys, &serial));
        CHECK_OK(hybsol_system_copy(s.sys, &parallel));

        CHECK_OK(hybsol_system_decompose(serial, 1));
        CHECK_OK(hybsol_system_decompose(parallel, 0));
        CHECK(hybsol_system_is_decomposed(serial));

        double solution[MAX_DIM];
        memcpy(solution, b, sizeof(double) * s.dim);
        CHECK_OK(hybsol_system_solve(serial, solution));
        CHECK(max_abs_difference(s.dim, solution, expected) < 1e-8);

        memcpy(solution, b, sizeof(double) * s.dim);
        CHECK_OK(hybsol_system_solve(parallel, solution));
        CHECK(max_abs_difference(s.dim, solution, expected) < 1e-8);

        // Repeated decomposition is refused.
        CHECK_RESULT(hybsol_system_decompose(serial, 1), HYBSOL_ERROR_ALREADY_DECOMPOSED);

        hybsol_system_destroy(parallel);
        hybsol_system_destroy(serial);
        random_system_destroy(&s);
    }
}

static void test_rejects_invalid_and_singular(void)
{
    const uint64_t sizes[] = {2, 2};

    // Asymmetric pattern.
    hybsol_system_t *sys = NULL;
    CHECK_OK(hybsol_system_create(2, sizes, &sys, &CUTL_STD_ALLOCATOR));
    double block[4] = {2.0, 1.0, 1.0, 2.0};
    CHECK_OK(hybsol_system_add_block(sys, 0, 0, 2, 2, block));
    CHECK_OK(hybsol_system_add_block(sys, 0, 1, 2, 2, block));
    CHECK(!hybsol_system_is_valid(sys));
    CHECK_RESULT(hybsol_system_decompose(sys, 1), HYBSOL_ERROR_SYSTEM_INVALID);
    hybsol_system_destroy(sys);

    // Exactly singular diagonal block.
    CHECK_OK(hybsol_system_create(1, sizes, &sys, &CUTL_STD_ALLOCATOR));
    double zero[4] = {0};
    CHECK_OK(hybsol_system_add_block(sys, 0, 0, 2, 2, zero));
    CHECK(hybsol_system_is_valid(sys));
    CHECK_RESULT(hybsol_system_decompose(sys, 1), HYBSOL_ERROR_SINGULAR);
    CHECK(!hybsol_system_is_decomposed(sys));
    CHECK(hybsol_system_n_operations(sys) == 0);
    CHECK(hybsol_system_operations(sys) == NULL);
    hybsol_system_destroy(sys);
}

static void test_operations_replay(void)
{
    rng_t r = {.state = 7};
    random_system_t s = {0};
    build_random_system(&r, &s);

    CHECK_OK(hybsol_system_decompose(s.sys, 1));

    const hybsol_operation_t *const ops = hybsol_system_operations(s.sys);
    const uint64_t n_ops = hybsol_system_n_operations(s.sys);
    CHECK(ops != NULL);
    CHECK(n_ops > 0);

    // Every index the operations mention has to be inside the system.
    for (uint64_t i = 0; i < n_ops; ++i)
    {
        CHECK(ops[i].idx_row < s.n_blocks);
        switch (ops[i].type)
        {
        case HYBSOL_OPERATION_INVERT_DIAGONAL:
            break;
        case HYBSOL_OPERATION_ELIMINATE:
            CHECK(ops[i].idx_col < ops[i].idx_row);
            break;
        default:
            CHECK_MSG(0, "operation %llu has unknown type %d", (unsigned long long)i, (int)ops[i].type);
        }
    }

    // Replaying them by hand has to reproduce `hybsol_system_solve`.
    double by_hand[MAX_DIM], via_solve[MAX_DIM];
    for (uint64_t i = 0; i < s.dim; ++i)
        by_hand[i] = via_solve[i] = rng_next(&r);

    hybsol_system_apply_operations(s.sys, n_ops, ops, by_hand);
    hybsol_system_solve_upper(s.sys, by_hand);

    hybsol_system_t *helper = NULL;
    CHECK_OK(hybsol_system_copy(s.sys, &helper));
    CHECK_OK(hybsol_system_solve(helper, via_solve));
    hybsol_system_destroy(helper);

    CHECK(max_abs_difference(s.dim, by_hand, via_solve) < 1e-12);
    random_system_destroy(&s);
}

static void test_diagonal_helpers(void)
{
    const uint64_t sizes[] = {2};
    hybsol_system_t *sys = NULL;
    CHECK_OK(hybsol_system_create(1, sizes, &sys, &CUTL_STD_ALLOCATOR));

    double values[4] = {4.0, 1.0, 1.0, 3.0};
    CHECK_OK(hybsol_system_add_block(sys, 0, 0, 2, 2, values));

    // Solving before decomposing the diagonal must be refused.
    double rhs[2] = {1.0, 2.0};
    const hybsol_matrix_t b = hybsol_matrix_view(2, 1, rhs);
    const hybsol_matrix_t x = hybsol_matrix_view(2, 1, rhs);
    CHECK_RESULT(hybsol_system_solve_diagonal(sys, 0, &b, &x), HYBSOL_ERROR_NOT_DECOMPOSED);

    CHECK_OK(hybsol_system_decompose_diagonal(sys, 0));
    CHECK_OK(hybsol_system_solve_diagonal(sys, 0, &b, &x));

    // [[4,1],[1,3]]^-1 @ [1,2] = [1/11, 7/11]
    CHECK_NEAR(x.data[0], 1.0 / 11.0, 1e-12);
    CHECK_NEAR(x.data[1], 7.0 / 11.0, 1e-12);

    hybsol_system_destroy(sys);
}

int main(void)
{
    test_solve_matches_reference();
    test_rejects_invalid_and_singular();
    test_operations_replay();
    test_diagonal_helpers();
    return test_report("test_decompose");
}
