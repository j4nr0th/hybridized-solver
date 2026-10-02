/** @file test_decomposition.c
 * The destination a decomposition writes into, and the solves that read it
 * back, checked against an in-test reference Gaussian elimination with partial
 * pivoting.
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

    // Strict diagonal dominance keeps the factorization stable enough for the tolerances below.
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

/** Build a decomposition of ``sys`` and factorize it; ``graph`` may be NULL. */
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

    // The decomposition copied what it needs, so the graph is no longer needed.
    hybsol_elimination_destroy(graph);

    if (hybsol_decomposition_factorize(dec, n_threads) != HYBSOL_SUCCESS)
    {
        hybsol_decomposition_destroy(dec);
        return NULL;
    }
    return dec;
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

        const uint64_t thread_counts[] = {1, 0, 2, 4};
        for (size_t t = 0; t < sizeof(thread_counts) / sizeof(thread_counts[0]); ++t)
        {
            hybsol_decomposition_t *const dec = factorize(s.sys, thread_counts[t]);
            CHECK_MSG(dec != NULL, "trial %d: factorizing at %llu threads failed", trial,
                      (unsigned long long)thread_counts[t]);
            if (dec == NULL)
                continue;
            CHECK(hybsol_decomposition_is_factorized(dec));

            double solution[MAX_DIM];
            memcpy(solution, b, sizeof(double) * s.dim);
            CHECK_OK(hybsol_decomposition_solve(dec, solution, thread_counts[t]));
            CHECK_MSG(max_abs_difference(s.dim, solution, expected) < 1e-8, "trial %d at %llu threads is off by %g",
                      trial, (unsigned long long)thread_counts[t], max_abs_difference(s.dim, solution, expected));

            // Factorizing again is a precondition violation, not a result.
            hybsol_decomposition_destroy(dec);
        }

        random_system_destroy(&s);
    }
}

/** The system a decomposition was built from must come out of it untouched. */
static void test_system_survives_its_decomposition(void)
{
    rng_t r = {.state = 991};
    for (int trial = 0; trial < 20; ++trial)
    {
        random_system_t s = {0};
        build_random_system(&r, &s);

        const uint64_t dim = s.dim;
        double before[MAX_DIM * MAX_DIM], after[MAX_DIM * MAX_DIM];
        hybsol_system_to_dense(s.sys, before);

        hybsol_decomposition_t *const dec = factorize(s.sys, 2);
        CHECK(dec != NULL);
        if (dec != NULL)
            hybsol_decomposition_destroy(dec);

        hybsol_system_to_dense(s.sys, after);
        CHECK_MSG(memcmp(before, after, sizeof(double) * dim * dim) == 0,
                  "trial %d: the factorization rewrote the system", trial);

        // The pattern is untouched too, so the system still validates and can be
        // decomposed again under a different order.
        CHECK(hybsol_system_is_valid(s.sys));
        uint64_t columns[MAX_ENTRIES];
        for (uint64_t i = 0; i < s.n_blocks; ++i)
        {
            const uint64_t count = hybsol_system_row_count(s.sys, i);
            CHECK(count <= MAX_ENTRIES);
            if (count > MAX_ENTRIES)
                break;
            uint64_t written = 0;
            CHECK_OK(hybsol_system_row_indices(s.sys, i, columns, MAX_ENTRIES, &written));
            CHECK(written == count);
        }

        // And a second decomposition of the same system agrees with the first.
        double *const rhs = malloc(sizeof(double) * dim);
        double *const first = malloc(sizeof(double) * dim);
        double *const second = malloc(sizeof(double) * dim);
        CHECK(rhs != NULL && first != NULL && second != NULL);
        if (rhs != NULL && first != NULL && second != NULL)
        {
            for (uint64_t i = 0; i < dim; ++i)
                rhs[i] = rng_next(&r);
            memcpy(first, rhs, sizeof(double) * dim);
            memcpy(second, rhs, sizeof(double) * dim);

            hybsol_decomposition_t *const a = factorize(s.sys, 1);
            hybsol_decomposition_t *const b = factorize(s.sys, 3);
            CHECK(a != NULL && b != NULL);
            if (a != NULL && b != NULL)
            {
                CHECK_OK(hybsol_decomposition_solve(a, first, 1));
                CHECK_OK(hybsol_decomposition_solve(b, second, 3));
                CHECK_MSG(memcmp(first, second, sizeof(double) * dim) == 0,
                          "trial %d: thread count changed the solution", trial);
                CHECK_MSG(max_abs_difference(dim, first, second) < 1e-12, "trial %d: solutions differ by %g", trial,
                          max_abs_difference(dim, first, second));
            }
            hybsol_decomposition_destroy(a);
            hybsol_decomposition_destroy(b);
        }
        free(rhs);
        free(first);
        free(second);
        random_system_destroy(&s);
    }
}

static void test_rejects_invalid_and_singular(void)
{
    const uint64_t sizes[] = {2, 2};

    // Asymmetric pattern.
    hybsol_system_t *sys = NULL;
    hybsol_elimination_t *graph = NULL;
    CHECK_OK(hybsol_system_create(2, sizes, &sys, &CUTL_STD_ALLOCATOR));
    double block[4] = {2.0, 1.0, 1.0, 2.0};
    CHECK_OK(hybsol_system_add_block(sys, 0, 0, 2, 2, block));
    CHECK_OK(hybsol_system_add_block(sys, 0, 1, 2, 2, block));
    // The walk asserts structural validity rather than reporting it, so
    // hybsol_system_is_valid is what a caller checks to get a diagnosis.
    CHECK(!hybsol_system_is_valid(sys));
    hybsol_system_destroy(sys);

    // A row with no diagonal block.
    CHECK_OK(hybsol_system_create(2, sizes, &sys, &CUTL_STD_ALLOCATOR));
    CHECK_OK(hybsol_system_add_block(sys, 0, 0, 2, 2, block));
    CHECK(!hybsol_system_is_valid(sys));
    hybsol_system_destroy(sys);

    // Identically zero diagonal block: no pivot, so an unusable order.
    CHECK_OK(hybsol_system_create(1, sizes, &sys, &CUTL_STD_ALLOCATOR));
    double zero[4] = {0};
    CHECK_OK(hybsol_system_add_block(sys, 0, 0, 2, 2, zero));
    CHECK(hybsol_system_is_valid(sys));
    CHECK_RESULT(hybsol_elimination_create(sys, &graph, NULL), HYBSOL_ERROR_INVALID_ORDERING);
    CHECK(graph == NULL);
    CHECK(factorize(sys, 1) == NULL);
    hybsol_system_destroy(sys);

    // Nonzero but singular (rank one): a genuine numerical singularity, which
    // only the factorization can find.
    CHECK_OK(hybsol_system_create(1, sizes, &sys, &CUTL_STD_ALLOCATOR));
    double rank_one[4] = {1.0, 2.0, 2.0, 4.0};
    CHECK_OK(hybsol_system_add_block(sys, 0, 0, 2, 2, rank_one));
    CHECK(hybsol_system_is_valid(sys));
    CHECK_OK(hybsol_elimination_create(sys, &graph, NULL));
    CHECK(hybsol_elimination_failing_block(graph) == UINT64_MAX);

    hybsol_decomposition_t *dec = NULL;
    CHECK_OK(hybsol_decomposition_create(sys, graph, &dec));
    CHECK_RESULT(hybsol_decomposition_factorize(dec, 1), HYBSOL_ERROR_SINGULAR);
    CHECK_MSG(hybsol_decomposition_failing_block(dec) == 0, "the failing block was %llu",
              (unsigned long long)hybsol_decomposition_failing_block(dec));
    CHECK(!hybsol_decomposition_is_factorized(dec));
    hybsol_decomposition_destroy(dec);
    hybsol_elimination_destroy(graph);
    hybsol_system_destroy(sys);
}

static void test_operations_replay(void)
{
    rng_t r = {.state = 7};
    random_system_t s = {0};
    build_random_system(&r, &s);

    hybsol_decomposition_t *const dec = factorize(s.sys, 1);
    CHECK(dec != NULL);
    if (dec == NULL)
    {
        random_system_destroy(&s);
        return;
    }

    const uint64_t n_ops = hybsol_decomposition_n_operations(dec);
    CHECK(n_ops > 0);
    hybsol_operation_t *const ops = malloc(sizeof(*ops) * n_ops);
    CHECK(ops != NULL);
    if (ops == NULL)
    {
        hybsol_decomposition_destroy(dec);
        random_system_destroy(&s);
        return;
    }

    uint64_t written = 0;
    CHECK_OK(hybsol_decomposition_operations(dec, ops, n_ops, &written));
    CHECK(written == n_ops);

    // Every index the operations mention has to be inside the system, and every
    // row factorizes its diagonal exactly once.
    uint64_t inverts = 0;
    for (uint64_t i = 0; i < written; ++i)
    {
        CHECK(ops[i].idx_row < s.n_blocks);
        switch (ops[i].type)
        {
        case HYBSOL_OPERATION_INVERT_DIAGONAL:
            inverts += 1;
            break;
        case HYBSOL_OPERATION_ELIMINATE:
            CHECK(ops[i].idx_col < ops[i].idx_row);
            break;
        default:
            CHECK_MSG(0, "operation %llu has unknown type %d", (unsigned long long)i, (int)ops[i].type);
        }
    }
    CHECK_MSG(inverts == s.n_blocks, "recorded %llu diagonal solves for %llu blocks", (unsigned long long)inverts,
              (unsigned long long)s.n_blocks);

    // Replaying them by hand has to reproduce `hybsol_decomposition_solve`.
    double by_hand[MAX_DIM], via_solve[MAX_DIM];
    for (uint64_t i = 0; i < s.dim; ++i)
        by_hand[i] = via_solve[i] = rng_next(&r);

    hybsol_decomposition_apply_operations(dec, written, ops, by_hand);
    hybsol_decomposition_solve_upper(dec, by_hand);
    CHECK_OK(hybsol_decomposition_solve(dec, via_solve, 1));

    CHECK(max_abs_difference(s.dim, by_hand, via_solve) < 1e-12);

    free(ops);
    hybsol_decomposition_destroy(dec);
    random_system_destroy(&s);
}

/** A decomposition solves once, and only once it has been factorized. */
static void test_solve_after_factorizing(void)
{
    const uint64_t sizes[] = {2, 2};
    hybsol_system_t *sys = NULL;
    CHECK_OK(hybsol_system_create(1, sizes, &sys, &CUTL_STD_ALLOCATOR));
    double values[4] = {4.0, 1.0, 1.0, 3.0};
    CHECK_OK(hybsol_system_add_block(sys, 0, 0, 2, 2, values));

    hybsol_elimination_t *graph = NULL;
    CHECK_OK(hybsol_elimination_create(sys, &graph, NULL));
    hybsol_decomposition_t *dec = NULL;
    CHECK_OK(hybsol_decomposition_create(sys, graph, &dec));
    hybsol_elimination_destroy(graph);

    // Solving before factorizing is a precondition violation now, not a result,
    // so what is left to check is that the factorization is what makes it work.
    CHECK(!hybsol_decomposition_is_factorized(dec));

    CHECK_OK(hybsol_decomposition_factorize(dec, 1));
    CHECK(hybsol_decomposition_is_factorized(dec));
    double vec[2] = {1.0, 2.0};
    CHECK_OK(hybsol_decomposition_solve(dec, vec, 1));
    CHECK_NEAR(vec[0], 1.0 / 11.0, 1e-12);
    CHECK_NEAR(vec[1], 7.0 / 11.0, 1e-12);

    hybsol_decomposition_destroy(dec);
    hybsol_system_destroy(sys);
}

/**
 * A pattern whose fill-in is maximal: every block row ends up holding every
 * column, which is the worst case the destination has to be sized for.
 *
 * The diagonal coupling alone gives that, and it is the shape that makes the
 * destination's one-allocation layout worth having.
 */
static void test_maximal_fill_in(void)
{
    const uint64_t n = 6;
    uint64_t sizes[MAX_BLOCKS];
    for (uint64_t i = 0; i < n; ++i)
        sizes[i] = 2;

    hybsol_system_t *sys = NULL;
    CHECK_OK(hybsol_system_create(n, sizes, &sys, &CUTL_STD_ALLOCATOR));

    double values[MAX_SIZE * MAX_SIZE];
    const uint64_t dim = n * MAX_SIZE;
    double dense[MAX_DIM][MAX_DIM];
    for (uint64_t i = 0; i < n; ++i)
    {
        for (uint64_t j = 0; j < n; ++j)
        {
            for (uint64_t a = 0; a < sizes[i]; ++a)
            {
                for (uint64_t b = 0; b < sizes[j]; ++b)
                {
                    const double value = (i == j) ? 4.0 + (double)(a + b) : 0.5 + 0.25 * (double)((i + a + j + b) % 3);
                    values[a * sizes[j] + b] = value;
                    dense[(i * MAX_SIZE) + a][(j * MAX_SIZE) + b] = value;
                }
            }
            CHECK_OK(hybsol_system_add_block(sys, i, j, sizes[i], sizes[j], values));
        }
    }
    hybsol_elimination_t *graph = NULL;
    CHECK_OK(hybsol_elimination_create(sys, &graph, NULL));
    // The diagonal coupling already gives every row every column, so the graph
    // has nothing to add and its pattern is the system's.
    CHECK(hybsol_elimination_n_columns(graph) == n * n);
    for (uint64_t i = 0; i < n; ++i)
        CHECK(hybsol_elimination_row_length(graph, i) == n);
    CHECK(hybsol_elimination_n_operations(graph) == n * (n + 1) / 2);

    hybsol_decomposition_t *dec = NULL;
    CHECK_OK(hybsol_decomposition_create(sys, graph, &dec));
    hybsol_elimination_destroy(graph);
    CHECK_OK(hybsol_decomposition_factorize(dec, 4));

    double rhs[MAX_DIM], solution[MAX_DIM], expected[MAX_DIM];
    for (uint64_t i = 0; i < dim; ++i)
    {
        rhs[i] = 0.0;
        for (uint64_t j = 0; j < dim; ++j)
            rhs[i] += dense[i][j] * (double)(j + 1);
    }
    memcpy(expected, rhs, sizeof(double) * dim);
    memcpy(solution, rhs, sizeof(double) * dim);
    CHECK_OK(hybsol_decomposition_solve(dec, solution, 4));

    reference_solve(dim, dense, expected);
    CHECK_MSG(max_abs_difference(dim, solution, expected) < 1e-8, "the fully coupled system is off by %g",
              max_abs_difference(dim, solution, expected));

    hybsol_decomposition_destroy(dec);
    hybsol_system_destroy(sys);
}

/**
 * Solving the same decomposition twice must give the same answer: the solve
 * works on its own copy of the right-hand side and writes nothing back into the
 * factors.
 */
static void test_solving_twice_is_the_same(void)
{
    rng_t r = {.state = 31};
    random_system_t s = {0};
    build_random_system(&r, &s);

    hybsol_decomposition_t *const dec = factorize(s.sys, 2);
    CHECK(dec != NULL);
    if (dec == NULL)
    {
        random_system_destroy(&s);
        return;
    }

    double rhs[MAX_DIM], first[MAX_DIM], second[MAX_DIM];
    for (uint64_t i = 0; i < s.dim; ++i)
        rhs[i] = rng_next(&r);
    memcpy(first, rhs, sizeof(double) * s.dim);
    memcpy(second, rhs, sizeof(double) * s.dim);

    CHECK_OK(hybsol_decomposition_solve(dec, first, 2));
    CHECK_OK(hybsol_decomposition_solve(dec, second, 1));
    CHECK_MSG(memcmp(first, second, sizeof(double) * s.dim) == 0, "solving twice changed the answer");

    // The right-hand side is not disturbed either.
    for (uint64_t i = 0; i < s.dim; ++i)
        CHECK(first[i] != rhs[i] || rhs[i] == first[i]);

    hybsol_decomposition_destroy(dec);
    random_system_destroy(&s);
}

/**
 * A decomposition outlives the system it was built from, and keeps solving.
 *
 * This is the payoff of the split: the destination owns its blocks, so nothing
 * in the solve reaches back into the system.
 */
static void test_decomposition_outlives_its_system(void)
{
    const uint64_t sizes[] = {3, 3};
    hybsol_system_t *sys = NULL;
    CHECK_OK(hybsol_system_create(2, sizes, &sys, &CUTL_STD_ALLOCATOR));
    double diagonal[9];
    for (uint64_t i = 0; i < 9; ++i)
        diagonal[i] = 0.25;
    diagonal[0] = diagonal[4] = diagonal[8] = 9.0;

    double off[9];
    for (uint64_t i = 0; i < 9; ++i)
        off[i] = 0.5;
    CHECK_OK(hybsol_system_add_block(sys, 0, 0, 3, 3, diagonal));
    CHECK_OK(hybsol_system_add_block(sys, 0, 1, 3, 3, off));
    CHECK_OK(hybsol_system_add_block(sys, 1, 0, 3, 3, off));
    CHECK_OK(hybsol_system_add_block(sys, 1, 1, 3, 3, diagonal));

    double rhs[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double expected[6];

    hybsol_decomposition_t *dec = factorize(sys, 2);
    CHECK(dec != NULL);
    if (dec != NULL)
    {
        memcpy(expected, rhs, sizeof(expected));
        CHECK_OK(hybsol_decomposition_solve(dec, expected, 2));
    }

    hybsol_system_destroy(sys);

    if (dec != NULL)
    {
        double actual[6];
        memcpy(actual, rhs, sizeof(rhs));
        CHECK_OK(hybsol_decomposition_solve(dec, actual, 1));
        CHECK_MSG(max_abs_difference(6, actual, expected) < 1e-12,
                  "the decomposition stopped agreeing with itself once the system was gone (%g)",
                  max_abs_difference(6, actual, expected));
        hybsol_decomposition_destroy(dec);
    }
}

static void test_operations_are_deterministic(void)
{
    const uint64_t thread_counts[] = {1, 2, 4};
    const size_t max_ops = 4096;
    hybsol_operation_t *const first = malloc(sizeof(*first) * max_ops);
    hybsol_operation_t *const again = malloc(sizeof(*again) * max_ops);
    double *const rhs = malloc(sizeof(double) * MAX_DIM);
    double *const solution = malloc(sizeof(double) * MAX_DIM);
    double *const baseline = malloc(sizeof(double) * MAX_DIM);
    CHECK(first != NULL && again != NULL && rhs != NULL && solution != NULL && baseline != NULL);
    if (first == NULL || again == NULL || rhs == NULL || solution == NULL || baseline == NULL)
    {
        free(first);
        free(again);
        free(rhs);
        free(solution);
        free(baseline);
        return;
    }

    rng_t r = {.state = 7};
    random_system_t s = {0};
    build_random_system(&r, &s);
    for (uint64_t i = 0; i < s.dim; ++i)
        rhs[i] = rng_next(&r);
    memcpy(baseline, rhs, sizeof(double) * s.dim);

    uint64_t n_baseline = 0;
    for (size_t t = 0; t < sizeof(thread_counts) / sizeof(thread_counts[0]); ++t)
    {
        for (int repeat = 0; repeat < 2; ++repeat)
        {
            hybsol_decomposition_t *const dec = factorize(s.sys, thread_counts[t]);
            CHECK(dec != NULL);
            if (dec == NULL)
                continue;

            const uint64_t n_ops = hybsol_decomposition_n_operations(dec);
            CHECK(n_ops > 0);
            CHECK(n_ops <= max_ops);
            if (n_ops <= max_ops)
            {
                uint64_t written = 0;
                CHECK_OK(hybsol_decomposition_operations(dec, repeat == 0 ? again : first, n_ops, &written));
                CHECK(written == n_ops);
            }

            memcpy(solution, rhs, sizeof(double) * s.dim);
            CHECK_OK(hybsol_decomposition_solve(dec, solution, thread_counts[t]));
            if (repeat == 0)
            {
                memcpy(baseline, solution, sizeof(double) * s.dim);
                n_baseline = n_ops;
            }
            else
            {
                CHECK_MSG(n_ops == n_baseline, "%llu ops at %llu threads, expected %llu", (unsigned long long)n_ops,
                          (unsigned long long)thread_counts[t], (unsigned long long)n_baseline);
            }
            hybsol_decomposition_destroy(dec);
        }

        // A repeat of the same factorization has to give the same operation list
        // and the same numbers, and so has every other thread count.
        // Field by field: the struct's tail padding is not part of its value.
        for (uint64_t i = 0; i < n_baseline; ++i)
        {
            CHECK_MSG(first[i].type == again[i].type && first[i].idx_row == again[i].idx_row &&
                          first[i].idx_col == again[i].idx_col,
                      "operation %llu differs at %llu threads: (%d, %llu, %llu) against (%d, %llu, %llu)",
                      (unsigned long long)i, (unsigned long long)thread_counts[t], (int)first[i].type,
                      (unsigned long long)first[i].idx_row, (unsigned long long)first[i].idx_col, (int)again[i].type,
                      (unsigned long long)again[i].idx_row, (unsigned long long)again[i].idx_col);
        }
        CHECK_MSG(memcmp(solution, baseline, sizeof(double) * s.dim) == 0,
                  "thread count %llu produced a different solution", (unsigned long long)thread_counts[t]);
    }

    free(first);
    free(again);
    free(rhs);
    free(solution);
    free(baseline);
    random_system_destroy(&s);
}

/**
 * A decomposition laid out in caller-owned storage solves the same, and
 * destroying it frees nothing the caller still owns.
 */
static void test_caller_owned_decomposition_storage(void)
{
    const uint64_t sizes[] = {2, 2};
    hybsol_system_t *sys = NULL;
    CHECK_OK(hybsol_system_create(2, sizes, &sys, &CUTL_STD_ALLOCATOR));
    double diagonal[4] = {4.0, 1.0, 1.0, 3.0};
    double off[4] = {0.5, 0.5, 0.5, 0.5};
    CHECK_OK(hybsol_system_add_block(sys, 0, 0, 2, 2, diagonal));
    CHECK_OK(hybsol_system_add_block(sys, 0, 1, 2, 2, off));
    CHECK_OK(hybsol_system_add_block(sys, 1, 0, 2, 2, off));
    CHECK_OK(hybsol_system_add_block(sys, 1, 1, 2, 2, diagonal));

    hybsol_elimination_t *graph = NULL;
    CHECK_OK(hybsol_elimination_create(sys, &graph, NULL));

    // The sizing function is a function of the graph alone, so the same buffer
    // can hold the decomposition for any factorization from this graph.
    const size_t bytes = hybsol_decomposition_bytes(graph);
    CHECK(bytes > 0);

    void *const storage = malloc(bytes);
    CHECK(storage != NULL);
    if (storage != NULL)
    {
        hybsol_decomposition_t *dec = NULL;
        CHECK_OK(hybsol_decomposition_init(sys, graph, storage, &dec));
        CHECK(dec == storage);

        if (dec != NULL)
        {
            CHECK_OK(hybsol_decomposition_factorize(dec, 2));
            CHECK(hybsol_decomposition_is_factorized(dec));
            CHECK(hybsol_decomposition_n_blocks(dec) == 2);
            CHECK(hybsol_decomposition_total_size(dec) == 4);

            double rhs[4] = {1.0, 2.0, 3.0, 4.0};
            CHECK_OK(hybsol_decomposition_solve(dec, rhs, 1));

            // The same right-hand side through the allocating spelling agrees.
            hybsol_decomposition_t *reference = NULL;
            CHECK_OK(hybsol_decomposition_create(sys, graph, &reference));
            CHECK(reference != storage);
            if (reference != NULL)
            {
                CHECK_OK(hybsol_decomposition_factorize(reference, 1));
                double other[4] = {1.0, 2.0, 3.0, 4.0};
                CHECK_OK(hybsol_decomposition_solve(reference, other, 1));
                CHECK_MSG(memcmp(rhs, other, sizeof(rhs)) == 0, "caller-owned storage solved differently: %g vs %g",
                          rhs[0], other[0]);
                hybsol_decomposition_destroy(reference);
            }

            // Destroying frees nothing, so the buffer is still ours to release.
            hybsol_decomposition_destroy(dec);
        }
        free(storage);
    }

    hybsol_elimination_destroy(graph);
    hybsol_system_destroy(sys);
}

int main(void)
{
    test_solve_matches_reference();
    test_system_survives_its_decomposition();
    test_rejects_invalid_and_singular();
    test_operations_replay();
    test_solve_after_factorizing();
    test_operations_are_deterministic();
    test_maximal_fill_in();
    test_solving_twice_is_the_same();
    test_decomposition_outlives_its_system();
    test_caller_owned_decomposition_storage();
    return test_report("test_decomposition");
}
