/** @file test_elimination.c
 * The elimination graph: the final pattern it predicts, the passes it hands
 * the factorization, and the failures it rejects.
 *
 * The worked example is four 1x1 blocks coupled by the below-diagonal entries
 * ``(1,0)``, ``(2,1)`` and ``(3,1)``. Rows 2 and 3 each gain a column the
 * system does not store, which is the fill-in the graph has to account for.
 */

#include "test_util.h"

#include <hybsol/hybsol.h>

#include <stdlib.h>
#include <string.h>

/* A system this size has blocks of at most MAX_SIZE scalars each. */
#define MAX_BLOCKS 8
#define MAX_SIZE 3

/** The four-block example: stored rows {0,1}, {0,1,2,3}, {1,2}, {1,3}. */
static hybsol_system_t *build_fill_in_system(void)
{
    const uint64_t sizes[] = {1, 1, 1, 1};
    hybsol_system_t *sys = NULL;
    if (hybsol_system_create(4, sizes, &sys, &CUTL_STD_ALLOCATOR) != HYBSOL_SUCCESS)
        return NULL;

    const uint64_t pairs[][2] = {{1, 0}, {2, 1}, {3, 1}};
    for (size_t k = 0; k < sizeof(pairs) / sizeof(pairs[0]); ++k)
    {
        const double value = 1.0 + (double)k;
        if (hybsol_system_add_block(sys, pairs[k][0], pairs[k][1], 1, 1, &value) != HYBSOL_SUCCESS ||
            hybsol_system_add_block(sys, pairs[k][1], pairs[k][0], 1, 1, &value) != HYBSOL_SUCCESS)
        {
            hybsol_system_destroy(sys);
            return NULL;
        }
    }

    for (uint64_t i = 0; i < 4; ++i)
    {
        const double diagonal = 4.0 + (double)i;
        if (hybsol_system_add_block(sys, i, i, 1, 1, &diagonal) != HYBSOL_SUCCESS)
        {
            hybsol_system_destroy(sys);
            return NULL;
        }
    }
    return sys;
}

static void test_graph_of_the_fill_in_example(void)
{
    hybsol_system_t *const sys = build_fill_in_system();
    CHECK(sys != NULL);
    if (sys == NULL)
        return;

    hybsol_elimination_t *graph = NULL;
    CHECK_OK(hybsol_elimination_create(sys, &graph, NULL));
    CHECK(graph != NULL);
    if (graph == NULL)
    {
        hybsol_system_destroy(sys);
        return;
    }

    CHECK(hybsol_elimination_n_blocks(graph) == 4);
    CHECK(hybsol_elimination_precision(graph) == HYBSOL_PRECISION_DOUBLE);
    CHECK(hybsol_elimination_failing_block(graph) == UINT64_MAX);

    // Rows 2 and 3 each gain the column the other brought in.
    CHECK(hybsol_elimination_n_columns(graph) == 12);
    const uint64_t stored_columns[] = {2, 4, 2, 2};
    const uint64_t final_lengths[4] = {2, 4, 3, 3};
    const uint64_t final_columns[4][4] = {{0, 1, 0, 0}, {0, 1, 2, 3}, {1, 2, 3, 0}, {1, 2, 3, 0}};
    for (uint64_t row = 0; row < 4; ++row)
    {
        CHECK_MSG(hybsol_elimination_row_length(graph, row) == final_lengths[row], "row %llu has length %llu",
                  (unsigned long long)row, (unsigned long long)hybsol_elimination_row_length(graph, row));
        // Only the two rows that had to wait for another row's columns gain any.
        if (row >= 2)
            CHECK_MSG(hybsol_elimination_row_length(graph, row) > stored_columns[row], "row %llu did not gain a column",
                      (unsigned long long)row);
        else
            CHECK_MSG(hybsol_elimination_row_length(graph, row) == stored_columns[row],
                      "row %llu should not have gained a column", (unsigned long long)row);
        uint64_t columns[4] = {0};
        uint64_t written = 0;
        CHECK_OK(hybsol_elimination_row_columns(graph, row, columns, 4, &written));
        CHECK(written == final_lengths[row]);
        for (uint64_t j = 0; j < written; ++j)
            CHECK_MSG(columns[j] == final_columns[row][j], "row %llu column %llu is %llu, expected %llu",
                      (unsigned long long)row, (unsigned long long)j, (unsigned long long)columns[j],
                      (unsigned long long)final_columns[row][j]);
    }

    // Every block is a 1x1 double: an 8-byte header plus 8 bytes of value,
    // rounded up to the library's 64-byte granularity.
    CHECK_MSG(hybsol_elimination_value_bytes(graph) == 12 * 64, "value_bytes is %zu",
              hybsol_elimination_value_bytes(graph));
    CHECK_MSG(hybsol_elimination_total_bytes(graph) > 0, "total_bytes is %zu", hybsol_elimination_total_bytes(graph));

    // The passes, in the order the walk found them.
    CHECK(hybsol_elimination_n_levels(graph) == 4);
    const uint64_t pass_counts[4] = {1, 1, 2, 1};
    const uint64_t pass_rows[4][2] = {{0, 0}, {1, 0}, {2, 3}, {3, 0}};
    for (uint64_t pass = 0; pass < 4; ++pass)
    {
        CHECK_MSG(hybsol_elimination_level_size(graph, pass) == pass_counts[pass], "pass %llu holds %llu rows",
                  (unsigned long long)pass, (unsigned long long)hybsol_elimination_level_size(graph, pass));

        uint64_t rows[2] = {0};
        uint64_t written = 0;
        CHECK_OK(hybsol_elimination_level_rows(graph, pass, rows, 2, &written));
        for (uint64_t k = 0; k < written; ++k)
            CHECK_MSG(rows[k] == pass_rows[pass][k], "pass %llu row %llu is %llu, expected %llu",
                      (unsigned long long)pass, (unsigned long long)k, (unsigned long long)rows[k],
                      (unsigned long long)pass_rows[pass][k]);
    }

    // A row is processed in as many passes as it has eliminations, except a
    // diagonal-first row, which is processed once and eliminates nothing.
    const uint64_t first[4] = {0, 1, 2, 2};
    const uint64_t n_elim[4] = {0, 1, 1, 2};
    const uint64_t last[4] = {0, 1, 2, 3};
    for (uint64_t row = 0; row < 4; ++row)
    {
        CHECK_MSG(hybsol_elimination_row_first_level(graph, row) == first[row], "row %llu first pass is %llu",
                  (unsigned long long)row, (unsigned long long)hybsol_elimination_row_first_level(graph, row));
        CHECK_MSG(hybsol_elimination_row_n_eliminations(graph, row) == n_elim[row], "row %llu eliminates %llu times",
                  (unsigned long long)row, (unsigned long long)hybsol_elimination_row_n_eliminations(graph, row));
        CHECK_MSG(hybsol_elimination_row_level(graph, row) == last[row], "row %llu last pass is %llu",
                  (unsigned long long)row, (unsigned long long)hybsol_elimination_row_level(graph, row));
    }

    // Four diagonal solves and four eliminations.
    CHECK(hybsol_elimination_n_operations(graph) == 8);

    hybsol_elimination_destroy(graph);
    hybsol_system_destroy(sys);
}

/** Out-of-range indices report nothing rather than reading past the end. */
static void test_out_of_range_queries(void)
{
    hybsol_system_t *const sys = build_fill_in_system();
    CHECK(sys != NULL);
    if (sys == NULL)
        return;

    hybsol_elimination_t *graph = NULL;
    CHECK_OK(hybsol_elimination_create(sys, &graph, NULL));
    CHECK(graph != NULL);
    if (graph == NULL)
    {
        hybsol_system_destroy(sys);
        return;
    }

    CHECK(hybsol_elimination_row_length(graph, 4) == 0);
    CHECK(hybsol_elimination_level_size(graph, 4) == 0);

    hybsol_elimination_destroy(graph);
    hybsol_elimination_destroy(NULL);
    hybsol_system_destroy(sys);
}

/** The same system always yields the same graph. */
static void test_graph_is_reproducible(void)
{
    hybsol_system_t *const sys = build_fill_in_system();
    CHECK(sys != NULL);
    if (sys == NULL)
        return;

    hybsol_elimination_t *first = NULL;
    hybsol_elimination_t *second = NULL;
    CHECK_OK(hybsol_elimination_create(sys, &first, NULL));
    CHECK_OK(hybsol_elimination_create(sys, &second, NULL));
    if (first == NULL || second == NULL)
    {
        hybsol_elimination_destroy(first);
        hybsol_elimination_destroy(second);
        hybsol_system_destroy(sys);
        return;
    }

    CHECK(hybsol_elimination_value_bytes(first) == hybsol_elimination_value_bytes(second));
    CHECK(hybsol_elimination_total_bytes(first) == hybsol_elimination_total_bytes(second));
    CHECK(hybsol_elimination_n_operations(first) == hybsol_elimination_n_operations(second));
    for (uint64_t row = 0; row < 4; ++row)
    {
        uint64_t a[8] = {0};
        uint64_t b[8] = {0};
        uint64_t na = 0;
        uint64_t nb = 0;
        CHECK_OK(hybsol_elimination_row_columns(first, row, a, 8, &na));
        CHECK_OK(hybsol_elimination_row_columns(second, row, b, 8, &nb));
        CHECK_MSG(na == nb && memcmp(a, b, sizeof(a[0]) * na) == 0, "row %llu differs between two graphs",
                  (unsigned long long)row);
    }

    // The graph outlives the system it describes.
    hybsol_system_destroy(sys);
    CHECK(hybsol_elimination_n_blocks(first) == 4);
    CHECK(hybsol_elimination_n_operations(first) == 8);

    hybsol_elimination_destroy(first);
    hybsol_elimination_destroy(second);
}

/** A fully coupled system records the maximum number of operations. */
static void test_dense_system_operation_count(void)
{
    const uint64_t sizes[] = {2, 2, 2};
    hybsol_system_t *sys = NULL;
    CHECK_OK(hybsol_system_create(3, sizes, &sys, &CUTL_STD_ALLOCATOR));

    for (uint64_t i = 0; i < 3; ++i)
        for (uint64_t j = 0; j < 3; ++j)
        {
            double values[4] = {0};
            values[0] = 2.0;
            values[1] = 0.5;
            values[2] = 0.5;
            values[3] = 2.0 + (double)(i + j) * 0.25;
            CHECK_OK(hybsol_system_add_block(sys, i, j, 2, 2, values));
        }

    hybsol_elimination_t *graph = NULL;
    CHECK_OK(hybsol_elimination_create(sys, &graph, NULL));
    CHECK(graph != NULL);
    if (graph != NULL)
    {
        // One diagonal solve per block, one elimination per (target, source) pair.
        CHECK_MSG(hybsol_elimination_n_operations(graph) == 3 * 4 / 2, "dense 3-block system records %llu operations",
                  (unsigned long long)hybsol_elimination_n_operations(graph));
        hybsol_elimination_destroy(graph);
    }
    hybsol_system_destroy(sys);
}

/**
 * Whatever a reorder accepts, a decomposition must go on to accept.
 *
 * The reorder runs the same walk and rejects a bad order up front, so a
 * permutation that gets through it is one the factorization can actually use.
 * Random permutations make the two disagree if that ever stops being true.
 */
static void test_reorder_and_decomposition_agree(void)
{
    const uint64_t n = 5;
    const uint64_t sizes[] = {2, 3, 2, 3, 2};
    hybsol_system_t *sys = NULL;
    CHECK_OK(hybsol_system_create(n, sizes, &sys, &CUTL_STD_ALLOCATOR));

    // A chain of couplings: symmetric, diagonally dominant, and the kind of
    // pattern a bad permutation can wreck.
    double values[MAX_SIZE * MAX_SIZE];
    for (uint64_t i = 0; i < n; ++i)
    {
        for (uint64_t j = i; j < n; ++j)
        {
            const uint64_t span = (j == i) ? sizes[i] : sizes[i] * sizes[j];
            for (uint64_t k = 0; k < span; ++k)
                values[k] = 0.25;
            if (i == j)
            {
                for (uint64_t a = 0; a < sizes[i]; ++a)
                    values[a * sizes[i] + a] = 6.0 + (double)(a + i);
            }
            CHECK_OK(hybsol_system_add_block(sys, i, j, sizes[i], sizes[j], values));
            if (i != j)
                CHECK_OK(hybsol_system_add_block(sys, j, i, sizes[j], sizes[i], values));
        }
    }

    uint64_t state = 20240915u;
    uint64_t accepted = 0;
    for (int trial = 0; trial < 40; ++trial)
    {
        uint64_t order[MAX_BLOCKS];
        for (uint64_t i = 0; i < n; ++i)
            order[i] = i;
        for (uint64_t i = n; i > 1; --i)
        {
            state = state * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
            const uint64_t pick = (uint64_t)((state >> 33) % i);
            const uint64_t tmp = order[i - 1];
            order[i - 1] = order[pick];
            order[pick] = tmp;
        }

        hybsol_system_t *shuffled = NULL;
        CHECK_OK(hybsol_system_copy(sys, &shuffled));
        uint64_t failing = UINT64_MAX;
        if (hybsol_system_reorder_blocks(shuffled, order, 1, &failing) != HYBSOL_SUCCESS)
        {
            // Rejected, so the failing block must be named.
            CHECK_MSG(failing < n, "trial %d named block %llu", trial, (unsigned long long)failing);
            hybsol_system_destroy(shuffled);
            continue;
        }
        accepted += 1;

        hybsol_elimination_t *graph = NULL;
        const hybsol_result_t walked = hybsol_elimination_create(shuffled, &graph, NULL);
        CHECK_MSG(walked == HYBSOL_SUCCESS, "trial %d: the reorder accepted a walk that did not (%s)", trial,
                  hybsol_result_str(walked));
        if (walked == HYBSOL_SUCCESS)
        {
            hybsol_decomposition_t *dec = NULL;
            CHECK_OK(hybsol_decomposition_create(shuffled, graph, &dec));
            hybsol_elimination_destroy(graph);
            const hybsol_result_t factorized = hybsol_decomposition_factorize(dec, 1);
            CHECK_MSG(factorized == HYBSOL_SUCCESS,
                      "trial %d: the reorder accepted a system that would not factor (%s)", trial,
                      hybsol_result_str(factorized));
            hybsol_decomposition_destroy(dec);
        }
        hybsol_system_destroy(shuffled);
    }

    CHECK_MSG(accepted > 0, "every permutation was rejected, so nothing was actually checked");
    hybsol_system_destroy(sys);
}

/**
 * A block whose diagonal is structurally zero becomes one only after a reorder,
 * so the check has to see the order the caller actually asked for.
 */
static void test_reorder_reports_a_block_it_creates(void)
{
    const uint64_t sizes[] = {1, 1, 1};
    hybsol_system_t *sys = NULL;
    CHECK_OK(hybsol_system_create(3, sizes, &sys, &CUTL_STD_ALLOCATOR));

    // Element block 0, coupled to multiplier blocks 1 and 2, which are
    // structurally zero: an augmented system, which a coloring can order badly.
    const double one = 1.0;
    CHECK_OK(hybsol_system_add_block(sys, 0, 0, 1, 1, &one));
    CHECK_OK(hybsol_system_add_block(sys, 0, 1, 1, 1, &one));
    CHECK_OK(hybsol_system_add_block(sys, 1, 0, 1, 1, &one));
    const double zero = 0.0;
    CHECK_OK(hybsol_system_add_block(sys, 1, 1, 1, 1, &zero));
    CHECK_OK(hybsol_system_add_block(sys, 0, 2, 1, 1, &one));
    CHECK_OK(hybsol_system_add_block(sys, 2, 0, 1, 1, &one));
    CHECK_OK(hybsol_system_add_block(sys, 2, 2, 1, 1, &zero));

    // Elements first is factorizable.
    const uint64_t good[] = {0, 1, 2};
    CHECK_OK(hybsol_system_reorder_blocks(sys, good, 1, NULL));

    hybsol_system_t *bad_sys = NULL;
    CHECK_OK(hybsol_system_copy(sys, &bad_sys));
    const uint64_t bad[] = {1, 2, 0};
    uint64_t failing = UINT64_MAX;
    CHECK_RESULT(hybsol_system_reorder_blocks(bad_sys, bad, 1, &failing), HYBSOL_ERROR_INVALID_ORDERING);
    CHECK_MSG(failing < 3, "the failing block was %llu", (unsigned long long)failing);

    hybsol_system_destroy(bad_sys);
    hybsol_system_destroy(sys);
}

static void test_rejected_systems(void)
{
    const uint64_t sizes[] = {2, 2};
    double values[4] = {2.0, 1.0, 1.0, 2.0};

    // The walk asserts structural validity rather than reporting it, so
    // hybsol_system_is_valid is what a caller checks to get a diagnosis: a row
    // with no diagonal block, then a block below the diagonal with no mirror.
    hybsol_system_t *sys = NULL;
    CHECK_OK(hybsol_system_create(2, sizes, &sys, &CUTL_STD_ALLOCATOR));
    CHECK_OK(hybsol_system_add_block(sys, 0, 0, 2, 2, values));
    CHECK(!hybsol_system_is_valid(sys));
    hybsol_system_destroy(sys);

    CHECK_OK(hybsol_system_create(2, sizes, &sys, &CUTL_STD_ALLOCATOR));
    CHECK_OK(hybsol_system_add_block(sys, 0, 0, 2, 2, values));
    CHECK_OK(hybsol_system_add_block(sys, 1, 1, 2, 2, values));
    CHECK_OK(hybsol_system_add_block(sys, 1, 0, 2, 2, values));
    CHECK(!hybsol_system_is_valid(sys));
    hybsol_system_destroy(sys);

    // A structurally zero diagonal: no ordering can factorize it, and the
    // offending block is named through the out-parameter.
    CHECK_OK(hybsol_system_create(1, sizes, &sys, &CUTL_STD_ALLOCATOR));
    double zero[4] = {0};
    CHECK_OK(hybsol_system_add_block(sys, 0, 0, 2, 2, zero));
    CHECK(hybsol_system_is_valid(sys));
    hybsol_elimination_t *graph = (hybsol_elimination_t *)(uintptr_t)1;
    uint64_t failing = UINT64_MAX;
    CHECK_RESULT(hybsol_elimination_create(sys, &graph, &failing), HYBSOL_ERROR_INVALID_ORDERING);
    CHECK(graph == NULL);
    CHECK_MSG(failing == 0, "the failing block was %llu", (unsigned long long)failing);
    hybsol_system_destroy(sys);
}

int main(void)
{
    test_graph_of_the_fill_in_example();
    test_out_of_range_queries();
    test_graph_is_reproducible();
    test_dense_system_operation_count();
    test_reorder_and_decomposition_agree();
    test_reorder_reports_a_block_it_creates();
    test_rejected_systems();
    return test_report("test_elimination");
}
