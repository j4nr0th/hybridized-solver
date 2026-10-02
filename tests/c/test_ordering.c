/** @file test_ordering.c
 * Coloring-based reordering: permutation validity, the coloring property,
 * ``max_colors`` handling and the reorder/unorder round trip.
 */

#include "test_util.h"

#include <hybsol/hybsol.h>

#include <stdlib.h>
#include <string.h>

/** Walk the graph, lay out the destination and factorize it. */
static hybsol_decomposition_t *factorize(hybsol_system_t *const sys, const uint64_t n_threads)
{
    hybsol_elimination_t *graph = NULL;
    if (hybsol_elimination_create(sys, &graph) != HYBSOL_SUCCESS)
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

#define MAX_BLOCKS 40
#define MAX_DIM (MAX_BLOCKS * 4)

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

typedef struct
{
    hybsol_system_t *sys;
    uint64_t n_blocks;
    uint64_t dim;
} random_system_t;

static void random_system_destroy(random_system_t *const s)
{
    hybsol_system_destroy(s->sys);
    s->sys = NULL;
}

/** Random symmetric sparsity pattern with a diagonal block in every row. */
static void build_random_system(rng_t *const r, random_system_t *const s)
{
    s->n_blocks = 2 + rng_below(r, MAX_BLOCKS - 1);
    uint64_t sizes[MAX_BLOCKS], offsets[MAX_BLOCKS + 1] = {0};
    for (uint64_t i = 0; i < s->n_blocks; ++i)
    {
        sizes[i] = 1 + rng_below(r, 4);
        offsets[i + 1] = offsets[i] + sizes[i];
    }
    s->dim = offsets[s->n_blocks];

    hybsol_system_t *sys = NULL;
    CHECK_OK(hybsol_system_create(s->n_blocks, sizes, &sys, &CUTL_STD_ALLOCATOR));
    s->sys = sys;

    double block[16];
    for (uint64_t i = 0; i < s->n_blocks; ++i)
    {
        for (uint64_t k = 0; k < sizes[i] * sizes[i]; ++k)
            block[k] = rng_next(r);
        CHECK_OK(hybsol_system_add_block(sys, i, i, sizes[i], sizes[i], block));
    }

    for (uint64_t i = 0; i < s->n_blocks; ++i)
    {
        for (uint64_t j = i + 1; j < s->n_blocks; ++j)
        {
            if (rng_next(r) < 0.7)
                continue;

            double pair[16], mirror[16];
            for (uint64_t a = 0; a < sizes[i] * sizes[j]; ++a)
                pair[a] = rng_next(r);
            for (uint64_t a = 0; a < sizes[j]; ++a)
            {
                for (uint64_t b = 0; b < sizes[i]; ++b)
                    mirror[a * sizes[i] + b] = pair[b * sizes[j] + a];
            }

            CHECK_OK(hybsol_system_add_block(sys, i, j, sizes[i], sizes[j], pair));
            CHECK_OK(hybsol_system_add_block(sys, j, i, sizes[j], sizes[i], mirror));
        }
    }

    CHECK(hybsol_system_is_valid(sys));
}

static void test_ordering_is_a_permutation(void)
{
    static const hybsol_ordering_strategy_t strategies[] = {
        HYBSOL_ORDERING_FIRST,
        HYBSOL_ORDERING_GREEDY,
        HYBSOL_ORDERING_BALANCED,
    };

    rng_t r = {.state = 31337};
    for (int trial = 0; trial < 20; ++trial)
    {
        random_system_t s = {0};
        build_random_system(&r, &s);

        for (size_t k = 0; k < sizeof(strategies) / sizeof(strategies[0]); ++k)
        {
            uint64_t order[MAX_BLOCKS];
            uint8_t seen[MAX_BLOCKS] = {0};

            CHECK_OK(hybsol_system_compute_reordering(s.sys, strategies[k], 0, order));

            for (uint64_t i = 0; i < s.n_blocks; ++i)
            {
                CHECK_MSG(order[i] < s.n_blocks, "order[%llu] = %llu is out of range", (unsigned long long)i,
                          (unsigned long long)order[i]);
                if (order[i] < s.n_blocks)
                    seen[order[i]] += 1;
            }
            for (uint64_t i = 0; i < s.n_blocks; ++i)
                CHECK_MSG(seen[i] == 1, "index %llu was used %u times", (unsigned long long)i, seen[i]);

            // The identity is a valid input to coloring an already reordered system, so this must be stable.
            uint64_t again[MAX_BLOCKS];
            CHECK_OK(hybsol_system_compute_reordering(s.sys, strategies[k], 0, again));
            CHECK(memcmp(order, again, sizeof(uint64_t) * s.n_blocks) == 0);
        }

        random_system_destroy(&s);
    }
}

/** Group the reordered blocks into maximal runs of mutually non-adjacent ones. */
static void test_coloring_property(void)
{
    rng_t r = {.state = 5150};
    for (int trial = 0; trial < 20; ++trial)
    {
        random_system_t s = {0};
        build_random_system(&r, &s);

        uint64_t order[MAX_BLOCKS];
        CHECK_OK(hybsol_system_compute_reordering(s.sys, HYBSOL_ORDERING_FIRST, 0, order));

        // `order[i]` is the new position of old block `i`; build the inverse.
        uint64_t where[MAX_BLOCKS];
        for (uint64_t i = 0; i < s.n_blocks; ++i)
            where[order[i]] = i;

        // Walk the rows in the reordered order and assign the lowest color no
        // already-placed neighbour uses; `n_colors` is then exactly what a
        // bounded coloring must accept.
        uint8_t color[MAX_BLOCKS] = {0};
        uint64_t n_colors = 0;
        for (uint64_t slot = 0; slot < s.n_blocks; ++slot)
        {
            const uint64_t i = where[slot];
            bool used[MAX_BLOCKS] = {false};
            for (uint64_t j = 0; j < s.n_blocks; ++j)
            {
                if (j == i || !hybsol_system_has_block(s.sys, i, j))
                    continue;
                // `j` has only been placed if its slot is before `slot`.
                if (order[j] < slot)
                    used[color[j]] = true;
            }

            uint64_t c = 0;
            while (used[c])
                c += 1;
            color[i] = (uint8_t)c;
            if (c + 1 > n_colors)
                n_colors = c + 1;
        }

        // Asking for exactly that many colors has to succeed.
        uint64_t bounded[MAX_BLOCKS];
        CHECK_OK(hybsol_system_compute_reordering(s.sys, HYBSOL_ORDERING_FIRST, n_colors, bounded));

        // ... and one color less has to be refused.
        if (n_colors > 1)
        {
            CHECK_RESULT(hybsol_system_compute_reordering(s.sys, HYBSOL_ORDERING_FIRST, n_colors - 1, bounded),
                         HYBSOL_ERROR_MAX_COLORS);
        }

        // No two blocks of the same color may share an off-diagonal block.
        for (uint64_t i = 0; i < s.n_blocks; ++i)
        {
            for (uint64_t j = i + 1; j < s.n_blocks; ++j)
            {
                if (color[i] != color[j])
                    continue;
                CHECK_MSG(!hybsol_system_has_block(s.sys, i, j), "blocks %llu and %llu share color %u",
                          (unsigned long long)i, (unsigned long long)j, color[i]);
            }
        }

        random_system_destroy(&s);
    }
}

static void test_reorder_roundtrip(void)
{
    rng_t r = {.state = 909};
    for (int trial = 0; trial < 15; ++trial)
    {
        random_system_t s = {0};
        build_random_system(&r, &s);

        double before[MAX_DIM];
        for (uint64_t i = 0; i < s.dim; ++i)
            before[i] = rng_next(&r);

        // A random permutation.
        uint64_t order[MAX_BLOCKS];
        for (uint64_t i = 0; i < s.n_blocks; ++i)
            order[i] = i;
        for (uint64_t i = s.n_blocks; i > 1; --i)
        {
            const uint64_t j = rng_below(&r, i);
            const uint64_t tmp = order[i - 1];
            order[i - 1] = order[j];
            order[j] = tmp;
        }

        // The permutation applied to the vectors and blocks, as offsets.
        uint64_t sizes_before[MAX_BLOCKS], offsets_before[MAX_BLOCKS + 1] = {0}, offsets_after[MAX_BLOCKS + 1] = {0};
        for (uint64_t i = 0; i < s.n_blocks; ++i)
        {
            sizes_before[i] = hybsol_system_block_size(s.sys, i);
            offsets_before[i + 1] = offsets_before[i] + sizes_before[i];
        }
        for (uint64_t i = 0; i < s.n_blocks; ++i)
            offsets_after[order[i] + 1] = sizes_before[i];
        for (uint64_t i = 0; i < s.n_blocks; ++i)
            offsets_after[i + 1] += offsets_after[i];

        double permuted[MAX_DIM];
        for (uint64_t i = 0; i < s.n_blocks; ++i)
        {
            const uint64_t size = sizes_before[i];
            for (uint64_t k = 0; k < size; ++k)
                permuted[offsets_after[order[i]] + k] = before[offsets_before[i] + k];
        }

        double dense_before[MAX_DIM * MAX_DIM] = {0};
        CHECK_OK(hybsol_system_to_dense(s.sys, dense_before));

        // The vector helpers use the offsets of the system they are given, so reorder it first.
        CHECK_OK(hybsol_system_reorder_blocks(s.sys, order, 1));
        CHECK(hybsol_system_n_blocks(s.sys) == s.n_blocks);
        for (uint64_t i = 0; i < s.n_blocks; ++i)
            CHECK(hybsol_system_block_size(s.sys, order[i]) == sizes_before[i]);

        // Reordering the vector must match the matrix permutation ...
        double reordered[MAX_DIM];
        hybsol_system_reorder_vector(s.sys, order, before, reordered);
        for (uint64_t i = 0; i < s.dim; ++i)
            CHECK_NEAR(reordered[i], permuted[i], 0.0);
        // ... and undoing it has to give the identity.
        double restored[MAX_DIM];
        hybsol_system_unorder_vector(s.sys, order, reordered, restored);
        for (uint64_t i = 0; i < s.dim; ++i)
            CHECK_NEAR(restored[i], before[i], 0.0);

        // ... and the matrix must become P A P^T.
        double dense_after[MAX_DIM * MAX_DIM] = {0};
        CHECK_OK(hybsol_system_to_dense(s.sys, dense_after));
        for (uint64_t i = 0; i < s.n_blocks; ++i)
        {
            for (uint64_t j = 0; j < s.n_blocks; ++j)
            {
                const uint64_t si = sizes_before[i], sj = sizes_before[j];
                for (uint64_t a = 0; a < si; ++a)
                {
                    for (uint64_t b = 0; b < sj; ++b)
                    {
                        CHECK_NEAR(dense_after[(offsets_after[order[i]] + a) * s.dim + (offsets_after[order[j]] + b)],
                                   dense_before[(offsets_before[i] + a) * s.dim + (offsets_before[j] + b)], 0.0);
                    }
                }
            }
        }

        random_system_destroy(&s);
    }
}

static void test_reordered_system_solves_consistently(void)
{
    // Solving a reordered system with a reordered right-hand side must give
    // the reordered solution; this is the whole point of the reordering.
    rng_t r = {.state = 4242};
    for (int trial = 0; trial < 10; ++trial)
    {
        random_system_t s = {0};
        build_random_system(&r, &s);

        uint64_t order[MAX_BLOCKS];
        for (uint64_t i = 0; i < s.n_blocks; ++i)
            order[i] = i;
        for (uint64_t i = s.n_blocks; i > 1; --i)
        {
            const uint64_t j = rng_below(&r, i);
            const uint64_t tmp = order[i - 1];
            order[i - 1] = order[j];
            order[j] = tmp;
        }

        hybsol_system_t *reordered = NULL;
        CHECK_OK(hybsol_system_copy(s.sys, &reordered));
        CHECK_OK(hybsol_system_reorder_blocks(reordered, order, 1));

        double rhs[MAX_DIM];
        for (uint64_t i = 0; i < s.dim; ++i)
            rhs[i] = rng_next(&r);

        hybsol_decomposition_t *plain = factorize(s.sys, 1);
        hybsol_decomposition_t *shuffled = factorize(reordered, 0);
        CHECK(plain != NULL && shuffled != NULL);
        if (plain == NULL || shuffled == NULL)
        {
            hybsol_decomposition_destroy(plain);
            hybsol_decomposition_destroy(shuffled);
            hybsol_system_destroy(reordered);
            random_system_destroy(&s);
            return;
        }

        double reference[MAX_DIM], actual[MAX_DIM], permuted_rhs[MAX_DIM];
        memcpy(reference, rhs, sizeof(double) * s.dim);
        CHECK_OK(hybsol_decomposition_solve(plain, reference, 1));

        hybsol_system_reorder_vector(reordered, order, rhs, permuted_rhs);
        CHECK_OK(hybsol_decomposition_solve(shuffled, permuted_rhs, 0));

        hybsol_decomposition_destroy(plain);
        hybsol_decomposition_destroy(shuffled);

        // `permuted_rhs` now holds the reordered solution.
        hybsol_system_unorder_vector(reordered, order, permuted_rhs, actual);
        // Different elimination orders, so the comparison has to be relative.
        for (uint64_t i = 0; i < s.dim; ++i)
            CHECK_NEAR(actual[i], reference[i], 1e-7 * (1.0 + fabs(reference[i])));

        hybsol_system_destroy(reordered);
        random_system_destroy(&s);
    }
}

static void test_reorder_and_vector_shuffle(void)
{
    // Two coupled blocks need two colors, so a bounded coloring can actually fail here.
    const uint64_t sizes[] = {1, 1};
    hybsol_system_t *sys = NULL;
    CHECK_OK(hybsol_system_create(2, sizes, &sys, &CUTL_STD_ALLOCATOR));

    double one[1] = {1.0}, two[1] = {2.0};
    CHECK_OK(hybsol_system_add_block(sys, 0, 0, 1, 1, one));
    CHECK_OK(hybsol_system_add_block(sys, 1, 1, 1, 1, one));
    CHECK_OK(hybsol_system_add_block(sys, 0, 1, 1, 1, two));
    CHECK_OK(hybsol_system_add_block(sys, 1, 0, 1, 1, two));

    double vector[2] = {1.0, 2.0};
    double out[2] = {0};
    uint64_t order[2] = {0, 1};

    CHECK_OK(hybsol_system_compute_reordering(sys, HYBSOL_ORDERING_FIRST, 2, order));
    CHECK_RESULT(hybsol_system_compute_reordering(sys, HYBSOL_ORDERING_FIRST, 1, order), HYBSOL_ERROR_MAX_COLORS);

    hybsol_system_reorder_vector(sys, order, vector, out);
    CHECK(out[0] == 1.0 && out[1] == 2.0);

    hybsol_system_destroy(sys);
}

int main(void)
{
    test_ordering_is_a_permutation();
    test_coloring_property();
    test_reorder_roundtrip();
    test_reordered_system_solves_consistently();
    test_reorder_and_vector_shuffle();
    return test_report("test_ordering");
}
