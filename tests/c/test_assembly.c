/** @file test_assembly.c
 * The bulk assembly path and its agreement with block-by-block assembly.
 */

#include "test_util.h"

#include <hybsol/hybsol.h>

#include <string.h>

#define MAX_BLOCKS 8
#define MAX_SIZE 4
#define MAX_ENTRIES 64
#define MAX_DIM (MAX_BLOCKS * MAX_SIZE)

/** A deterministic generator, so failures are reproducible. */
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

/** Flat COO description of a random, structurally valid system. */
typedef struct
{
    uint64_t n_blocks;
    uint64_t sizes[MAX_BLOCKS];
    uint64_t n_entries;
    uint64_t rows[MAX_ENTRIES];
    uint64_t cols[MAX_ENTRIES];
    /** Start of block ``k`` inside ``data``; blocks are packed, as
     * :c:func:`hybsol_system_add_blocks` expects. */
    uint64_t data_offset[MAX_ENTRIES];
    uint64_t data_end;
    double data[MAX_ENTRIES * MAX_SIZE * MAX_SIZE];
    double dense[MAX_BLOCKS * MAX_SIZE][MAX_BLOCKS * MAX_SIZE];
} pattern_t;

/** Store one block of ``p`` and mirror it into the dense reference. */
static void pattern_store(pattern_t *const p, const uint64_t *const offsets, const uint64_t row, const uint64_t col,
                          const double *const values)
{
    const uint64_t n_rows = p->sizes[row], n_cols = p->sizes[col];
    double *const slot = p->data + p->data_end;

    for (uint64_t a = 0; a < n_rows; ++a)
    {
        for (uint64_t b = 0; b < n_cols; ++b)
        {
            slot[a * n_cols + b] = values[a * n_cols + b];
            p->dense[offsets[row] + a][offsets[col] + b] = values[a * n_cols + b];
        }
    }
    p->data_offset[p->n_entries] = p->data_end;
    p->data_end += n_rows * n_cols;
    p->rows[p->n_entries] = row;
    p->cols[p->n_entries] = col;
    p->n_entries += 1;
}

static void build_pattern(rng_t *const r, pattern_t *const p)
{
    p->n_blocks = 1 + rng_below(r, MAX_BLOCKS);
    uint64_t offsets[MAX_BLOCKS + 1] = {0};
    for (uint64_t i = 0; i < p->n_blocks; ++i)
    {
        p->sizes[i] = 1 + rng_below(r, MAX_SIZE);
        offsets[i + 1] = offsets[i] + p->sizes[i];
    }

    memset(p->dense, 0, sizeof(p->dense));
    p->n_entries = 0;
    p->data_end = 0;

    for (uint64_t i = 0; i < p->n_blocks; ++i)
    {
        // The diagonal block is always present.
        double block[MAX_SIZE * MAX_SIZE];
        const uint64_t n = p->sizes[i];
        for (uint64_t k = 0; k < n * n; ++k)
            block[k] = rng_next(r) - 0.5;
        for (uint64_t k = 0; k < n; ++k)
            block[k * n + k] += 4.0;
        pattern_store(p, offsets, i, i, block);

        // Off-diagonal pairs are kept or dropped together, so the pattern stays symmetric.
        for (uint64_t j = i + 1; j < p->n_blocks; ++j)
        {
            if (rng_next(r) < 0.5)
                continue;

            const uint64_t rows = p->sizes[i], cols = p->sizes[j];
            for (uint64_t a = 0; a < rows; ++a)
            {
                for (uint64_t b = 0; b < cols; ++b)
                    block[a * cols + b] = rng_next(r) - 0.5;
            }
            pattern_store(p, offsets, i, j, block);

            double mirror[MAX_SIZE * MAX_SIZE];
            for (uint64_t a = 0; a < cols; ++a)
            {
                for (uint64_t b = 0; b < rows; ++b)
                    mirror[a * rows + b] = block[b * cols + a];
            }
            pattern_store(p, offsets, j, i, mirror);
        }
    }
}

static void test_bulk_matches_add_block(void)
{
    rng_t r = {.state = 12345};
    for (int trial = 0; trial < 40; ++trial)
    {
        pattern_t p;
        build_pattern(&r, &p);
        CHECK(p.n_entries <= MAX_ENTRIES);

        hybsol_system_t *bulk = NULL;
        hybsol_system_t *single = NULL;
        CHECK_OK(hybsol_system_create(p.n_blocks, p.sizes, &bulk, &CUTL_STD_ALLOCATOR));
        CHECK_OK(hybsol_system_create(p.n_blocks, p.sizes, &single, &CUTL_STD_ALLOCATOR));

        CHECK_OK(hybsol_system_add_blocks(bulk, p.n_entries, p.rows, p.cols, p.data));
        CHECK(hybsol_system_is_valid(bulk));

        for (uint64_t k = 0; k < p.n_entries; ++k)
        {
            CHECK_OK(hybsol_system_add_block(single, p.rows[k], p.cols[k], p.sizes[p.rows[k]], p.sizes[p.cols[k]],
                                             p.data + p.data_offset[k]));
        }

        // Both paths must agree with each other and with the reference the entries were drawn from.
        const uint64_t total = hybsol_system_total_size(bulk);
        double a[MAX_DIM * MAX_DIM] = {0}, b[MAX_DIM * MAX_DIM] = {0};
        CHECK_OK(hybsol_system_to_dense(bulk, a));
        CHECK_OK(hybsol_system_to_dense(single, b));

        for (uint64_t i = 0; i < total; ++i)
        {
            for (uint64_t j = 0; j < total; ++j)
            {
                CHECK_NEAR(a[i * total + j], b[i * total + j], 0.0);
                CHECK_NEAR(a[i * total + j], p.dense[i][j], 1e-12);
            }
        }

        hybsol_system_destroy(bulk);
        hybsol_system_destroy(single);
    }
}

static void test_bulk_accumulates_duplicates(void)
{
    const uint64_t sizes[] = {2};
    hybsol_system_t *sys = NULL;
    CHECK_OK(hybsol_system_create(1, sizes, &sys, &CUTL_STD_ALLOCATOR));

    const uint64_t rows[] = {0, 0};
    const uint64_t cols[] = {0, 0};
    const double data[] = {1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0};

    CHECK_OK(hybsol_system_add_blocks(sys, 2, rows, cols, data));

    hybsol_matrix_t block;
    CHECK_OK(hybsol_system_get_block(sys, 0, 0, &block));
    CHECK(block.data[0] == 11.0 && block.data[1] == 22.0);
    CHECK(block.data[2] == 33.0 && block.data[3] == 44.0);

    hybsol_system_destroy(sys);
}

static void test_reserve(void)
{
    const uint64_t sizes[] = {1, 1, 1};
    hybsol_system_t *sys = NULL;
    CHECK_OK(hybsol_system_create(3, sizes, &sys, &CUTL_STD_ALLOCATOR));

    const double one[1] = {1.0};
    CHECK_OK(hybsol_system_reserve(sys, 1, 3));
    CHECK_OK(hybsol_system_add_block(sys, 1, 1, 1, 1, one));
    CHECK_OK(hybsol_system_add_block(sys, 1, 2, 1, 1, one));
    CHECK(hybsol_system_row_count(sys, 1) == 2);

    hybsol_system_destroy(sys);
}

int main(void)
{
    test_bulk_matches_add_block();
    test_bulk_accumulates_duplicates();
    test_reserve();
    return test_report("test_assembly");
}
