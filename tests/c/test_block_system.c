/** @file test_block_system.c
 * Construction, queries, deep copies and the structural validity check.
 */

#include "test_util.h"

#include <hybsol/hybsol.h>

#include <string.h>

#define MAX_DIM 8

/** Deterministic, non-singular ``n x n`` pattern. */
static void fill_sequential(const uint64_t n, double mat[MAX_DIM][MAX_DIM])
{
    uint64_t k = 0;
    for (uint64_t i = 0; i < n; ++i)
    {
        for (uint64_t j = 0; j < n; ++j)
            mat[i][j] = (double)(k++ % 7) * 0.25 - 1.0;
    }
    for (uint64_t i = 0; i < n; ++i)
        mat[i][i] += 10.0;
}

/** Copy the block at ``(row_off, col_off)`` of ``src`` into ``dst``. */
static void extract_block(const double src[MAX_DIM][MAX_DIM], const uint64_t row_off, const uint64_t col_off,
                          const uint64_t rows, const uint64_t cols, double *const dst)
{
    for (uint64_t i = 0; i < rows; ++i)
    {
        for (uint64_t j = 0; j < cols; ++j)
            dst[i * cols + j] = src[row_off + i][col_off + j];
    }
}

/** Add every block of ``n x n`` matrix ``src`` to ``sys``. */
static hybsol_result_t add_dense_blocks(hybsol_system_t *const sys, const uint64_t n,
                                        const double src[MAX_DIM][MAX_DIM], const uint64_t *const sizes)
{
    uint64_t offsets[MAX_DIM + 1] = {0}, n_blocks = 0;
    for (uint64_t i = 0; i < n; ++i)
    {
        if (sizes[i] == 0)
            break;
        offsets[i + 1] = offsets[i] + sizes[i];
        n_blocks += 1;
    }

    double block[MAX_DIM * MAX_DIM];
    for (uint64_t i = 0; i < n_blocks; ++i)
    {
        for (uint64_t j = 0; j < n_blocks; ++j)
        {
            extract_block(src, offsets[i], offsets[j], sizes[i], sizes[j], block);
            const hybsol_result_t res = hybsol_system_add_block(sys, i, j, sizes[i], sizes[j], block);
            if (res != HYBSOL_SUCCESS)
                return res;
        }
    }
    return HYBSOL_SUCCESS;
}

static void test_create_and_query(void)
{
    const uint64_t sizes[] = {2, 3, 1};
    hybsol_system_t *sys = NULL;
    CHECK_OK(hybsol_system_create(3, sizes, &sys, &CUTL_STD_ALLOCATOR));

    CHECK(hybsol_system_n_blocks(sys) == 3);
    CHECK(hybsol_system_total_size(sys) == 6);
    CHECK(hybsol_system_block_size(sys, 0) == 2);
    CHECK(hybsol_system_block_size(sys, 1) == 3);
    CHECK(hybsol_system_block_size(sys, 2) == 1);
    CHECK(hybsol_system_block_size(sys, 3) == 0);

    const uint64_t *const offsets = hybsol_system_block_offsets(sys);
    CHECK(offsets[0] == 0 && offsets[1] == 2 && offsets[2] == 5 && offsets[3] == 6);

    // An empty system is structurally invalid: no row has its diagonal.
    CHECK(!hybsol_system_is_valid(sys));

    hybsol_system_destroy(sys);
}

static void test_rows_and_blocks(void)
{
    const uint64_t sizes[] = {1, 1, 1};
    hybsol_system_t *sys = NULL;
    CHECK_OK(hybsol_system_create(3, sizes, &sys, &CUTL_STD_ALLOCATOR));

    double two[1] = {2.0}, three[1] = {3.0}, one[1] = {1.0};
    CHECK_OK(hybsol_system_add_block(sys, 0, 0, 1, 1, one));
    CHECK_OK(hybsol_system_add_block(sys, 1, 0, 1, 1, two));
    CHECK_OK(hybsol_system_add_block(sys, 1, 1, 1, 1, one));
    CHECK_OK(hybsol_system_add_block(sys, 1, 2, 1, 1, three));
    CHECK_OK(hybsol_system_add_block(sys, 2, 0, 1, 1, two));
    CHECK_OK(hybsol_system_add_block(sys, 2, 2, 1, 1, one));

    CHECK(hybsol_system_row_count(sys, 1) == 3);
    CHECK(hybsol_system_row_count(sys, 0) == 1);
    CHECK(hybsol_system_has_block(sys, 1, 1));
    CHECK(hybsol_system_has_block(sys, 1, 2));
    CHECK(!hybsol_system_has_block(sys, 2, 1));
    CHECK(!hybsol_system_has_block(sys, 3, 0));

    uint64_t indices[4] = {0};
    uint64_t written = 0;
    CHECK_OK(hybsol_system_row_indices(sys, 1, indices, 4, &written));
    CHECK(written == 3);
    CHECK(indices[0] == 0 && indices[1] == 1 && indices[2] == 2);

    uint64_t col = 0;
    CHECK_OK(hybsol_system_first_column(sys, 1, &col));
    CHECK(col == 0);
    CHECK_OK(hybsol_system_first_column(sys, 0, &col));
    CHECK(col == 0);

    CHECK_OK(hybsol_system_next_column(sys, 1, 1, &col));
    CHECK(col == 2);
    CHECK_RESULT(hybsol_system_next_column(sys, 1, 2, &col), HYBSOL_ERROR_NO_MORE_COLUMNS);

    hybsol_matrix_t block;
    CHECK_OK(hybsol_system_get_block(sys, 1, 2, &block));
    CHECK(block.rows == 1 && block.cols == 1 && block.data[0] == 3.0);

    // Rows 1 and 2 have blocks in column 0; row 0 has only its diagonal.
    uint8_t flags[3] = {0, 0, 0};
    hybsol_system_no_lower_connections(sys, flags);
    CHECK(flags[0] == 1 && flags[1] == 0 && flags[2] == 0);

    hybsol_system_destroy(sys);
}

static void test_add_block_accumulates(void)
{
    const uint64_t sizes[] = {2};
    hybsol_system_t *sys = NULL;
    CHECK_OK(hybsol_system_create(1, sizes, &sys, &CUTL_STD_ALLOCATOR));

    double first[4] = {1.0, 2.0, 3.0, 4.0};
    double second[4] = {10.0, 20.0, 30.0, 40.0};
    CHECK_OK(hybsol_system_add_block(sys, 0, 0, 2, 2, first));
    CHECK_OK(hybsol_system_add_block(sys, 0, 0, 2, 2, second));

    hybsol_matrix_t block;
    CHECK_OK(hybsol_system_get_block(sys, 0, 0, &block));
    CHECK(block.data[0] == 11.0 && block.data[1] == 22.0);
    CHECK(block.data[2] == 33.0 && block.data[3] == 44.0);

    hybsol_system_destroy(sys);
}

static void test_dense_export(void)
{
    const uint64_t sizes[] = {2, 1};
    hybsol_system_t *sys = NULL;
    CHECK_OK(hybsol_system_create(2, sizes, &sys, &CUTL_STD_ALLOCATOR));

    double full[MAX_DIM][MAX_DIM] = {{0}};
    fill_sequential(3, full);
    CHECK_OK(add_dense_blocks(sys, 2, full, sizes));

    // `to_dense` writes rows back to back, so the reference uses that stride.
    double out[3 * 3] = {0}, reference[3 * 3] = {0};
    for (uint64_t i = 0; i < 3; ++i)
    {
        for (uint64_t j = 0; j < 3; ++j)
            reference[i * 3 + j] = full[i][j];
    }
    CHECK_OK(hybsol_system_to_dense(sys, out));
    CHECK(memcmp(out, reference, sizeof(reference)) == 0);

    hybsol_system_destroy(sys);
}

static void test_copy(void)
{
    const uint64_t sizes[] = {2, 2};
    hybsol_system_t *sys = NULL;
    CHECK_OK(hybsol_system_create(2, sizes, &sys, &CUTL_STD_ALLOCATOR));

    double full[MAX_DIM][MAX_DIM] = {{0}};
    fill_sequential(4, full);
    CHECK_OK(add_dense_blocks(sys, 2, full, sizes));

    hybsol_system_t *dup = NULL;
    CHECK_OK(hybsol_system_copy(sys, &dup));

    double a[4 * 4] = {0}, b[4 * 4] = {0};
    CHECK_OK(hybsol_system_to_dense(sys, a));
    CHECK_OK(hybsol_system_to_dense(dup, b));
    CHECK(memcmp(a, b, sizeof(a)) == 0);

    // The copy is independent of the original.
    double again[MAX_DIM][MAX_DIM] = {{0}};
    fill_sequential(4, again);
    CHECK_OK(add_dense_blocks(dup, 2, again, sizes));
    CHECK_OK(hybsol_system_to_dense(sys, a));
    CHECK_OK(hybsol_system_to_dense(dup, b));
    CHECK(a[0] != b[0]);

    hybsol_system_destroy(dup);
    hybsol_system_destroy(sys);
}

static void test_is_valid(void)
{
    const uint64_t sizes[] = {1, 1, 1};
    double one[1] = {1.0};

    hybsol_system_t *sys = NULL;
    CHECK_OK(hybsol_system_create(3, sizes, &sys, &CUTL_STD_ALLOCATOR));

    // A row that only has entries below the diagonal is not valid either.
    CHECK_OK(hybsol_system_add_block(sys, 0, 0, 1, 1, one));
    CHECK_OK(hybsol_system_add_block(sys, 1, 0, 1, 1, one));
    CHECK(!hybsol_system_is_valid(sys));

    // Adding the mirror block repairs the structure.
    CHECK_OK(hybsol_system_add_block(sys, 0, 1, 1, 1, one));
    CHECK_OK(hybsol_system_add_block(sys, 1, 1, 1, 1, one));
    CHECK_OK(hybsol_system_add_block(sys, 2, 2, 1, 1, one));
    CHECK(hybsol_system_is_valid(sys));

    // ... and a row without any diagonal block is still invalid.
    hybsol_system_t *other = NULL;
    CHECK_OK(hybsol_system_create(2, sizes, &other, &CUTL_STD_ALLOCATOR));
    CHECK_OK(hybsol_system_add_block(other, 0, 1, 1, 1, one));
    CHECK_OK(hybsol_system_add_block(other, 1, 0, 1, 1, one));
    CHECK(!hybsol_system_is_valid(other));

    hybsol_system_destroy(other);
    hybsol_system_destroy(sys);
}

static void test_row_operations(void)
{
    const uint64_t sizes[] = {2, 2};
    hybsol_system_t *sys = NULL;
    CHECK_OK(hybsol_system_create(2, sizes, &sys, &CUTL_STD_ALLOCATOR));

    double full[MAX_DIM][MAX_DIM] = {{0}};
    fill_sequential(4, full);
    CHECK_OK(add_dense_blocks(sys, 2, full, sizes));

    // multiply_row(0, identity, start=1) must leave the row untouched.
    double identity[4] = {1.0, 0.0, 0.0, 1.0};
    hybsol_matrix_t id = hybsol_matrix_view(2, 2, identity);
    CHECK_OK(hybsol_system_multiply_row(sys, 0, 1, &id));

    double after[4 * 4] = {0};
    CHECK_OK(hybsol_system_to_dense(sys, after));
    for (uint64_t i = 0; i < 4; ++i)
    {
        for (uint64_t j = 0; j < 4; ++j)
            CHECK_NEAR(after[i * 4 + j], full[i][j], 0.0);
    }

    // eliminate_row_with reproduces `target - block(tgt,src) @ src`.
    double factor[4];
    extract_block(full, 2, 0, 2, 2, factor);
    hybsol_matrix_t multiplier = hybsol_matrix_view(2, 2, factor);
    CHECK_OK(hybsol_system_eliminate_row_with(sys, 1, 0, &multiplier));

    double expected[4 * 4];
    for (uint64_t i = 0; i < 4; ++i)
    {
        for (uint64_t j = 0; j < 4; ++j)
            expected[i * 4 + j] = full[i][j];
    }
    for (uint64_t i = 0; i < 2; ++i)
    {
        for (uint64_t j = 0; j < 2; ++j)
        {
            double acc = 0.0;
            for (uint64_t k = 0; k < 2; ++k)
                acc += factor[i * 2 + k] * full[k][2 + j];
            expected[(2 + i) * 4 + (2 + j)] = full[2 + i][2 + j] - acc;
        }
    }

    CHECK_OK(hybsol_system_to_dense(sys, after));
    for (uint64_t i = 0; i < 16; ++i)
        CHECK_NEAR(after[i], expected[i], 1e-12);

    hybsol_system_destroy(sys);
}

/** First use allocates a zeroed view; later calls hand back the same storage. */
static void test_block_storage_creates_and_reuses(void)
{
    hybsol_system_t *sys = NULL;
    const uint64_t sizes[3] = {2, 3, 1};
    CHECK_OK(hybsol_system_create(3, sizes, &sys, &CUTL_STD_ALLOCATOR));

    hybsol_matrix_t view;
    CHECK_OK(hybsol_system_block_storage(sys, 0, 1, &view));
    CHECK(view.rows == 2 && view.cols == 3);
    CHECK(hybsol_system_has_block(sys, 0, 1));
    for (uint64_t i = 0; i < 6; ++i)
        CHECK(view.data[i] == 0.0);

    // Writes through the view land in the system.
    for (uint64_t i = 0; i < 6; ++i)
        view.data[i] = (double)(i + 1);

    hybsol_matrix_t read_back;
    CHECK_OK(hybsol_system_get_block(sys, 0, 1, &read_back));
    CHECK(read_back.data == view.data);
    CHECK(read_back.data[5] == 6.0);

    // Re-fetching must not clear or accumulate, so an assembler can return to the same block.
    hybsol_matrix_t again;
    CHECK_OK(hybsol_system_block_storage(sys, 0, 1, &again));
    CHECK(again.data == view.data);
    CHECK(again.rows == view.rows && again.cols == view.cols);
    for (uint64_t i = 0; i < 6; ++i)
        CHECK(again.data[i] == (double)(i + 1));

    // Inserting a block earlier in the row moves pointers, not entries.
    hybsol_matrix_t first;
    CHECK_OK(hybsol_system_block_storage(sys, 0, 0, &first));
    double *const pinned = first.data;
    CHECK_OK(hybsol_system_add_block(sys, 0, 2, 2, 1, (const double[2]){1.0, 2.0}));
    CHECK(hybsol_system_row_count(sys, 0) == 3);
    CHECK(first.data == pinned);
    CHECK(view.data[0] == 1.0 && view.data[5] == 6.0);
    for (uint64_t i = 0; i < 2; ++i)
        CHECK(first.data[i] == 0.0);

    // The diagonal block does not exist yet, so storage has to create it.
    hybsol_matrix_t diagonal;
    CHECK_OK(hybsol_system_block_storage(sys, 2, 2, &diagonal));
    CHECK(diagonal.rows == 1 && diagonal.cols == 1);
    CHECK(diagonal.data[0] == 0.0);

    hybsol_system_destroy(sys);
}

/** Touching the diagonal through storage drops the cached factorization. */
static void test_block_storage_invalidates_diagonal(void)
{
    hybsol_system_t *sys = NULL;
    const uint64_t sizes[2] = {2, 2};
    const double diag[4] = {4.0, 0.0, 0.0, 4.0};
    CHECK_OK(hybsol_system_create(2, sizes, &sys, &CUTL_STD_ALLOCATOR));
    CHECK_OK(hybsol_system_add_block(sys, 0, 0, 2, 2, diag));

    CHECK_OK(hybsol_system_decompose_diagonal(sys, 0));
    CHECK_OK(hybsol_system_apply_diagonal_inverse(sys, 0));

    hybsol_matrix_t view;
    CHECK_OK(hybsol_system_block_storage(sys, 0, 0, &view));
    CHECK_RESULT(hybsol_system_apply_diagonal_inverse(sys, 0), HYBSOL_ERROR_NOT_DECOMPOSED);

    hybsol_system_destroy(sys);
}

/** Storage may not be created once the system has been factorized. */
static void test_block_storage_rejects_decomposed(void)
{
    hybsol_system_t *sys = NULL;
    const uint64_t sizes[1] = {2};
    const double diag[4] = {4.0, 0.0, 0.0, 4.0};
    CHECK_OK(hybsol_system_create(1, sizes, &sys, &CUTL_STD_ALLOCATOR));
    CHECK_OK(hybsol_system_add_block(sys, 0, 0, 2, 2, diag));
    CHECK_OK(hybsol_system_decompose(sys, 1));

    hybsol_matrix_t view;
    CHECK_RESULT(hybsol_system_block_storage(sys, 0, 0, &view), HYBSOL_ERROR_ALREADY_DECOMPOSED);

    hybsol_system_destroy(sys);
}

int main(void)
{
    test_create_and_query();
    test_rows_and_blocks();
    test_add_block_accumulates();
    test_dense_export();
    test_copy();
    test_is_valid();
    test_row_operations();
    test_block_storage_creates_and_reuses();
    test_block_storage_invalidates_diagonal();
    test_block_storage_rejects_decomposed();
    return test_report("test_block_system");
}
