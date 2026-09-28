/** @file core/block_system.c
 * Construction, inspection, assembly and row operations of a block system.
 */

#include "internal.h"

/* ------------------------------------------------------------------------- */
/* Lookup helpers                                                            */
/* ------------------------------------------------------------------------- */

uint64_t hybsol_row_find_geq(const hybsol_row_t *const row, const uint64_t val)
{
    const uint64_t size = row->count;
    hybsol_row_entry_t *const *const array = row->entries;

    if (size == 0)
        return 0;

    // Quickly check the extremes
    if (array[0]->col >= val)
        return 0;

    if (array[size - 1]->col < val)
        return size;

    // Use binary search until we are down to 8 values
    uint64_t lo = 0, len = size;

    while (len > 8)
    {
        const uint64_t pivot = lo + len / 2;
        const uint64_t col = array[pivot]->col;
        if (col == val)
            return pivot;

        if (col > val)
        {
            len = pivot - lo + 1;
        }
        else
        {
            len = len - (pivot - lo);
            lo = pivot;
        }
    }

    HYBSOL_ASSERT(lo + len <= size, "Internal error: lo + len > size (%llu + %llu <= %llu).", (unsigned long long)lo,
                  (unsigned long long)len, (unsigned long long)size);
    for (uint64_t idx = lo; idx < lo + len; ++idx)
    {
        if (array[idx]->col >= val)
            return idx;
    }

    return size;
}

int hybsol_row_find(const hybsol_row_t *const row, const uint64_t col, uint64_t *const out)
{
    const uint64_t idx = hybsol_row_find_geq(row, col);
    if (idx == row->count || row->entries[idx]->col != col)
        return 0;
    if (out)
        *out = idx;
    return 1;
}

hybsol_result_t hybsol_row_reserve(hybsol_row_t *const row, const uint64_t needed)
{
    if (needed <= row->capacity)
        return HYBSOL_SUCCESS;

    uint64_t new_capacity = row->capacity ? row->capacity : 8;
    while (new_capacity < needed)
    {
        if (new_capacity > UINT64_MAX / 2)
        {
            new_capacity = needed;
            break;
        }
        new_capacity *= 2;
    }

    if (new_capacity > SIZE_MAX / sizeof(*row->entries))
        return HYBSOL_ERROR_OUT_OF_MEMORY;

    hybsol_row_entry_t **const ptr = hybsol_grow(row->entries, (size_t)new_capacity * sizeof(*ptr));
    if (!ptr)
        return HYBSOL_ERROR_OUT_OF_MEMORY;

    for (uint64_t i = row->capacity; i < new_capacity; ++i)
        ptr[i] = NULL;

    row->entries = ptr;
    row->capacity = new_capacity;
    return HYBSOL_SUCCESS;
}

hybsol_result_t hybsol_require_mutable(const hybsol_system_t *const sys)
{
    return sys->decomposed ? HYBSOL_ERROR_ALREADY_DECOMPOSED : HYBSOL_SUCCESS;
}

hybsol_result_t hybsol_check_index(const hybsol_system_t *const sys, const uint64_t idx)
{
    return idx < sys->n ? HYBSOL_SUCCESS : HYBSOL_ERROR_INDEX_OUT_OF_RANGE;
}

void hybsol_invalidate_diagonal(hybsol_system_t *const sys, const uint64_t idx)
{
    if (idx < sys->n)
        sys->diag_decomposed[idx] = 0;
}

/* ------------------------------------------------------------------------- */
/* Lifetime                                                                   */
/* ------------------------------------------------------------------------- */

static void free_row(hybsol_row_t *const row)
{
    for (uint64_t i = 0; i < row->count; ++i)
    {
        hybsol_free(row->entries[i]);
        row->entries[i] = NULL;
    }
    hybsol_free(row->entries);
    row->entries = NULL;
    row->count = 0;
    row->capacity = 0;
}

hybsol_result_t hybsol_system_create(const uint64_t n_blocks, const uint64_t block_sizes[static n_blocks],
                                     hybsol_system_t **const out)
{
    if (out == NULL)
        return HYBSOL_ERROR_INVALID_ARGUMENT;
    *out = NULL;
    if (n_blocks == 0)
        return HYBSOL_ERROR_INVALID_ARGUMENT;

    hybsol_system_t *const sys = hybsol_alloc(sizeof(*sys));
    if (sys == NULL)
        return HYBSOL_ERROR_OUT_OF_MEMORY;
    *sys = (hybsol_system_t){0};
    sys->n = n_blocks;

    sys->block_offsets = hybsol_alloc(sizeof(*sys->block_offsets) * (size_t)(n_blocks + 1));
    sys->rows = hybsol_alloc(sizeof(*sys->rows) * (size_t)n_blocks);
    sys->diag_decomposed = hybsol_alloc(sizeof(*sys->diag_decomposed) * (size_t)n_blocks);
    if (sys->block_offsets == NULL || sys->rows == NULL || sys->diag_decomposed == NULL)
    {
        hybsol_system_destroy(sys);
        return HYBSOL_ERROR_OUT_OF_MEMORY;
    }

    sys->block_offsets[0] = 0;
    for (uint64_t i = 0; i < n_blocks; ++i)
    {
        if (block_sizes[i] == 0)
        {
            hybsol_system_destroy(sys);
            return HYBSOL_ERROR_INVALID_ARGUMENT;
        }
        sys->block_offsets[i + 1] = sys->block_offsets[i] + block_sizes[i];
        sys->rows[i] = (hybsol_row_t){0};
        sys->diag_decomposed[i] = 0;
    }

    *out = sys;
    return HYBSOL_SUCCESS;
}

void hybsol_system_destroy(hybsol_system_t *const sys)
{
    if (sys == NULL)
        return;

    if (sys->rows != NULL)
    {
        for (uint64_t i = 0; i < sys->n; ++i)
            free_row(sys->rows + i);
    }

    hybsol_free(sys->rows);
    hybsol_free(sys->block_offsets);
    hybsol_free(sys->diag_decomposed);
    hybsol_free(sys->ops);
    hybsol_free(sys);
}

hybsol_result_t hybsol_system_copy(const hybsol_system_t *const sys, hybsol_system_t **const out)
{
    if (out == NULL)
        return HYBSOL_ERROR_INVALID_ARGUMENT;
    *out = NULL;

    hybsol_system_t *const dst = hybsol_alloc(sizeof(*dst));
    if (dst == NULL)
        return HYBSOL_ERROR_OUT_OF_MEMORY;
    *dst = (hybsol_system_t){0};
    dst->n = sys->n;
    dst->decomposed = sys->decomposed;
    dst->n_ops = sys->n_ops;
    dst->ops_capacity = sys->n_ops;

    dst->block_offsets = hybsol_alloc(sizeof(*dst->block_offsets) * (size_t)(sys->n + 1));
    dst->rows = hybsol_alloc(sizeof(*dst->rows) * (size_t)sys->n);
    dst->diag_decomposed = hybsol_alloc(sizeof(*dst->diag_decomposed) * (size_t)sys->n);
    if (dst->block_offsets == NULL || dst->rows == NULL || dst->diag_decomposed == NULL)
    {
        hybsol_system_destroy(dst);
        return HYBSOL_ERROR_OUT_OF_MEMORY;
    }

    memcpy(dst->block_offsets, sys->block_offsets, sizeof(*dst->block_offsets) * (size_t)(sys->n + 1));
    memcpy(dst->diag_decomposed, sys->diag_decomposed, sizeof(*dst->diag_decomposed) * (size_t)sys->n);
    for (uint64_t i = 0; i < sys->n; ++i)
        dst->rows[i] = (hybsol_row_t){0};

    for (uint64_t i = 0; i < sys->n; ++i)
    {
        const hybsol_row_t *const src = sys->rows + i;
        hybsol_row_t *const tgt = dst->rows + i;

        if (src->count == 0)
            continue;

        tgt->entries = hybsol_alloc(sizeof(*tgt->entries) * (size_t)src->count);
        if (tgt->entries == NULL)
        {
            hybsol_system_destroy(dst);
            return HYBSOL_ERROR_OUT_OF_MEMORY;
        }
        tgt->capacity = src->count;
        for (uint64_t j = 0; j < src->count; ++j)
            tgt->entries[j] = NULL;

        for (uint64_t j = 0; j < src->count; ++j)
        {
            const hybsol_row_entry_t *const entry = src->entries[j];
            const uint64_t n_values = hybsol_block_size(sys, i) * hybsol_block_size(sys, entry->col);

            hybsol_row_entry_t *const copy = hybsol_alloc(sizeof(*copy) + (size_t)n_values * sizeof(*copy->vals));
            if (copy == NULL)
            {
                hybsol_system_destroy(dst);
                return HYBSOL_ERROR_OUT_OF_MEMORY;
            }
            copy->col = entry->col;
            memcpy(copy->vals, entry->vals, (size_t)n_values * sizeof(*copy->vals));
            tgt->entries[j] = copy;
            tgt->count += 1;
        }
    }

    if (sys->n_ops > 0)
    {
        dst->ops = hybsol_alloc(sizeof(*dst->ops) * (size_t)sys->n_ops);
        if (dst->ops == NULL)
        {
            hybsol_system_destroy(dst);
            return HYBSOL_ERROR_OUT_OF_MEMORY;
        }
        memcpy(dst->ops, sys->ops, sizeof(*dst->ops) * (size_t)sys->n_ops);
    }

    *out = dst;
    return HYBSOL_SUCCESS;
}

/* ------------------------------------------------------------------------- */
/* Queries                                                                    */
/* ------------------------------------------------------------------------- */

uint64_t hybsol_system_n_blocks(const hybsol_system_t *const sys)
{
    return sys->n;
}

uint64_t hybsol_system_total_size(const hybsol_system_t *const sys)
{
    return sys->block_offsets[sys->n];
}

uint64_t hybsol_system_block_size(const hybsol_system_t *const sys, const uint64_t idx)
{
    if (idx >= sys->n)
        return 0;
    return hybsol_block_size(sys, idx);
}

const uint64_t *hybsol_system_block_offsets(const hybsol_system_t *const sys)
{
    return sys->block_offsets;
}

int hybsol_system_is_valid(const hybsol_system_t *const sys)
{
    for (uint64_t i = 0; i < sys->n; ++i)
    {
        const hybsol_row_t *const row = sys->rows + i;
        if (row->count == 0)
            return 0;

        int has_diagonal = 0;
        for (uint64_t j = 0; j < row->count; ++j)
        {
            const hybsol_row_entry_t *const entry = row->entries[j];
            if (entry->col == i)
            {
                // Entries are sorted, so the diagonal ends the interesting part of the row
                has_diagonal = 1;
                break;
            }

            if (entry->col > i)
            {
                // We walked past the diagonal without finding it
                return 0;
            }

            // Below the diagonal: there must be a mirrored block
            const hybsol_row_t *const other = sys->rows + entry->col;
            if (!hybsol_row_find(other, i, NULL))
                return 0;
        }

        if (!has_diagonal)
            return 0;
    }

    return 1;
}

uint64_t hybsol_system_row_count(const hybsol_system_t *const sys, const uint64_t row)
{
    if (row >= sys->n)
        return 0;
    return sys->rows[row].count;
}

hybsol_result_t hybsol_system_row_indices(const hybsol_system_t *const sys, const uint64_t row, uint64_t *const out,
                                          const uint64_t capacity, uint64_t *const n_written)
{
    hybsol_result_t res = hybsol_check_index(sys, row);
    if (res != HYBSOL_SUCCESS)
        return res;

    const hybsol_row_t *const r = sys->rows + row;
    if (n_written)
        *n_written = r->count;
    if (out == NULL || capacity < r->count)
        return HYBSOL_ERROR_INVALID_ARGUMENT;

    for (uint64_t i = 0; i < r->count; ++i)
        out[i] = r->entries[i]->col;

    return HYBSOL_SUCCESS;
}

int hybsol_system_has_block(const hybsol_system_t *const sys, const uint64_t row, const uint64_t col)
{
    if (row >= sys->n || col >= sys->n)
        return 0;
    return hybsol_row_find(sys->rows + row, col, NULL);
}

hybsol_result_t hybsol_system_get_block(hybsol_system_t *const sys, const uint64_t row, const uint64_t col,
                                        hybsol_matrix_t *const out)
{
    return hybsol_lookup_block(sys, row, col, out);
}

hybsol_result_t hybsol_system_first_column(const hybsol_system_t *const sys, const uint64_t row, uint64_t *const out)
{
    hybsol_result_t res = hybsol_check_index(sys, row);
    if (res != HYBSOL_SUCCESS)
        return res;

    const hybsol_row_t *const r = sys->rows + row;
    if (r->count == 0)
        return HYBSOL_ERROR_EMPTY_ROW;

    if (out)
        *out = r->entries[0]->col;
    return HYBSOL_SUCCESS;
}

hybsol_result_t hybsol_system_next_column(const hybsol_system_t *const sys, const uint64_t row, const uint64_t col,
                                          uint64_t *const out)
{
    hybsol_result_t res = hybsol_check_index(sys, row);
    if (res != HYBSOL_SUCCESS)
        return res;

    const hybsol_row_t *const r = sys->rows + row;
    if (r->count == 0)
        return HYBSOL_ERROR_EMPTY_ROW;

    const uint64_t idx = hybsol_row_find_geq(r, col + 1);
    if (idx == r->count)
        return HYBSOL_ERROR_NO_MORE_COLUMNS;

    if (out)
        *out = r->entries[idx]->col;
    return HYBSOL_SUCCESS;
}

void hybsol_system_no_lower_connections(const hybsol_system_t *const sys, uint8_t *const out)
{
    for (uint64_t i = 0; i < sys->n; ++i)
    {
        const hybsol_row_t *const row = sys->rows + i;
        out[i] = row->count == 0 ? 1 : (row->entries[0]->col >= i);
    }
}

/* ------------------------------------------------------------------------- */
/* Assembly                                                                   */
/* ------------------------------------------------------------------------- */

static hybsol_result_t row_insert_block(hybsol_row_t *const row, const uint64_t col, const uint64_t n_values,
                                        const double *const vals)
{
    const uint64_t idx = hybsol_row_find_geq(row, col);
    if (idx < row->count && row->entries[idx]->col == col)
    {
        // The block is already there, accumulate into it
        double *const dst = row->entries[idx]->vals;
        for (uint64_t i = 0; i < n_values; ++i)
            dst[i] += vals[i];
        return HYBSOL_SUCCESS;
    }

    hybsol_result_t res = hybsol_row_reserve(row, row->count + 1);
    if (res != HYBSOL_SUCCESS)
        return res;

    hybsol_row_entry_t *const entry = hybsol_alloc(sizeof(*entry) + (size_t)n_values * sizeof(*entry->vals));
    if (entry == NULL)
        return HYBSOL_ERROR_OUT_OF_MEMORY;

    entry->col = col;
    memcpy(entry->vals, vals, (size_t)n_values * sizeof(*entry->vals));

    if (idx < row->count)
    {
        HYBSOL_ASSERT(row->entries[idx]->col > col,
                      "Binary search was WRONG: %llu is not greater than the column %llu at index %llu.",
                      (unsigned long long)col, (unsigned long long)row->entries[idx]->col, (unsigned long long)idx);
        memmove(row->entries + idx + 1, row->entries + idx, (size_t)(row->count - idx) * sizeof(*row->entries));
    }

    row->entries[idx] = entry;
    row->count += 1;
    return HYBSOL_SUCCESS;
}

hybsol_result_t hybsol_system_reserve(hybsol_system_t *const sys, const uint64_t row, const uint64_t capacity)
{
    hybsol_result_t res = hybsol_check_index(sys, row);
    if (res != HYBSOL_SUCCESS)
        return res;

    res = hybsol_require_mutable(sys);
    if (res != HYBSOL_SUCCESS)
        return res;

    return hybsol_row_reserve(sys->rows + row, capacity);
}

hybsol_result_t hybsol_system_add_block(hybsol_system_t *const sys, const uint64_t row, const uint64_t col,
                                        const uint64_t n_rows, const uint64_t n_cols, const double *const vals)
{
    hybsol_result_t res = hybsol_check_index(sys, row);
    if (res != HYBSOL_SUCCESS)
        return res;
    res = hybsol_check_index(sys, col);
    if (res != HYBSOL_SUCCESS)
        return res;

    res = hybsol_require_mutable(sys);
    if (res != HYBSOL_SUCCESS)
        return res;

    if (n_rows != hybsol_block_size(sys, row) || n_cols != hybsol_block_size(sys, col))
        return HYBSOL_ERROR_INVALID_ARGUMENT;
    if (vals == NULL)
        return HYBSOL_ERROR_INVALID_ARGUMENT;

    res = row_insert_block(sys->rows + row, col, n_rows * n_cols, vals);
    if (res != HYBSOL_SUCCESS)
        return res;

    // Touching the diagonal of this row invalidates a cached factorization
    if (row == col)
        hybsol_invalidate_diagonal(sys, row);

    return HYBSOL_SUCCESS;
}

hybsol_result_t hybsol_system_add_blocks(hybsol_system_t *const sys, const uint64_t n_entries,
                                         const uint64_t rows[static n_entries], const uint64_t cols[static n_entries],
                                         const double *const data)
{
    if (n_entries == 0)
        return HYBSOL_SUCCESS;
    if (data == NULL)
        return HYBSOL_ERROR_INVALID_ARGUMENT;

    hybsol_result_t res = hybsol_require_mutable(sys);
    if (res != HYBSOL_SUCCESS)
        return res;

    for (uint64_t k = 0; k < n_entries; ++k)
    {
        res = hybsol_check_index(sys, rows[k]);
        if (res != HYBSOL_SUCCESS)
            return res;
        res = hybsol_check_index(sys, cols[k]);
        if (res != HYBSOL_SUCCESS)
            return res;
    }

    // Count how many blocks each row is about to receive so that every row
    // grows its entry array exactly once.
    uint64_t *const counts = hybsol_alloc(sizeof(*counts) * (size_t)sys->n);
    if (counts == NULL)
        return HYBSOL_ERROR_OUT_OF_MEMORY;
    memset(counts, 0, sizeof(*counts) * (size_t)sys->n);
    for (uint64_t k = 0; k < n_entries; ++k)
        counts[rows[k]] += 1;

    for (uint64_t i = 0; i < sys->n; ++i)
    {
        if (counts[i] == 0)
            continue;
        res = hybsol_row_reserve(sys->rows + i, sys->rows[i].count + counts[i]);
        if (res != HYBSOL_SUCCESS)
        {
            hybsol_free(counts);
            return res;
        }
    }
    hybsol_free(counts);

    // Insert the blocks, accumulating into duplicates on the way
    uint64_t offset = 0;
    for (uint64_t k = 0; k < n_entries; ++k)
    {
        const uint64_t n_values = hybsol_block_size(sys, rows[k]) * hybsol_block_size(sys, cols[k]);
        res = row_insert_block(sys->rows + rows[k], cols[k], n_values, data + offset);
        if (res != HYBSOL_SUCCESS)
            return res;
        if (rows[k] == cols[k])
            hybsol_invalidate_diagonal(sys, rows[k]);
        offset += n_values;
    }

    return HYBSOL_SUCCESS;
}

/* ------------------------------------------------------------------------- */
/* Row operations                                                             */
/* ------------------------------------------------------------------------- */

hybsol_result_t hybsol_system_multiply_row(hybsol_system_t *const sys, const uint64_t row, const uint64_t start_col,
                                           const hybsol_matrix_t *const mat)
{
    hybsol_result_t res = hybsol_check_index(sys, row);
    if (res != HYBSOL_SUCCESS)
        return res;
    res = hybsol_check_index(sys, start_col);
    if (res != HYBSOL_SUCCESS)
        return res;

    res = hybsol_require_mutable(sys);
    if (res != HYBSOL_SUCCESS)
        return res;

    const uint64_t n_rows = hybsol_block_size(sys, row);
    if (mat == NULL || mat->rows != mat->cols || mat->rows != n_rows)
        return HYBSOL_ERROR_INVALID_ARGUMENT;

    hybsol_row_t *const r = sys->rows + row;

    // Find the largest block in the row to size the temporary buffer
    uint64_t max_cols = 0;
    for (uint64_t i = 0; i < r->count; ++i)
    {
        const uint64_t block_size = hybsol_block_size(sys, r->entries[i]->col);
        if (block_size > max_cols)
            max_cols = block_size;
    }

    double *const buffer = hybsol_alloc(sizeof(*buffer) * (size_t)max_cols * (size_t)n_rows);
    if (buffer == NULL)
        return HYBSOL_ERROR_OUT_OF_MEMORY;

    for (uint64_t i = 0; i < r->count; ++i)
    {
        hybsol_row_entry_t *const entry = r->entries[i];
        if (entry->col < start_col)
            continue;

        const hybsol_matrix_t block = hybsol_matrix_view(n_rows, hybsol_block_size(sys, entry->col), entry->vals);
        const hybsol_matrix_t out = hybsol_matrix_view(block.rows, block.cols, buffer);

        hybsol_matrix_multiply(mat, &block, &out);
        memcpy(entry->vals, buffer, sizeof(*buffer) * (size_t)(block.rows * block.cols));
    }

    hybsol_free(buffer);
    hybsol_invalidate_diagonal(sys, row);
    return HYBSOL_SUCCESS;
}

hybsol_result_t hybsol_system_eliminate_row_with(hybsol_system_t *const sys, const uint64_t row_tgt,
                                                 const uint64_t row_src, const hybsol_matrix_t *const mat)
{
    hybsol_result_t res = hybsol_check_index(sys, row_tgt);
    if (res != HYBSOL_SUCCESS)
        return res;
    res = hybsol_check_index(sys, row_src);
    if (res != HYBSOL_SUCCESS)
        return res;

    res = hybsol_require_mutable(sys);
    if (res != HYBSOL_SUCCESS)
        return res;

    const uint64_t size_tgt = hybsol_block_size(sys, row_tgt);
    const uint64_t size_src = hybsol_block_size(sys, row_src);
    if (mat == NULL || mat->rows != size_tgt || mat->cols != size_src)
        return HYBSOL_ERROR_INVALID_ARGUMENT;

    hybsol_row_t *const row_t = sys->rows + row_tgt;
    const hybsol_row_t *const row_s = sys->rows + row_src;
    if (row_t->count == 0 || row_s->count == 0)
        return HYBSOL_ERROR_EMPTY_ROW;

    // Find how big the buffer has to be for intermediate results, and how many
    // entries the target row will hold once the two rows have been merged.
    uint64_t unique_cols = 0, pos_tgt, pos_src;
    uint64_t max_cols = 0;
    for (pos_tgt = row_t->count, pos_src = row_s->count;; ++unique_cols)
    {
        const uint64_t col_tgt = row_t->entries[pos_tgt - 1]->col;
        const uint64_t col_src = row_s->entries[pos_src - 1]->col;

        if (col_tgt <= row_src && col_src <= row_src)
            break;

        uint64_t block_size;
        if (col_tgt == col_src)
        {
            pos_tgt -= 1;
            pos_src -= 1;
            block_size = hybsol_block_size(sys, col_tgt);
        }
        else if (col_src > col_tgt)
        {
            pos_src -= 1;
            block_size = hybsol_block_size(sys, col_src);
        }
        else
        {
            pos_tgt -= 1;
            block_size = hybsol_block_size(sys, col_tgt);
        }
        if (block_size > max_cols)
            max_cols = block_size;
    }

    if (row_t->entries[pos_tgt - 1]->col != row_src && row_s->entries[pos_src - 1]->col != row_src)
        return HYBSOL_ERROR_BLOCK_NOT_IN_SYSTEM;

    const uint64_t needed_size = unique_cols + pos_tgt;

    res = hybsol_row_reserve(row_t, needed_size);
    if (res != HYBSOL_SUCCESS)
        return res;

    double *const buffer = hybsol_alloc(sizeof(*buffer) * (size_t)max_cols * (size_t)size_tgt);
    if (buffer == NULL)
        return HYBSOL_ERROR_OUT_OF_MEMORY;

    // Subtract the product of the multiplier and the source entries from the
    // target entries, working backwards so untouched entries only move once.
    // Allocations are made while the row is being rewritten; if one fails the
    // whole target row is emptied so that the system at least stays walkable.
    uint64_t i, k;
    for (i = unique_cols, k = needed_size, pos_tgt = row_t->count, pos_src = row_s->count; i > 0; --i, --k)
    {
        hybsol_row_entry_t *const entry_tgt = row_t->entries[pos_tgt - 1];
        const hybsol_row_entry_t *const entry_src = row_s->entries[pos_src - 1];
        const uint64_t col_src = entry_src->col;
        const uint64_t col_tgt = entry_tgt->col;

        const hybsol_matrix_t out = hybsol_matrix_view(size_tgt, hybsol_block_size(sys, col_src), buffer);
        const hybsol_matrix_t mat_src =
            hybsol_matrix_view(size_src, hybsol_block_size(sys, col_src), (double *)entry_src->vals);
        const hybsol_matrix_t mat_tgt =
            hybsol_matrix_view(size_tgt, hybsol_block_size(sys, col_tgt), (double *)entry_tgt->vals);

        HYBSOL_ASSERT(pos_tgt == k || row_t->entries[k - 1] == NULL,
                      "Destination at index %llu was non-null and contained entry for column %llu!",
                      (unsigned long long)(k - 1), (unsigned long long)col_tgt);

        if (col_tgt == col_src)
        {
            hybsol_matrix_multiply(mat, &mat_src, &out);
            hybsol_matrix_subtract_inplace(&mat_tgt, &out);

            row_t->entries[pos_tgt - 1] = NULL;
            row_t->entries[k - 1] = entry_tgt;
            pos_tgt -= 1;
            pos_src -= 1;
        }
        else if (col_src > col_tgt)
        {
            hybsol_matrix_multiply(mat, &mat_src, &out);
            hybsol_row_entry_t *const new_entry =
                hybsol_alloc(sizeof(*new_entry) + (size_t)out.rows * (size_t)out.cols * sizeof(double));
            if (new_entry == NULL)
            {
                // The row can not be brought back to a consistent state, so
                // drop it entirely rather than leave a half-updated mess.
                for (uint64_t j = 0; j < row_t->capacity; ++j)
                {
                    hybsol_free(row_t->entries[j]);
                    row_t->entries[j] = NULL;
                }
                row_t->count = 0;
                hybsol_free(buffer);
                return HYBSOL_ERROR_OUT_OF_MEMORY;
            }
            new_entry->col = col_src;
            for (uint64_t j = 0; j < out.rows * out.cols; ++j)
                new_entry->vals[j] = -out.data[j];

            row_t->entries[k - 1] = new_entry;
            pos_src -= 1;
        }
        else
        {
            row_t->entries[pos_tgt - 1] = NULL;
            row_t->entries[k - 1] = entry_tgt;
            pos_tgt -= 1;
        }
    }

    row_t->count = needed_size;

#if HYBSOL_ENABLE_ASSERTS
    for (i = 0; i < row_t->count; ++i)
    {
        const hybsol_row_entry_t *const entry = row_t->entries[i];
        HYBSOL_ASSERT(entry != NULL, "Entry %llu was NULL!", (unsigned long long)i);
        if (i > 0)
            HYBSOL_ASSERT(entry->col > row_t->entries[i - 1]->col, "Entries %llu and %llu were not sorted!",
                          (unsigned long long)(i - 1), (unsigned long long)i);
    }
#endif

    hybsol_free(buffer);
    hybsol_invalidate_diagonal(sys, row_tgt);
    return HYBSOL_SUCCESS;
}

hybsol_result_t hybsol_system_eliminate_row(hybsol_system_t *const sys, const uint64_t row_tgt, const uint64_t row_src)
{
    hybsol_matrix_t mat;
    const hybsol_result_t res = hybsol_system_get_block(sys, row_tgt, row_src, &mat);
    if (res != HYBSOL_SUCCESS)
        return res;

    return hybsol_system_eliminate_row_with(sys, row_tgt, row_src, &mat);
}

/* ------------------------------------------------------------------------- */
/* Dense export                                                               */
/* ------------------------------------------------------------------------- */

hybsol_result_t hybsol_system_to_dense(const hybsol_system_t *const sys, double *const out)
{
    if (out == NULL)
        return HYBSOL_ERROR_INVALID_ARGUMENT;

    const uint64_t total = hybsol_system_total_size(sys);
    memset(out, 0, sizeof(*out) * (size_t)total * (size_t)total);

    uint64_t offset_row = 0;
    for (uint64_t i = 0; i < sys->n; ++i)
    {
        const hybsol_row_t *const row = sys->rows + i;
        const uint64_t n_rows = hybsol_block_size(sys, i);
        uint64_t offset_col = 0, i_col = 0;
        for (uint64_t j = 0; j < row->count; ++j)
        {
            const hybsol_row_entry_t *const entry = row->entries[j];
            while (i_col < entry->col)
            {
                offset_col += hybsol_block_size(sys, i_col);
                i_col += 1;
            }
            const uint64_t n_cols = hybsol_block_size(sys, entry->col);
            for (uint64_t k1 = 0; k1 < n_rows; ++k1)
                for (uint64_t k2 = 0; k2 < n_cols; ++k2)
                    out[(offset_row + k1) * total + (offset_col + k2)] = entry->vals[k1 * n_cols + k2];
        }
        offset_row += n_rows;
    }

    return HYBSOL_SUCCESS;
}
