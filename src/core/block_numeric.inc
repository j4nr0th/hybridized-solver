/*
 * The value-carrying half of assembly and the row operations, instantiated
 * once per floating-point type.
 *
 * Unguarded on purpose: ``block_system.c`` includes this file twice, after
 * defining
 *
 *   HYBSOL_SCALAR    the type the system stores its blocks in
 *   HYBSOL_ACC       the type products accumulate in
 *   HYBSOL_VIEW      the view type spelling that storage
 *   HYBSOL_KERNEL(n) the dense-kernel name matching ``HYBSOL_VIEW``
 *   HYBSOL_FN(name)  the name the resulting function gets
 *
 * Only the arithmetic is instantiated. Index checks, the row bookkeeping and
 * the allocation sizes are shared: an entry is ``n_values *`` the size of
 * ``HYBSOL_SCALAR`` bytes either way, and the helpers this file reaches —
 * ``row_place_entry``, ``row_new_entry``, ``row_find_or_create`` — are the
 * same statics for both spellings.
 *
 * Everything here is ``static``: the two instantiations are private to
 * ``block_system.c`` and reached through its public spellings, which are what
 * own the precision check. Every macro is undefined at the bottom so a second
 * inclusion is well defined.
 */

static hybsol_result_t HYBSOL_FN(row_insert_block)(hybsol_row_t *const row, const uint64_t col, const uint64_t n_values,
                                                   const HYBSOL_SCALAR *const vals)
{
    const uint64_t idx = hybsol_row_find_geq(row, col);
    if (idx < row->count && row->entries[idx]->col == col)
    {
        // The block is already there, accumulate into it
        HYBSOL_SCALAR *const dst = (HYBSOL_SCALAR *)row->entries[idx]->vals;
        for (uint64_t i = 0; i < n_values; ++i)
            dst[i] += vals[i];
        return HYBSOL_SUCCESS;
    }

    hybsol_result_t res = hybsol_row_reserve(row, row->count + 1);
    if (res != HYBSOL_SUCCESS)
        return res;

    hybsol_row_entry_t *const entry = row_new_entry(col, n_values, sizeof(HYBSOL_SCALAR));
    if (entry == NULL)
        return HYBSOL_ERROR_OUT_OF_MEMORY;

    memcpy(entry->vals, vals, (size_t)n_values * sizeof(HYBSOL_SCALAR));
    row_place_entry(row, idx, entry);
    return HYBSOL_SUCCESS;
}

static hybsol_result_t HYBSOL_FN(add_block)(hybsol_system_t *const sys, const uint64_t row, const uint64_t col,
                                            const uint64_t n_rows, const uint64_t n_cols,
                                            const HYBSOL_SCALAR *const vals)
{
    hybsol_require_index(sys, row);
    hybsol_require_index(sys, col);

    hybsol_result_t res = hybsol_require_mutable(sys);
    if (res != HYBSOL_SUCCESS)
        return res;

    HYBSOL_ASSERT(n_rows == hybsol_block_size(sys, row) && n_cols == hybsol_block_size(sys, col),
                  "Block (%llu, %llu) is %llu x %llu in the system, but %llu x %llu was given.",
                  (unsigned long long)row, (unsigned long long)col, (unsigned long long)hybsol_block_size(sys, row),
                  (unsigned long long)hybsol_block_size(sys, col), (unsigned long long)n_rows,
                  (unsigned long long)n_cols);
    HYBSOL_ASSERT(vals != NULL, "The block values must not be NULL.");

    res = HYBSOL_FN(row_insert_block)(sys->rows + row, col, n_rows * n_cols, vals);
    if (res != HYBSOL_SUCCESS)
        return res;

    // Touching the diagonal of this row invalidates a cached factorization
    if (row == col)
        hybsol_invalidate_diagonal(sys, row);

    return HYBSOL_SUCCESS;
}

static hybsol_result_t HYBSOL_FN(add_blocks)(hybsol_system_t *const sys, const uint64_t n_entries,
                                             const uint64_t *const rows, const uint64_t *const cols,
                                             const HYBSOL_SCALAR *const data)
{
    if (n_entries == 0)
        return HYBSOL_SUCCESS;
    HYBSOL_ASSERT(data != NULL, "The block values must not be NULL.");

    hybsol_result_t res = hybsol_require_mutable(sys);
    if (res != HYBSOL_SUCCESS)
        return res;

    for (uint64_t k = 0; k < n_entries; ++k)
    {
        hybsol_require_index(sys, rows[k]);
        hybsol_require_index(sys, cols[k]);
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
        res = HYBSOL_FN(row_insert_block)(sys->rows + rows[k], cols[k], n_values, data + offset);
        if (res != HYBSOL_SUCCESS)
            return res;
        if (rows[k] == cols[k])
            hybsol_invalidate_diagonal(sys, rows[k]);
        offset += n_values;
    }

    return HYBSOL_SUCCESS;
}

static hybsol_result_t HYBSOL_FN(block_storage)(hybsol_system_t *const sys, const uint64_t row, const uint64_t col,
                                                HYBSOL_VIEW *const out)
{
    hybsol_require_index(sys, row);
    hybsol_require_index(sys, col);

    hybsol_result_t res = hybsol_require_mutable(sys);
    if (res != HYBSOL_SUCCESS)
        return res;

    const uint64_t n_rows = hybsol_block_size(sys, row);
    const uint64_t n_cols = hybsol_block_size(sys, col);

    hybsol_row_entry_t *entry;
    res = row_find_or_create(sys->rows + row, col, n_rows * n_cols, sizeof(HYBSOL_SCALAR), &entry);
    if (res != HYBSOL_SUCCESS)
        return res;

    // Whoever writes through this view is replacing the diagonal block
    if (row == col)
        hybsol_invalidate_diagonal(sys, row);

    *out = HYBSOL_KERNEL(view)(n_rows, n_cols, (HYBSOL_SCALAR *)entry->vals);
    return HYBSOL_SUCCESS;
}

static void HYBSOL_FN(lookup_block)(const hybsol_system_t *const sys, const uint64_t row, const uint64_t col,
                                    HYBSOL_VIEW *const out)
{
    HYBSOL_ASSERT(out != NULL, "The output pointer must not be NULL.");
    hybsol_require_index(sys, row);
    hybsol_require_index(sys, col);

    uint64_t idx;
    HYBSOL_ASSERT(hybsol_row_find(sys->rows + row, col, &idx), "The system does not contain the block (%llu, %llu).",
                  (unsigned long long)row, (unsigned long long)col);

    *out = HYBSOL_KERNEL(view)(hybsol_block_size(sys, row), hybsol_block_size(sys, col),
                               (HYBSOL_SCALAR *)sys->rows[row].entries[idx]->vals);
}

static hybsol_result_t HYBSOL_FN(multiply_row)(hybsol_system_t *const sys, const uint64_t row, const uint64_t start_col,
                                               const HYBSOL_VIEW *const mat)
{
    const uint64_t n_rows = hybsol_block_size(sys, row);
    HYBSOL_ASSERT(mat != NULL, "The multiplier matrix must not be NULL.");
    HYBSOL_ASSERT(mat->rows == mat->cols && mat->rows == n_rows,
                  "The multiplier must be %llu x %llu to match block row %llu, but was %llu x %llu.",
                  (unsigned long long)n_rows, (unsigned long long)n_rows, (unsigned long long)row,
                  (unsigned long long)mat->rows, (unsigned long long)mat->cols);

    hybsol_row_t *const r = sys->rows + row;

    // Find the largest block in the row to size the temporary buffer. An
    // empty row has nothing to multiply (and would otherwise ask for a
    // zero-sized buffer, which reads as an allocation failure).
    uint64_t max_cols = 0;
    for (uint64_t i = 0; i < r->count; ++i)
    {
        const uint64_t block_size = hybsol_block_size(sys, r->entries[i]->col);
        if (block_size > max_cols)
            max_cols = block_size;
    }
    if (max_cols == 0)
        return HYBSOL_SUCCESS;

    HYBSOL_SCALAR *const buffer = hybsol_alloc(sizeof(*buffer) * (size_t)max_cols * (size_t)n_rows);
    if (buffer == NULL)
        return HYBSOL_ERROR_OUT_OF_MEMORY;

    for (uint64_t i = 0; i < r->count; ++i)
    {
        hybsol_row_entry_t *const entry = r->entries[i];
        if (entry->col < start_col)
            continue;

        const HYBSOL_VIEW block =
            HYBSOL_KERNEL(view)(n_rows, hybsol_block_size(sys, entry->col), (HYBSOL_SCALAR *)entry->vals);
        const HYBSOL_VIEW out = HYBSOL_KERNEL(view)(block.rows, block.cols, buffer);

        HYBSOL_KERNEL(multiply)(mat, &block, &out);
        memcpy(entry->vals, buffer, sizeof(*buffer) * (size_t)(block.rows * block.cols));
    }

    hybsol_free(buffer);
    hybsol_invalidate_diagonal(sys, row);
    return HYBSOL_SUCCESS;
}

static hybsol_result_t HYBSOL_FN(eliminate_row_with)(hybsol_system_t *const sys, const uint64_t row_tgt,
                                                     const uint64_t row_src, const HYBSOL_VIEW *const mat)
{
    const uint64_t size_tgt = hybsol_block_size(sys, row_tgt);
    const uint64_t size_src = hybsol_block_size(sys, row_src);
    HYBSOL_ASSERT(mat != NULL, "The multiplier matrix must not be NULL.");
    HYBSOL_ASSERT(mat->rows == size_tgt && mat->cols == size_src,
                  "The multiplier must be %llu x %llu, but was %llu x %llu.", (unsigned long long)size_tgt,
                  (unsigned long long)size_src, (unsigned long long)mat->rows, (unsigned long long)mat->cols);

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

    HYBSOL_ASSERT(row_t->entries[pos_tgt - 1]->col == row_src || row_s->entries[pos_src - 1]->col == row_src,
                  "Neither row contains column %llu, so there is nothing to eliminate with.",
                  (unsigned long long)row_src);

    const uint64_t needed_size = unique_cols + pos_tgt;

    hybsol_result_t res = hybsol_row_reserve(row_t, needed_size);
    if (res != HYBSOL_SUCCESS)
        return res;

    // `max_cols` is zero when there is nothing left to eliminate, in which
    // case the buffer is never touched; `hybsol_alloc(0)` returns ``NULL``,
    // so the allocation is skipped instead of being read as a failure.
    const size_t buffer_len = (size_t)max_cols * (size_t)size_tgt;
    HYBSOL_SCALAR *buffer = NULL;
    if (buffer_len != 0)
    {
        buffer = hybsol_alloc(sizeof(*buffer) * buffer_len);
        if (buffer == NULL)
            return HYBSOL_ERROR_OUT_OF_MEMORY;
    }

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

        const HYBSOL_VIEW out = HYBSOL_KERNEL(view)(size_tgt, hybsol_block_size(sys, col_src), buffer);
        const HYBSOL_VIEW mat_src =
            HYBSOL_KERNEL(view)(size_src, hybsol_block_size(sys, col_src), (HYBSOL_SCALAR *)entry_src->vals);
        const HYBSOL_VIEW mat_tgt =
            HYBSOL_KERNEL(view)(size_tgt, hybsol_block_size(sys, col_tgt), (HYBSOL_SCALAR *)entry_tgt->vals);

        HYBSOL_ASSERT(pos_tgt == k || row_t->entries[k - 1] == NULL,
                      "Destination at index %llu was non-null and contained entry for column %llu!",
                      (unsigned long long)(k - 1), (unsigned long long)col_tgt);

        if (col_tgt == col_src)
        {
            HYBSOL_KERNEL(multiply)(mat, &mat_src, &out);
            HYBSOL_KERNEL(subtract_inplace)(&mat_tgt, &out);

            row_t->entries[pos_tgt - 1] = NULL;
            row_t->entries[k - 1] = entry_tgt;
            pos_tgt -= 1;
            pos_src -= 1;
        }
        else if (col_src > col_tgt)
        {
            HYBSOL_KERNEL(multiply)(mat, &mat_src, &out);
            hybsol_row_entry_t *const new_entry =
                hybsol_alloc(sizeof(*new_entry) + (size_t)out.rows * (size_t)out.cols * sizeof(HYBSOL_SCALAR));
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
                ((HYBSOL_SCALAR *)new_entry->vals)[j] = -out.data[j];

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

static hybsol_result_t HYBSOL_FN(eliminate_row)(hybsol_system_t *const sys, const uint64_t row_tgt,
                                                const uint64_t row_src)
{
    hybsol_require_index(sys, row_tgt);
    hybsol_require_index(sys, row_src);

    // The multiplier is the block the system already stores, so this spelling
    // needs no operand of its own and works for either precision.
    HYBSOL_VIEW mat;
    HYBSOL_FN(lookup_block)(sys, row_tgt, row_src, &mat);

    return HYBSOL_FN(eliminate_row_with)(sys, row_tgt, row_src, &mat);
}

static hybsol_result_t HYBSOL_FN(to_dense)(const hybsol_system_t *const sys, HYBSOL_SCALAR *const out)
{
    const uint64_t total = hybsol_system_total_size(sys);
    memset(out, 0, sizeof(HYBSOL_SCALAR) * (size_t)total * (size_t)total);

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
            {
                for (uint64_t k2 = 0; k2 < n_cols; ++k2)
                {
                    out[(offset_row + k1) * total + (offset_col + k2)] =
                        ((const HYBSOL_SCALAR *)entry->vals)[k1 * n_cols + k2];
                }
            }
        }
        offset_row += n_rows;
    }

    return HYBSOL_SUCCESS;
}

#undef HYBSOL_SCALAR
#undef HYBSOL_ACC
#undef HYBSOL_VIEW
#undef HYBSOL_KERNEL
#undef HYBSOL_FN
