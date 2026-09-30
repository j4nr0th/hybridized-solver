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

    CUTL_ASSERT(lo + len <= size, "Internal error: lo + len > size (%llu + %llu <= %llu).", (unsigned long long)lo,
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

hybsol_result_t hybsol_row_reserve(hybsol_system_t *const sys, hybsol_row_t *const row, const uint64_t needed)
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

    // Not `realloc`: the old array may have come from the system's allocator
    // while the new one comes from a decomposition's bump region, and the two
    // must be told apart before the old block is released.
    const size_t old_bytes = (size_t)row->capacity * sizeof(*row->entries);
    hybsol_row_entry_t **const ptr = hybsol_alloc(hybsol_current_alloc(sys), (size_t)new_capacity * sizeof(*ptr));
    if (ptr == NULL)
        return HYBSOL_ERROR_OUT_OF_MEMORY;

    if (row->entries != NULL)
    {
        memcpy(ptr, row->entries, old_bytes);
        hybsol_free(hybsol_ptr_is_pooled(sys, row->entries) ? hybsol_current_alloc(sys) : row->allocator, row->entries);
    }

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

void hybsol_invalidate_diagonal(hybsol_system_t *const sys, const uint64_t idx)
{
    if (idx < sys->n)
        sys->diag_decomposed[idx] = 0;
}

/* ------------------------------------------------------------------------- */
/* Assembly                                                                   */
/* ------------------------------------------------------------------------- */

/**
 * Insert ``entry`` into ``row`` at the slot ``idx`` it was found to belong in.
 *
 * Only the pointer array moves; the entries themselves are individually
 * allocated and never relocated, which is what lets a caller keep a pointer
 * into one block while further blocks are added to the same row.
 */
static void row_place_entry(hybsol_row_t *const row, const uint64_t idx, hybsol_row_entry_t *const entry)
{
    if (idx < row->count)
    {
        CUTL_ASSERT(row->entries[idx]->col > entry->col,
                    "Binary search was WRONG: %llu is not greater than the column %llu at index %llu.",
                    (unsigned long long)entry->col, (unsigned long long)row->entries[idx]->col,
                    (unsigned long long)idx);
        memmove(row->entries + idx + 1, row->entries + idx, (size_t)(row->count - idx) * sizeof(*row->entries));
    }

    row->entries[idx] = entry;
    row->count += 1;
}

/* Defined below; reached from the per-precision instantiations in block_numeric.inc. */
static hybsol_result_t row_find_or_create(const cutl_allocator_t *alloc, hybsol_system_t *sys, hybsol_row_t *row,
                                          uint64_t col, uint64_t n_values, size_t elem_size, hybsol_row_entry_t **out);

/**
 * Allocate an entry for ``n_values`` elements of ``elem_size`` bytes and
 * stamp it with its column.
 *
 * Both insertion paths go through here so that an entry is never placed in a
 * row before its ``col`` has been set — the row's binary search reads it
 * immediately.
 *
 * :returns: The entry, or ``NULL`` when out of memory.
 */
static hybsol_row_entry_t *row_new_entry(const cutl_allocator_t *const alloc, const uint64_t col,
                                         const uint64_t n_values, const size_t elem_size)
{
    hybsol_row_entry_t *const entry = hybsol_alloc(alloc, sizeof(*entry) + (size_t)n_values * elem_size);
    if (entry == NULL)
        return NULL;
    entry->col = col;
    return entry;
}

/**
 * Find the entry for ``col``, creating it zero-filled when it is absent.
 *
 * An entry that is already present is handed back untouched: this never
 * clears and never accumulates, so a caller can re-fetch the same buffer
 * without losing whatever it wrote into it last time.
 */
static hybsol_result_t row_find_or_create(const cutl_allocator_t *const alloc, hybsol_system_t *const sys,
                                          hybsol_row_t *const row, const uint64_t col, const uint64_t n_values,
                                          const size_t elem_size, hybsol_row_entry_t **const out)
{
    const uint64_t idx = hybsol_row_find_geq(row, col);
    if (idx < row->count && row->entries[idx]->col == col)
    {
        *out = row->entries[idx];
        return HYBSOL_SUCCESS;
    }
    hybsol_result_t res = hybsol_row_reserve(sys, row, row->count + 1);
    if (res != HYBSOL_SUCCESS)
        return res;

    hybsol_row_entry_t *const entry = row_new_entry(alloc, col, n_values, elem_size);
    if (entry == NULL)
        return HYBSOL_ERROR_OUT_OF_MEMORY;

    memset(entry->vals, 0, (size_t)n_values * elem_size);
    row_place_entry(row, idx, entry);

    *out = entry;
    return HYBSOL_SUCCESS;
}

/* ------------------------------------------------------------------------- */
/* The value-carrying half, instantiated once per precision                   */
/* ------------------------------------------------------------------------- */

/*
 * Everything that touches block values is written once in
 * ``block_numeric.inc`` and instantiated here. Both instantiations are
 * private to this file: callers go through the spellings below, which are
 * what own the precision check.
 */
#define HYBSOL_SCALAR double
#define HYBSOL_ACC double
#define HYBSOL_VIEW hybsol_matrix_t
#define HYBSOL_KERNEL(name) hybsol_matrix_##name
#define HYBSOL_FN(name) hybsol_f64_##name
#include "block_numeric.inc"

#define HYBSOL_SCALAR float
#define HYBSOL_ACC float
#define HYBSOL_VIEW hybsol_fmatrix_t
#define HYBSOL_KERNEL(name) hybsol_fmatrix_##name
#define HYBSOL_FN(name) hybsol_f32_##name
#include "block_numeric.inc"

/* ------------------------------------------------------------------------- */
/* Lifetime                                                                   */
/* ------------------------------------------------------------------------- */

static void free_row(const hybsol_system_t *const sys, hybsol_row_t *const row)
{
    for (uint64_t i = 0; i < row->count; ++i)
    {
        // A decomposition carves its fill-in out of a region the system owns
        // outright, so those entries are not freed one by one.
        if (!hybsol_ptr_is_pooled(sys, row->entries[i]))
            hybsol_free(row->allocator, row->entries[i]);
        row->entries[i] = NULL;
    }
    if (!hybsol_ptr_is_pooled(sys, row->entries))
        hybsol_free(row->allocator, row->entries);
    row->entries = NULL;
    row->count = 0;
    row->capacity = 0;
}

hybsol_result_t hybsol_system_create_with_precision(const uint64_t n_blocks,
                                                    const uint64_t block_sizes[static n_blocks],
                                                    const hybsol_precision_t precision, hybsol_system_t **const out,
                                                    const cutl_allocator_t *const allocator)
{
    CUTL_ASSERT(out != NULL, "The output pointer must not be NULL.");
    CUTL_ASSERT(allocator != NULL, "The allocator must not be NULL; pass &CUTL_STD_ALLOCATOR for the default.");
    CUTL_ASSERT(n_blocks >= 1, "A system needs at least one block, but %llu were requested.",
                (unsigned long long)n_blocks);
    CUTL_ASSERT(precision == HYBSOL_PRECISION_DOUBLE || precision == HYBSOL_PRECISION_SINGLE,
                "Precision must be one of the enumerators, but was %d.", (int)precision);

    // The sizes are validated up front so that no failure path can run on a
    // partially initialised system.
    for (uint64_t i = 0; i < n_blocks; ++i)
    {
        CUTL_ASSERT(block_sizes[i] > 0, "Block %llu has size 0; every block must be non-empty.", (unsigned long long)i);
    }

    *out = NULL;

    hybsol_system_t *const sys = hybsol_alloc(allocator, sizeof(*sys));
    if (sys == NULL)
        return HYBSOL_ERROR_OUT_OF_MEMORY;
    *sys = (hybsol_system_t){0};
    sys->allocator = allocator;
    sys->n = n_blocks;
    sys->precision = precision;

    sys->block_offsets = hybsol_alloc(allocator, sizeof(*sys->block_offsets) * (size_t)(n_blocks + 1));
    sys->rows = hybsol_alloc(allocator, sizeof(*sys->rows) * (size_t)n_blocks);
    sys->diag_decomposed = hybsol_alloc(allocator, sizeof(*sys->diag_decomposed) * (size_t)n_blocks);
    if (sys->block_offsets == NULL || sys->rows == NULL || sys->diag_decomposed == NULL)
    {
        hybsol_system_destroy(sys);
        return HYBSOL_ERROR_OUT_OF_MEMORY;
    }

    sys->block_offsets[0] = 0;
    for (uint64_t i = 0; i < n_blocks; ++i)
    {
        sys->block_offsets[i + 1] = sys->block_offsets[i] + block_sizes[i];
        sys->rows[i] = (hybsol_row_t){.allocator = allocator};
        sys->diag_decomposed[i] = 0;
    }

    *out = sys;
    return HYBSOL_SUCCESS;
}

hybsol_result_t hybsol_system_create(const uint64_t n_blocks, const uint64_t block_sizes[static n_blocks],
                                     hybsol_system_t **const out, const cutl_allocator_t *const allocator)
{
    return hybsol_system_create_with_precision(n_blocks, block_sizes, HYBSOL_PRECISION_DOUBLE, out, allocator);
}

hybsol_precision_t hybsol_system_precision(const hybsol_system_t *const sys)
{
    return sys->precision;
}

void hybsol_system_destroy(hybsol_system_t *const sys)
{
    if (sys == NULL)
        return;

    if (sys->rows != NULL)
    {
        for (uint64_t i = 0; i < sys->n; ++i)
            free_row(sys, sys->rows + i);
    }

    // The regions are released after the rows, because the pooled check the
    // rows rely on reads them.
    hybsol_thread_allocs_done(sys);
    hybsol_regions_free(sys);

    hybsol_free(sys->allocator, sys->rows);
    hybsol_free(sys->allocator, sys->block_offsets);
    hybsol_free(sys->allocator, sys->diag_decomposed);
    hybsol_free(sys->allocator, sys->ops);
    hybsol_free(sys->allocator, sys);
}

hybsol_result_t hybsol_system_copy(const hybsol_system_t *const sys, hybsol_system_t **const out)
{
    CUTL_ASSERT(out != NULL, "The output pointer must not be NULL.");
    *out = NULL;

    hybsol_system_t *const dst = hybsol_alloc(sys->allocator, sizeof(*dst));
    if (dst == NULL)
        return HYBSOL_ERROR_OUT_OF_MEMORY;
    *dst = (hybsol_system_t){0};
    dst->allocator = sys->allocator;
    dst->n = sys->n;
    dst->precision = sys->precision;
    dst->decomposed = sys->decomposed;
    dst->n_ops = sys->n_ops;
    dst->ops_capacity = sys->n_ops;

    dst->block_offsets = hybsol_alloc(sys->allocator, sizeof(*dst->block_offsets) * (size_t)(sys->n + 1));
    dst->rows = hybsol_alloc(sys->allocator, sizeof(*dst->rows) * (size_t)sys->n);
    dst->diag_decomposed = hybsol_alloc(sys->allocator, sizeof(*dst->diag_decomposed) * (size_t)sys->n);
    if (dst->block_offsets == NULL || dst->rows == NULL || dst->diag_decomposed == NULL)
    {
        hybsol_system_destroy(dst);
        return HYBSOL_ERROR_OUT_OF_MEMORY;
    }

    memcpy(dst->block_offsets, sys->block_offsets, sizeof(*dst->block_offsets) * (size_t)(sys->n + 1));
    memcpy(dst->diag_decomposed, sys->diag_decomposed, sizeof(*dst->diag_decomposed) * (size_t)sys->n);
    for (uint64_t i = 0; i < sys->n; ++i)
        dst->rows[i] = (hybsol_row_t){.allocator = sys->allocator};

    for (uint64_t i = 0; i < sys->n; ++i)
    {
        const hybsol_row_t *const src = sys->rows + i;
        hybsol_row_t *const tgt = dst->rows + i;

        if (src->count == 0)
            continue;

        tgt->entries = hybsol_alloc(sys->allocator, sizeof(*tgt->entries) * (size_t)src->count);
        if (tgt->entries == NULL)
        {
            hybsol_system_destroy(dst);
            return HYBSOL_ERROR_OUT_OF_MEMORY;
        }
        tgt->capacity = src->count;
        for (uint64_t j = 0; j < src->count; ++j)
            tgt->entries[j] = NULL;

        const size_t elem_size = hybsol_scalar_size(sys->precision);
        for (uint64_t j = 0; j < src->count; ++j)
        {
            const hybsol_row_entry_t *const entry = src->entries[j];
            const uint64_t n_values = hybsol_block_size(sys, i) * hybsol_block_size(sys, entry->col);

            hybsol_row_entry_t *const copy = hybsol_alloc(sys->allocator, sizeof(*copy) + (size_t)n_values * elem_size);
            if (copy == NULL)
            {
                hybsol_system_destroy(dst);
                return HYBSOL_ERROR_OUT_OF_MEMORY;
            }
            copy->col = entry->col;
            memcpy(copy->vals, entry->vals, (size_t)n_values * elem_size);
            tgt->entries[j] = copy;
            tgt->count += 1;
        }
    }

    if (sys->n_ops > 0)
    {
        dst->ops = hybsol_alloc(sys->allocator, sizeof(*dst->ops) * (size_t)sys->n_ops);
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
    hybsol_require_index(sys, row);

    const hybsol_row_t *const r = sys->rows + row;
    if (n_written)
        *n_written = r->count;
    CUTL_ASSERT(out != NULL, "The output array must not be NULL.");
    CUTL_ASSERT(capacity >= r->count, "The destination holds %llu indices, but the row has %llu.",
                (unsigned long long)capacity, (unsigned long long)r->count);

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
    hybsol_require_precision(sys, HYBSOL_PRECISION_DOUBLE);

    hybsol_f64_lookup_block(sys, row, col, out);
    return HYBSOL_SUCCESS;
}

hybsol_result_t hybsol_system_get_block_f32(hybsol_system_t *const sys, const uint64_t row, const uint64_t col,
                                            hybsol_fmatrix_t *const out)
{
    hybsol_require_precision(sys, HYBSOL_PRECISION_SINGLE);

    hybsol_f32_lookup_block(sys, row, col, out);
    return HYBSOL_SUCCESS;
}

hybsol_result_t hybsol_system_first_column(const hybsol_system_t *const sys, const uint64_t row, uint64_t *const out)
{
    hybsol_require_index(sys, row);

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
    hybsol_require_index(sys, row);

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

hybsol_result_t hybsol_system_reserve(hybsol_system_t *const sys, const uint64_t row, const uint64_t capacity)
{
    hybsol_require_index(sys, row);

    const hybsol_result_t res = hybsol_require_mutable(sys);
    if (res != HYBSOL_SUCCESS)
        return res;

    return hybsol_row_reserve(sys, sys->rows + row, capacity);
}

hybsol_result_t hybsol_system_add_block(hybsol_system_t *const sys, const uint64_t row, const uint64_t col,
                                        const uint64_t n_rows, const uint64_t n_cols, const double *const vals)
{
    hybsol_require_precision(sys, HYBSOL_PRECISION_DOUBLE);

    return hybsol_f64_add_block(sys, row, col, n_rows, n_cols, vals);
}

hybsol_result_t hybsol_system_add_block_f32(hybsol_system_t *const sys, const uint64_t row, const uint64_t col,
                                            const uint64_t n_rows, const uint64_t n_cols, const float *const vals)
{
    hybsol_require_precision(sys, HYBSOL_PRECISION_SINGLE);

    return hybsol_f32_add_block(sys, row, col, n_rows, n_cols, vals);
}

hybsol_result_t hybsol_system_add_blocks(hybsol_system_t *const sys, const uint64_t n_entries,
                                         const uint64_t rows[static n_entries], const uint64_t cols[static n_entries],
                                         const double *const data)
{
    hybsol_require_precision(sys, HYBSOL_PRECISION_DOUBLE);

    return hybsol_f64_add_blocks(sys, n_entries, rows, cols, data);
}

hybsol_result_t hybsol_system_add_blocks_f32(hybsol_system_t *const sys, const uint64_t n_entries,
                                             const uint64_t rows[static n_entries],
                                             const uint64_t cols[static n_entries], const float *const data)
{
    hybsol_require_precision(sys, HYBSOL_PRECISION_SINGLE);

    return hybsol_f32_add_blocks(sys, n_entries, rows, cols, data);
}

hybsol_result_t hybsol_system_block_storage(hybsol_system_t *const sys, const uint64_t row, const uint64_t col,
                                            hybsol_matrix_t *const out)
{
    CUTL_ASSERT(out != NULL, "The output pointer must not be NULL.");
    hybsol_require_precision(sys, HYBSOL_PRECISION_DOUBLE);

    return hybsol_f64_block_storage(sys, row, col, out);
}

hybsol_result_t hybsol_system_block_storage_f32(hybsol_system_t *const sys, const uint64_t row, const uint64_t col,
                                                hybsol_fmatrix_t *const out)
{
    CUTL_ASSERT(out != NULL, "The output pointer must not be NULL.");
    hybsol_require_precision(sys, HYBSOL_PRECISION_SINGLE);

    return hybsol_f32_block_storage(sys, row, col, out);
}

/* ------------------------------------------------------------------------- */
/* Row operations                                                             */
/* ------------------------------------------------------------------------- */

hybsol_result_t hybsol_system_multiply_row(hybsol_system_t *const sys, const uint64_t row, const uint64_t start_col,
                                           const hybsol_matrix_t *const mat)
{
    hybsol_require_precision(sys, HYBSOL_PRECISION_DOUBLE);

    return hybsol_f64_multiply_row(sys, row, start_col, mat);
}

hybsol_result_t hybsol_system_multiply_row_f32(hybsol_system_t *const sys, const uint64_t row, const uint64_t start_col,
                                               const hybsol_fmatrix_t *const mat)
{
    hybsol_require_precision(sys, HYBSOL_PRECISION_SINGLE);

    return hybsol_f32_multiply_row(sys, row, start_col, mat);
}

hybsol_result_t hybsol_system_eliminate_row_with(hybsol_system_t *const sys, const uint64_t row_tgt,
                                                 const uint64_t row_src, const hybsol_matrix_t *const mat)
{
    hybsol_require_precision(sys, HYBSOL_PRECISION_DOUBLE);

    return hybsol_f64_eliminate_row_with(sys, row_tgt, row_src, mat, NULL);
}

hybsol_result_t hybsol_system_eliminate_row_with_f32(hybsol_system_t *const sys, const uint64_t row_tgt,
                                                     const uint64_t row_src, const hybsol_fmatrix_t *const mat)
{
    hybsol_require_precision(sys, HYBSOL_PRECISION_SINGLE);

    return hybsol_f32_eliminate_row_with(sys, row_tgt, row_src, mat, NULL);
}

hybsol_result_t hybsol_system_eliminate_row(hybsol_system_t *const sys, const uint64_t row_tgt, const uint64_t row_src)
{
    // The multiplier is a block of the system itself, so either precision works
    if (sys->precision == HYBSOL_PRECISION_SINGLE)
        return hybsol_f32_eliminate_row(sys, row_tgt, row_src, NULL);
    return hybsol_f64_eliminate_row(sys, row_tgt, row_src, NULL);
}

hybsol_result_t hybsol_system_eliminate_row_scratch(hybsol_system_t *const sys, const uint64_t row_tgt,
                                                    const uint64_t row_src, void *const scratch)
{
    // The multiplier is a block of the system itself, so either precision works
    if (sys->precision == HYBSOL_PRECISION_SINGLE)
        return hybsol_f32_eliminate_row(sys, row_tgt, row_src, (float *)scratch);
    return hybsol_f64_eliminate_row(sys, row_tgt, row_src, (double *)scratch);
}

/* ------------------------------------------------------------------------- */
/* Dense export                                                               */
/* ------------------------------------------------------------------------- */

hybsol_result_t hybsol_system_to_dense(const hybsol_system_t *const sys, double *const out)
{
    CUTL_ASSERT(out != NULL, "The output buffer must not be NULL.");
    hybsol_require_precision(sys, HYBSOL_PRECISION_DOUBLE);

    return hybsol_f64_to_dense(sys, out);
}

hybsol_result_t hybsol_system_to_dense_f32(const hybsol_system_t *const sys, float *const out)
{
    CUTL_ASSERT(out != NULL, "The output buffer must not be NULL.");
    hybsol_require_precision(sys, HYBSOL_PRECISION_SINGLE);

    return hybsol_f32_to_dense(sys, out);
}
