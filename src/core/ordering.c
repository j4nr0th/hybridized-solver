/** @file core/ordering.c
 * Coloring-based reordering of blocks, and the vector shuffles that go with it.
 */

#include "internal.h"

/* ------------------------------------------------------------------------- */
/* Coloring                                                                   */
/* ------------------------------------------------------------------------- */

static void order_from_colors(const uint64_t n, uint64_t ordering[const static n], uint64_t *const colored_by_color,
                              const uint64_t color_count, uint64_t *const counts)
{
    for (uint64_t i = 0; i < color_count; ++i)
        colored_by_color[i] = i;

    // Sort the color counts with the highest count first (bubble sort)
    for (uint64_t i = 0; i + 1 < color_count; ++i)
    {
        for (uint64_t j = i + 1; j < color_count; ++j)
        {
            if (counts[i] < counts[j])
            {
                const uint64_t tmp = counts[i];
                counts[i] = counts[j];
                counts[j] = tmp;
                const uint64_t tmp2 = colored_by_color[i];
                colored_by_color[i] = colored_by_color[j];
                colored_by_color[j] = tmp2;
            }
        }
    }

    // Turn the counts into offsets
    uint64_t total_count = 0;
    for (uint64_t i = 0; i < color_count; ++i)
    {
        const uint64_t current_count = counts[i];
        counts[i] = total_count;
        total_count += current_count;
    }

    // Re-sort the offsets so they are ordered by color index again
    for (uint64_t i = 0; i + 1 < color_count; ++i)
    {
        for (uint64_t j = i + 1; j < color_count; ++j)
        {
            if (colored_by_color[i] > colored_by_color[j])
            {
                const uint64_t tmp = counts[i];
                counts[i] = counts[j];
                counts[j] = tmp;
                const uint64_t tmp2 = colored_by_color[i];
                colored_by_color[i] = colored_by_color[j];
                colored_by_color[j] = tmp2;
            }
        }
    }

    for (uint64_t i = 0; i < color_count; ++i)
        colored_by_color[i] = 0;

    // Assign each block its final index from its color and its rank within it
    for (uint64_t i = 0; i < n; ++i)
    {
        const uint64_t color = ordering[i];
        const uint64_t local_offset = colored_by_color[color]++;
        ordering[i] = local_offset + counts[color];
    }
}

static hybsol_result_t update_color_counts(const uint64_t max_colors, uint64_t *const color_count,
                                           uint64_t *const counts, const uint64_t color)
{
    if (color == *color_count)
    {
        if (max_colors == *color_count)
            return HYBSOL_ERROR_MAX_COLORS;

        counts[*color_count] = 1;
        *color_count += 1;
    }
    else
    {
        counts[color] += 1;
    }

    return HYBSOL_SUCCESS;
}

static hybsol_result_t compute_ordering_first(const uint64_t n, const hybsol_row_t *const rows,
                                              uint64_t *const ordering, const uint64_t max_colors,
                                              uint64_t *const color_buffer)
{
    for (uint64_t i = 0; i < n; ++i)
        ordering[i] = ~0ULL;

    uint64_t color_count = 0;
    uint64_t *const counts = color_buffer;

    for (uint64_t i = 0; i < n; ++i)
    {
        const hybsol_row_t *const row = rows + i;
        const uint64_t block_count = row->count;
        uint64_t color = 0;

        // Pick the first color that none of the neighbors uses
    color_check:
        for (uint64_t j = 0; j < block_count && color < color_count; ++j)
        {
            if (ordering[row->entries[j]->col] == color)
            {
                color += 1;
                goto color_check;
            }
        }
        HYBSOL_ASSERT(color <= color_count, "Color chosen for row %llu was %llu, which is out of bounds!",
                      (unsigned long long)i, (unsigned long long)color);

        ordering[i] = color;
        const hybsol_result_t res = update_color_counts(max_colors, &color_count, counts, color);
        if (res != HYBSOL_SUCCESS)
            return res;
    }

    order_from_colors(n, ordering, color_buffer + max_colors, color_count, counts);
    return HYBSOL_SUCCESS;
}

static hybsol_result_t compute_ordering_ranked(const uint64_t n, const hybsol_row_t *const rows,
                                               uint64_t *const ordering, const uint64_t max_colors,
                                               uint64_t *const color_buffer, const int most_used_first)
{
    for (uint64_t i = 0; i < n; ++i)
        ordering[i] = ~0ULL;

    uint64_t color_count = 0;
    uint64_t *const counts = color_buffer;
    uint64_t *const color_order = color_buffer + max_colors;
    for (uint64_t i = 0; i < max_colors; ++i)
    {
        color_order[i] = i;
        counts[i] = 0;
    }

    for (uint64_t i = 0; i < n; ++i)
    {
        const hybsol_row_t *const row = rows + i;
        const uint64_t block_count = row->count;
        uint64_t i_color = 0;

        // Walk down the ordered list of colors until one is free
    color_check:
        for (uint64_t j = 0; j < block_count && i_color < color_count; ++j)
        {
            if (ordering[row->entries[j]->col] == color_order[i_color])
            {
                i_color += 1;
                goto color_check;
            }
        }
        const uint64_t color = color_order[i_color];
        HYBSOL_ASSERT(i_color <= color_count, "Color chosen for row %llu was %llu, which is out of bounds!",
                      (unsigned long long)i, (unsigned long long)color);

        const hybsol_result_t res = update_color_counts(max_colors, &color_count, counts, color);
        if (res != HYBSOL_SUCCESS)
            return res;

        ordering[i] = color;

        // The color just grew, so move it back into order
        if (most_used_first)
        {
            for (uint64_t j = i_color; j > 0; --j)
            {
                if (counts[color_order[j]] > counts[color_order[j - 1]])
                {
                    const uint64_t tmp = color_order[j];
                    color_order[j] = color_order[j - 1];
                    color_order[j - 1] = tmp;
                }
                else
                {
                    break;
                }
            }
        }
        else
        {
            for (uint64_t j = i_color + 1; j < color_count; ++j)
            {
                if (counts[color_order[j]] < counts[color_order[j - 1]])
                {
                    const uint64_t tmp = color_order[j];
                    color_order[j] = color_order[j - 1];
                    color_order[j - 1] = tmp;
                }
                else
                {
                    break;
                }
            }
        }
    }

#if HYBSOL_ENABLE_ASSERTS
    for (uint64_t color = 0; color < color_count; ++color)
    {
        uint64_t counted = 0;
        for (uint64_t i = 0; i < n; ++i)
            counted += (ordering[i] == color);
        HYBSOL_ASSERT(counted == counts[color], "Color %llu had %llu entries, but counted %llu!",
                      (unsigned long long)color, (unsigned long long)counts[color], (unsigned long long)counted);
    }
#endif

    order_from_colors(n, ordering, color_buffer + max_colors, color_count, counts);
    return HYBSOL_SUCCESS;
}

hybsol_result_t hybsol_system_compute_reordering(const hybsol_system_t *const sys,
                                                 const hybsol_ordering_strategy_t strategy, uint64_t max_colors,
                                                 uint64_t *const out_ordering)
{
    if (out_ordering == NULL)
        return HYBSOL_ERROR_INVALID_ARGUMENT;

    const uint64_t n = sys->n;
    if (n == 0)
        return HYBSOL_SUCCESS;

    // The number of colors can never exceed the densest row
    if (max_colors == 0)
    {
        for (uint64_t i = 0; i < n; ++i)
        {
            const uint64_t count = sys->rows[i].count;
            if (count > max_colors)
                max_colors = count;
        }
        if (max_colors == 0)
            max_colors = 1;
    }

    uint64_t *const color_buffer = hybsol_alloc(sizeof(*color_buffer) * (size_t)max_colors * 2);
    if (color_buffer == NULL)
        return HYBSOL_ERROR_OUT_OF_MEMORY;

    hybsol_result_t res;
    switch (strategy)
    {
    case HYBSOL_ORDERING_FIRST:
        res = compute_ordering_first(n, sys->rows, out_ordering, max_colors, color_buffer);
        break;
    case HYBSOL_ORDERING_GREEDY:
        res = compute_ordering_ranked(n, sys->rows, out_ordering, max_colors, color_buffer, 1);
        break;
    case HYBSOL_ORDERING_BALANCED:
        res = compute_ordering_ranked(n, sys->rows, out_ordering, max_colors, color_buffer, 0);
        break;
    default:
        res = HYBSOL_ERROR_INVALID_ARGUMENT;
        break;
    }

    hybsol_free(color_buffer);
    return res;
}

/* ------------------------------------------------------------------------- */
/* Reordering the system                                                      */
/* ------------------------------------------------------------------------- */

hybsol_result_t hybsol_system_reorder_blocks(hybsol_system_t *const sys, const uint64_t *const new_order,
                                             const uint64_t n_threads)
{
    hybsol_result_t res = hybsol_require_mutable(sys);
    if (res != HYBSOL_SUCCESS)
        return res;
    if (new_order == NULL)
        return HYBSOL_ERROR_INVALID_ARGUMENT;

    const uint64_t n = sys->n;

    // The ordering has to be a permutation of [0, n)
    uint8_t *const seen = hybsol_alloc((size_t)n * sizeof(*seen));
    if (seen == NULL)
        return HYBSOL_ERROR_OUT_OF_MEMORY;
    memset(seen, 0, (size_t)n * sizeof(*seen));
    for (uint64_t i = 0; i < n; ++i)
    {
        if (new_order[i] >= n || seen[new_order[i]])
        {
            hybsol_free(seen);
            return HYBSOL_ERROR_INVALID_ARGUMENT;
        }
        seen[new_order[i]] = 1;
    }
    hybsol_free(seen);

    // A scratch buffer per thread lets every row be re-sorted in parallel
    // without allocating from inside the parallel region.
    uint64_t max_entries = 0;
    for (uint64_t i = 0; i < n; ++i)
        if (sys->rows[i].count > max_entries)
            max_entries = sys->rows[i].count;
    if (max_entries == 0)
        max_entries = 1;

    const int threads = hybsol_resolve_threads(n_threads);
    hybsol_row_entry_t ***const scratch = hybsol_alloc((size_t)threads * sizeof(*scratch));
    hybsol_row_t *const new_rows = hybsol_alloc((size_t)n * sizeof(*new_rows));
    uint64_t *const new_sizes = hybsol_alloc((size_t)n * sizeof(*new_sizes));
    uint8_t *const new_flags = hybsol_alloc((size_t)n * sizeof(*new_flags));
    if (scratch == NULL || new_rows == NULL || new_sizes == NULL || new_flags == NULL)
    {
        hybsol_free(scratch);
        hybsol_free(new_rows);
        hybsol_free(new_sizes);
        hybsol_free(new_flags);
        return HYBSOL_ERROR_OUT_OF_MEMORY;
    }
    for (int t = 0; t < threads; ++t)
        scratch[t] = NULL;
    for (int t = 0; t < threads; ++t)
    {
        scratch[t] = hybsol_alloc((size_t)max_entries * sizeof(*scratch[t]));
        if (scratch[t] == NULL)
        {
            for (int u = 0; u < threads; ++u)
                hybsol_free(scratch[u]);
            hybsol_free(scratch);
            hybsol_free(new_rows);
            hybsol_free(new_sizes);
            hybsol_free(new_flags);
            return HYBSOL_ERROR_OUT_OF_MEMORY;
        }
    }

#pragma omp parallel default(none) shared(sys, n, new_order, new_rows, new_sizes, new_flags, scratch) if (threads > 1) \
    num_threads(threads)
    {
        hybsol_row_entry_t **const entry_buffer = scratch[HYBSOL_THREAD_NUM()];

        // Remap every entry's column to its new index, then sort the row again
#pragma omp for schedule(dynamic, 1)
        for (uint64_t idx_row = 0; idx_row < n; ++idx_row)
        {
            hybsol_row_t *const row = sys->rows + idx_row;
            for (uint64_t j = 0; j < row->count; ++j)
            {
                hybsol_row_entry_t *const entry = row->entries[j];
                entry_buffer[j] = entry;
                entry->col = new_order[entry->col];
            }

            uint64_t inserted = 0;
            while (inserted < row->count)
            {
                hybsol_row_entry_t *smallest = NULL;
                for (uint64_t j = 0; j < row->count; ++j)
                {
                    hybsol_row_entry_t *const entry = entry_buffer[j];
                    if (entry != NULL && (smallest == NULL || entry->col < smallest->col))
                    {
                        entry_buffer[j] = smallest;
                        smallest = entry;
                    }
                }
                HYBSOL_ASSERT(smallest != NULL, "There should be at least one more in the array.");
                row->entries[inserted] = smallest;
                inserted += 1;
            }

#if HYBSOL_ENABLE_ASSERTS
            for (uint64_t j = 1; j < row->count; ++j)
            {
                HYBSOL_ASSERT(row->entries[j - 1]->col < row->entries[j]->col,
                              "Entries should be sorted by column, but in row %llu entries %llu and %llu had column "
                              "indices %llu and %llu.",
                              (unsigned long long)idx_row, (unsigned long long)(j - 1), (unsigned long long)j,
                              (unsigned long long)row->entries[j - 1]->col, (unsigned long long)row->entries[j]->col);
            }
#endif
        }

        // Move the rows, their diagonal state and their sizes into place
#pragma omp for schedule(static)
        for (uint64_t i = 0; i < n; ++i)
        {
            const uint64_t new_idx = new_order[i];
            new_rows[new_idx] = sys->rows[i];
            new_flags[new_idx] = sys->diag_decomposed[i];
            new_sizes[new_idx] = hybsol_block_size(sys, i);
        }

#pragma omp for schedule(static)
        for (uint64_t i = 0; i < n; ++i)
        {
            sys->rows[i] = new_rows[i];
            sys->diag_decomposed[i] = new_flags[i];
        }
    }

    for (int t = 0; t < threads; ++t)
        hybsol_free(scratch[t]);
    hybsol_free(scratch);
    hybsol_free(new_rows);
    hybsol_free(new_flags);

    // Turn the shuffled sizes back into offsets
    sys->block_offsets[0] = 0;
    for (uint64_t i = 0; i < n; ++i)
        sys->block_offsets[i + 1] = sys->block_offsets[i] + new_sizes[i];

    hybsol_free(new_sizes);
    return HYBSOL_SUCCESS;
}

/* ------------------------------------------------------------------------- */
/* Vector shuffling                                                           */
/* ------------------------------------------------------------------------- */

void hybsol_system_reorder_vector(const hybsol_system_t *const sys, const uint64_t *const new_order,
                                  const double *const in, double *const out)
{
    uint64_t offset = 0;
    for (uint64_t i_block = 0; i_block < sys->n; ++i_block)
    {
        const uint64_t new_offset = sys->block_offsets[new_order[i_block]];
        const uint64_t block_size = hybsol_block_size(sys, new_order[i_block]);
#pragma omp simd
        for (uint64_t i_entry = 0; i_entry < block_size; ++i_entry)
            out[offset + i_entry] = in[new_offset + i_entry];
        offset += block_size;
    }
}

void hybsol_system_unorder_vector(const hybsol_system_t *const sys, const uint64_t *const new_order,
                                  const double *const in, double *const out)
{
    uint64_t offset = 0;
    for (uint64_t i_block = 0; i_block < sys->n; ++i_block)
    {
        const uint64_t new_offset = sys->block_offsets[new_order[i_block]];
        const uint64_t block_size = hybsol_block_size(sys, new_order[i_block]);
#pragma omp simd
        for (uint64_t i_entry = 0; i_entry < block_size; ++i_entry)
            out[new_offset + i_entry] = in[offset + i_entry];
        offset += block_size;
    }
}
