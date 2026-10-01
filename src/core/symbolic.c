/**
 * @file core/symbolic.c
 * The symbolic half of a decomposition: walk the same elimination the
 * factorization performs, carrying only column indices, so the fill-in it will
 * produce can be sized -- and the rows grown -- before any thread starts.
 *
 * No block value is touched. Everything this allocates comes from the system's
 * allocator, so a caller tracking allocations still sees it all, and it all
 * happens on one thread before the first parallel region.
 */

#include "internal.h"

/* One row's column indices, plus the longest the row ever gets. */
typedef struct
{
    uint64_t *cols;
    uint64_t len;
    uint64_t cap;
    /** Longest ``needed`` this row reaches; not the same as its final length. */
    uint64_t max_needed;
} sym_row_t;

static int sym_row_reserve(hybsol_system_t *const sys, sym_row_t *const r, const uint64_t need)
{
    if (need <= r->cap)
        return 0;

    uint64_t cap = r->cap ? r->cap : 8;
    while (cap < need)
    {
        if (cap > UINT64_MAX / 2)
        {
            cap = need;
            break;
        }
        cap *= 2;
    }

    uint64_t *const grown = hybsol_grow(sys->allocator, r->cols, (size_t)cap * sizeof(*grown));
    if (grown == NULL)
        return -1;
    r->cols = grown;
    r->cap = cap;
    return 0;
}

static void sym_rows_free(hybsol_system_t *const sys, sym_row_t *const rows, const uint64_t n)
{
    for (uint64_t i = 0; i < n; ++i)
        hybsol_free(sys->allocator, rows[i].cols);
    hybsol_free(sys->allocator, rows);
}

/**
 * The walk itself.
 *
 * ``peak`` may be ``NULL``; when given it receives the longest length each row
 * reaches, which is what the pre-grow step needs.
 */
static hybsol_result_t fill_walk(hybsol_system_t *const sys, hybsol_fill_plan_t *const out, uint64_t *const peak)
{
    const uint64_t n = sys->n;
    const size_t scalar = hybsol_scalar_size(sys->precision);

    *out = (hybsol_fill_plan_t){.operations = hybsol_operation_bound(n)};

    sym_row_t *const rows = hybsol_alloc(sys->allocator, (size_t)n * sizeof(*rows));
    target_row_t *const status = hybsol_alloc(sys->allocator, (size_t)n * sizeof(*status));
    uint64_t *const ready = hybsol_alloc(sys->allocator, (size_t)n * sizeof(*ready));
    if (rows == NULL || status == NULL || ready == NULL)
    {
        hybsol_free(sys->allocator, rows);
        hybsol_free(sys->allocator, status);
        hybsol_free(sys->allocator, ready);
        return HYBSOL_ERROR_OUT_OF_MEMORY;
    }
    memset(rows, 0, (size_t)n * sizeof(*rows));

    hybsol_result_t res = HYBSOL_SUCCESS;
    size_t pool_bytes = 0;
    size_t array_bytes = 0;

    for (uint64_t i = 0; i < n; ++i)
    {
        rows[i].len = sys->rows[i].count;
        rows[i].max_needed = rows[i].len;
        if (sym_row_reserve(sys, &rows[i], rows[i].len + 1) != 0)
        {
            res = HYBSOL_ERROR_OUT_OF_MEMORY;
            goto done;
        }
        for (uint64_t j = 0; j < rows[i].len; ++j)
            rows[i].cols[j] = sys->rows[i].entries[j]->col;

        // Same seed the factorization uses: a row already starting at its own
        // diagonal is finished; everything else waits on its first column.
        if (rows[i].len && rows[i].cols[0] == i)
            status[i].status = TARGET_DONE;
        else
        {
            status[i].status = TARGET_FREE;
            status[i].idx_src_needed = rows[i].cols[0];
        }
    }

    for (;;)
    {
        uint64_t n_ready = 0;
        for (uint64_t i = 0; i < n; ++i)
            if (status[i].status == TARGET_FREE && status[status[i].idx_src_needed].status == TARGET_DONE)
            {
                status[i].status = TARGET_IN_USE;
                ready[n_ready++] = i;
            }
        if (n_ready == 0)
            break;

        for (uint64_t k = 0; k < n_ready; ++k)
        {
            const uint64_t tgt = ready[k];
            const uint64_t src = status[tgt].idx_src_needed;
            const sym_row_t src_row = rows[src];

            if (rows[tgt].len == 0 || src_row.len == 0)
            {
                res = HYBSOL_ERROR_EMPTY_ROW;
                goto done;
            }

            // Backward merge, identical to HYBSOL_FN(eliminate_row_with): walk
            // both rows from the back until neither has a column past `src`.
            uint64_t pos_tgt = rows[tgt].len, pos_src = src_row.len, unique = 0;
            for (;; ++unique)
            {
                const uint64_t col_tgt = rows[tgt].cols[pos_tgt - 1];
                const uint64_t col_src = src_row.cols[pos_src - 1];
                if (col_tgt <= src && col_src <= src)
                    break;

                if (col_tgt == col_src)
                {
                    pos_tgt -= 1;
                    pos_src -= 1;
                }
                else if (col_src > col_tgt)
                    pos_src -= 1; // a column only the source has: fill-in
                else
                    pos_tgt -= 1;
            }

            const uint64_t needed = unique + pos_tgt;
            if (needed > rows[tgt].max_needed)
                rows[tgt].max_needed = needed;
            if (sym_row_reserve(sys, &rows[tgt], needed + 1) != 0)
            {
                res = HYBSOL_ERROR_OUT_OF_MEMORY;
                goto done;
            }

            // Ascending union of the two suffixes, filled from the back for the
            // same reason the factorization does: writing forward would clobber
            // a column that has not been read yet.
            uint64_t at = needed, a = rows[tgt].len, b = src_row.len;
            for (uint64_t i = 0; i < unique; ++i)
            {
                const uint64_t col_tgt = rows[tgt].cols[a - 1];
                const uint64_t col_src = src_row.cols[b - 1];

                if (col_tgt == col_src)
                {
                    rows[tgt].cols[at - 1] = col_tgt;
                    --a;
                    --b;
                }
                else if (col_src > col_tgt)
                {
                    // A column only the source has: this is the fill-in.
                    const size_t payload = sizeof(hybsol_row_entry_t) + (size_t)hybsol_block_size(sys, tgt) *
                                                                            (size_t)hybsol_block_size(sys, col_src) *
                                                                            scalar;
                    pool_bytes += HYBSOL_POOL_HDR_BYTES + hybsol_pool_align(payload);
                    rows[tgt].cols[at - 1] = col_src;
                    --b;
                }
                else
                {
                    rows[tgt].cols[at - 1] = col_tgt;
                    --a;
                }
                --at;
            }
            rows[tgt].len = needed;
        }

        // Settle each pass the way the factorization settles it: the row is
        // finished once the first entry past `src` is its own diagonal.
        for (uint64_t k = 0; k < n_ready; ++k)
        {
            const uint64_t tgt = ready[k];
            const uint64_t src = status[tgt].idx_src_needed;
            const sym_row_t *const row = &rows[tgt];

            uint64_t at = 0;
            while (at < row->len && row->cols[at] <= src)
                ++at;
            if (at >= row->len)
            {
                res = HYBSOL_ERROR_INTERNAL;
                goto done;
            }

            if (row->cols[at] == tgt)
                status[tgt].status = TARGET_DONE;
            else
            {
                status[tgt].idx_src_needed = row->cols[at];
                status[tgt].status = TARGET_FREE;
            }
        }
    }

    for (uint64_t i = 0; i < n; ++i)
    {
        if (peak != NULL)
            peak[i] = rows[i].max_needed;

        if (rows[i].max_needed <= sys->rows[i].capacity)
            continue;
        uint64_t cap = sys->rows[i].capacity ? sys->rows[i].capacity : 8;
        while (cap < rows[i].max_needed)
            cap *= 2;
        array_bytes += hybsol_pool_align((size_t)cap * sizeof(*sys->rows[i].entries));
    }

    out->pool_bytes = pool_bytes;
    out->array_bytes = array_bytes;

done:
    sym_rows_free(sys, rows, n);
    hybsol_free(sys->allocator, status);
    hybsol_free(sys->allocator, ready);
    return res;
}

hybsol_result_t hybsol_fill_plan_compute(const hybsol_system_t *const sys, hybsol_fill_plan_t *const out)
{
    CUTL_ASSERT(out != NULL, "The output plan must not be NULL.");
    CUTL_ASSERT(sys != NULL, "The system must not be NULL.");

    if (sys->decomposed)
        return HYBSOL_ERROR_ALREADY_DECOMPOSED;
    if (!hybsol_system_is_valid(sys))
        return HYBSOL_ERROR_SYSTEM_INVALID;

    return fill_walk((hybsol_system_t *)sys, out, NULL);
}

hybsol_result_t hybsol_fill_plan_prepare(hybsol_system_t *const sys, hybsol_fill_plan_t *const out)
{
    CUTL_ASSERT(out != NULL, "The output plan must not be NULL.");

    if (sys->decomposed)
        return HYBSOL_ERROR_ALREADY_DECOMPOSED;
    if (!hybsol_system_is_valid(sys))
        return HYBSOL_ERROR_SYSTEM_INVALID;

    uint64_t *const peak = hybsol_alloc(sys->allocator, (size_t)sys->n * sizeof(*peak));
    if (peak == NULL)
        return HYBSOL_ERROR_OUT_OF_MEMORY;

    const hybsol_result_t res = fill_walk(sys, out, peak);

    if (res == HYBSOL_SUCCESS)
    {
        // Grow every row to its peak now, on this thread. Once done, the row
        // reserves inside the elimination all find enough capacity and return
        // early, so nothing in a parallel region ever releases anything.
        for (uint64_t i = 0; i < sys->n; ++i)
        {
            if (peak[i] <= sys->rows[i].capacity)
                continue;
            const hybsol_result_t grown = hybsol_row_reserve(sys, sys->rows + i, peak[i]);
            if (grown != HYBSOL_SUCCESS)
            {
                hybsol_free(sys->allocator, peak);
                return grown;
            }
        }
    }

    hybsol_free(sys->allocator, peak);
    return res;
}
