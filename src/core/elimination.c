/**
 * @file core/elimination.c
 * The symbolic half of a decomposition: walk the same elimination the
 * factorization performs, carrying only column indices, so that the pattern it
 * will produce — and the memory that pattern needs — is known before a single
 * value is touched.
 *
 * No block value is read or written. Everything this allocates comes from the
 * system's allocator and is released before it returns, so the graph a caller
 * ends up holding is a single allocation the walk had no way to size until it
 * had finished.
 */

#include "internal.h"

/** How far along a row is between elimination passes. */
typedef enum
{
    TARGET_FREE,
    TARGET_IN_USE,
    TARGET_DONE,
} target_status_t;

typedef struct
{
    target_status_t status;
    /** Column of the row this one is waiting on before it can be eliminated. */
    uint64_t idx_src_needed;
} target_row_t;

/** One scheduled (row, pass) pair, with the step that pass performs. */
typedef struct
{
    uint64_t pass;
    uint64_t row;
    uint64_t step;
} occurrence_t;

/** One row's column indices while the walk runs. */
typedef struct
{
    uint64_t *cols;
    uint64_t len;
    uint64_t cap;
    /** ``cols`` is its own allocation rather than a slice of the walk's arena. */
    uint8_t owned;
} sym_row_t;

/**
 * What the walk accumulates on top of the per-row column lists.
 *
 * Owned by the walk and handed to the caller, which copies it into the graph's
 * single allocation and releases these arrays.
 */
typedef struct
{
    /**
     * One allocation holding ``rows`` and every per-row array, and each row's
     * starting columns. The walk only needs them all at once and all at the same
     * length, so there is no reason to ask for seven.
     */
    void *arena;
    sym_row_t *rows; /**< One per block row; ``rows[i].cols`` is a slice or owned. */
    target_row_t *status;
    uint64_t *ready;
    uint64_t n;
    uint64_t n_columns;
    uint64_t n_operations;
    uint64_t n_levels;
    size_t value_bytes;
    uint64_t failing_block; /**< UINT64_MAX unless an order was rejected. */
    uint64_t *row_n_elim;   /**< ``n``; eliminations a row performs. */
    uint64_t *row_level;    /**< ``n``; last pass a row is processed in. */
    uint64_t *row_step;     /**< ``n``; how many steps a row has taken. */
    /**
     * One entry per (row, pass) the walk scheduled, in walk order.
     *
     * A row's passes are not derivable from its chain: two of its sources can
     * finish in the same pass, which puts two of its steps there too, and a row
     * that is not ready again simply sits the following pass out. So the walk
     * writes down what it actually did.
     */
    occurrence_t *occurrences;
    uint64_t n_occurrences;
    uint64_t cap_occurrences;
} walk_t;

/**
 * Give row ``r`` room for ``need`` columns.
 *
 * The row's columns start life as a slice of the walk's arena, which cannot be
 * reallocated in place, so growing a row always moves it to an allocation of
 * its own. A row that never outgrows its slice never allocates again.
 */
static int sym_row_reserve(const cutl_allocator_t *const alloc, sym_row_t *const r, const uint64_t need)
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

    uint64_t *const grown = hybsol_alloc(alloc, (size_t)cap * sizeof(*grown));
    if (grown == NULL)
        return -1;
    memcpy(grown, r->cols, (size_t)r->len * sizeof(*grown));
    if (r->owned)
        hybsol_free(alloc, r->cols);
    r->cols = grown;
    r->cap = cap;
    r->owned = 1;
    return 0;
}

static void walk_release(const cutl_allocator_t *const alloc, walk_t *const w)
{
    if (w->rows != NULL)
    {
        // A row that outgrew its arena slice owns what it holds; the rest are
        // released with the arena below.
        for (uint64_t i = 0; i < w->n; ++i)
        {
            if (w->rows[i].owned)
                hybsol_free(alloc, w->rows[i].cols);
        }
    }
    hybsol_free(alloc, w->occurrences);
    hybsol_free(alloc, w->arena);
    *w = (walk_t){};
}

/**
 * Whether the diagonal block of block row ``idx`` holds nothing but zeros, which
 * no ordering could rescue.
 *
 * @return ``1`` when the block is present and entirely zero, ``0`` otherwise.
 */
static int diag_is_structurally_zero(const hybsol_system_t *const sys, const uint64_t idx)
{
    uint64_t i_entry;
    if (!hybsol_row_find(sys->rows + idx, idx, &i_entry))
        return 0;

    const unsigned char *const vals = sys->rows[idx].entries[i_entry]->vals;
    const size_t count = hybsol_block_size(sys, idx) * hybsol_block_size(sys, idx);
    if (sys->precision == HYBSOL_PRECISION_SINGLE)
    {
        for (size_t i = 0; i < count; ++i)
        {
            float value;
            memcpy(&value, vals + i * sizeof(value), sizeof(value));
            if (value != 0.0f)
                return 0;
        }
        return 1;
    }

    for (size_t i = 0; i < count; ++i)
    {
        double value;
        memcpy(&value, vals + i * sizeof(value), sizeof(value));
        if (value != 0.0)
            return 0;
    }
    return 1;
}

/**
 * The walk itself.
 *
 * Mirrors the factorization pass for pass, so a row the walk puts in a pass is
 * a row the factorization processes in that same pass. Everything the
 * factorization would have to rediscover — the final pattern, which rows are
 * ready together, how deep the elimination goes — is written down here instead.
 */
static hybsol_result_t elimination_walk(const hybsol_system_t *const sys, walk_t *const w)
{
    const uint64_t n = sys->n;
    const size_t scalar = hybsol_scalar_size(sys->precision);

    *w = (walk_t){.n = n, .failing_block = UINT64_MAX};

    // One allocation for every per-row array and every row's starting columns.
    // The lengths are all known here: n of each array, and the system's own
    // column count plus one per row for the columns.
    size_t columns = n;
    for (uint64_t i = 0; i < n; ++i)
        columns += sys->rows[i].count;

    const size_t off_rows = hybsol_align_up(0);
    const size_t off_status = off_rows + (size_t)n * sizeof(sym_row_t);
    const size_t off_ready = off_status + (size_t)n * sizeof(target_row_t);
    const size_t off_n_elim = off_ready + (size_t)n * sizeof(uint64_t);
    const size_t off_level = off_n_elim + (size_t)n * sizeof(uint64_t);
    const size_t off_step = off_level + (size_t)n * sizeof(uint64_t);
    const size_t off_cols = off_step + (size_t)n * sizeof(uint64_t);
    const size_t arena_bytes = off_cols + (size_t)columns * sizeof(uint64_t);

    void *const arena = hybsol_alloc(sys->allocator, arena_bytes);
    occurrence_t *const occurrences = hybsol_alloc(sys->allocator, (size_t)n * sizeof(*occurrences));
    if (arena == NULL || occurrences == NULL)
    {
        hybsol_free(sys->allocator, arena);
        hybsol_free(sys->allocator, occurrences);
        return HYBSOL_ERROR_OUT_OF_MEMORY;
    }
    memset(arena, 0, arena_bytes);

    sym_row_t *const rows = (sym_row_t *)((unsigned char *)arena + off_rows);
    target_row_t *const status = (target_row_t *)((unsigned char *)arena + off_status);
    uint64_t *const ready = (uint64_t *)((unsigned char *)arena + off_ready);
    uint64_t *const n_elim = (uint64_t *)((unsigned char *)arena + off_n_elim);
    uint64_t *const level = (uint64_t *)((unsigned char *)arena + off_level);
    uint64_t *const step = (uint64_t *)((unsigned char *)arena + off_step);
    uint64_t *seed = (uint64_t *)((unsigned char *)arena + off_cols);

    w->arena = arena;
    w->rows = rows;
    w->status = status;
    w->ready = ready;
    w->row_n_elim = n_elim;
    w->row_level = level;
    w->row_step = step;
    w->occurrences = occurrences;
    w->n_occurrences = 0;
    w->cap_occurrences = n;

    for (uint64_t i = 0; i < n; ++i)
    {
        rows[i].cols = seed;
        rows[i].len = sys->rows[i].count;
        rows[i].cap = rows[i].len + 1;
        rows[i].owned = 0;
        seed += rows[i].len + 1;
    }

    hybsol_result_t res = HYBSOL_SUCCESS;
    uint64_t n_operations = n; /* One diagonal solve per row. */
    uint64_t n_levels = 1;

    for (uint64_t i = 0; i < n; ++i)
    {
        for (uint64_t j = 0; j < rows[i].len; ++j)
            rows[i].cols[j] = sys->rows[i].entries[j]->col;

        // Same seed the factorization uses: a row already starting at its own
        // diagonal is finished in the very first pass; everything else waits on
        // its first column.
        if (rows[i].len && rows[i].cols[0] == i)
        {
            // Sound here because the row is diagonal-first from the outset, before any fill-in.
            if (diag_is_structurally_zero(sys, i))
            {
                w->failing_block = i;
                res = HYBSOL_ERROR_INVALID_ORDERING;
                goto done;
            }
            status[i].status = TARGET_DONE;
            n_elim[i] = 0;
            level[i] = 0;
            step[i] = 0;
            occurrences[w->n_occurrences++] = (occurrence_t){.pass = 0, .row = i, .step = 0};
        }
        else
        {
            status[i].status = TARGET_FREE;
            status[i].idx_src_needed = rows[i].cols[0];
            step[i] = 0;
        }
    }

    // The rows that are diagonal-first are processed in pass 0, so the passes of
    // this loop start at 1.
    uint64_t pass = 1;
    for (;; ++pass)
    {
        uint64_t n_ready = 0;
        for (uint64_t i = 0; i < n; ++i)
        {
            target_row_t *const st = status + i;
            if (st->status == TARGET_FREE && status[st->idx_src_needed].status == TARGET_DONE)
            {
                st->status = TARGET_IN_USE;
                ready[n_ready++] = i;
                if (w->n_occurrences == w->cap_occurrences)
                {
                    const uint64_t cap = w->cap_occurrences * 2;
                    occurrence_t *const grown =
                        hybsol_grow(sys->allocator, w->occurrences, (size_t)cap * sizeof(*grown));
                    if (grown == NULL)
                    {
                        res = HYBSOL_ERROR_OUT_OF_MEMORY;
                        goto done;
                    }
                    w->occurrences = grown;
                    w->cap_occurrences = cap;
                }
                w->occurrences[w->n_occurrences++] = (occurrence_t){.pass = pass, .row = i, .step = step[i]};
                step[i] += 1;
            }
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

            // Backward merge, identical to HYBSOL_FN(eliminate_into): walk both
            // rows from the back until neither has a column past `src`.
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
            if (sym_row_reserve(sys->allocator, &rows[tgt], needed + 1) != 0)
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
        // finished once the first entry past `src` is its own diagonal, which is
        // also the number of eliminations it has performed.
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
            {
                status[tgt].status = TARGET_DONE;
                n_elim[tgt] = at;
                CUTL_ASSERT(at == step[tgt], "Row %llu took %llu steps but settled after %llu.",
                            (unsigned long long)tgt, (unsigned long long)step[tgt], (unsigned long long)at);
                level[tgt] = pass;
                n_operations += at;
                if (pass + 1 > n_levels)
                    n_levels = pass + 1;
            }
            else
            {
                status[tgt].idx_src_needed = row->cols[at];
                status[tgt].status = TARGET_FREE;
            }
        }
    }

    // What the walk left behind is the final pattern; the factorization's
    // destination is exactly this, one carved block per entry.
    uint64_t n_columns = 0;
    size_t value_bytes = 0;
    for (uint64_t i = 0; i < n; ++i)
    {
        const sym_row_t *const row = &rows[i];
        n_columns += row->len;

        const size_t rows_of = (size_t)hybsol_block_size(sys, i);
        for (uint64_t j = 0; j < row->len; ++j)
        {
            const size_t payload =
                sizeof(hybsol_row_entry_t) + rows_of * (size_t)hybsol_block_size(sys, row->cols[j]) * scalar;
            value_bytes += hybsol_align_up(payload);
        }
    }

    w->n_columns = n_columns;
    w->n_operations = n_operations;
    w->n_levels = n_levels;
    w->value_bytes = value_bytes;
    w->n = n;

done:
    // The walk's own arrays are handed to the caller on success and released
    // here only on failure; status and ready are slices of the arena either way.
    if (res != HYBSOL_SUCCESS)
        walk_release(sys->allocator, w);
    return res;
}

uint64_t hybsol_elimination_signature(const hybsol_system_t *const sys)
{
    uint64_t hash = UINT64_C(0xcbf29ce484222325);
    for (uint64_t i = 0; i < sys->n + 1; ++i)
    {
        uint64_t value = sys->block_offsets[i];
        for (size_t byte = 0; byte < sizeof(value); ++byte)
        {
            hash ^= (value >> (byte * 8u)) & UINT64_C(0xff);
            hash *= UINT64_C(0x100000001b3);
        }
    }
    return hash;
}

hybsol_result_t hybsol_elimination_create(hybsol_system_t *const sys, hybsol_elimination_t **const out)
{
    CUTL_ASSERT(out != NULL, "The output pointer must not be NULL.");
    CUTL_ASSERT(sys != NULL, "The system must not be NULL.");
    *out = NULL;

    if (!hybsol_system_is_valid(sys))
        return HYBSOL_ERROR_SYSTEM_INVALID;

    walk_t w;
    const hybsol_result_t res = elimination_walk(sys, &w);
    if (res != HYBSOL_SUCCESS)
    {
        // There is no graph to carry the offending block, so the system does.
        if (w.failing_block != UINT64_MAX)
            sys->failing_block = w.failing_block;
        *out = NULL;
        return res;
    }

    const uint64_t n = sys->n;

    // How often a row is processed: once per elimination, plus once more if it
    // is diagonal-first from the outset and so has none.
    const uint64_t n_occupancy = w.n_occurrences;

    const size_t off_row_offset = hybsol_align_up(sizeof(hybsol_elimination_t));
    const size_t off_cols = off_row_offset + (size_t)(n + 1) * sizeof(uint64_t);
    const size_t off_level_offset = off_cols + (size_t)w.n_columns * sizeof(uint64_t);
    const size_t off_level_rows = off_level_offset + (size_t)(w.n_levels + 1) * sizeof(uint64_t);
    const size_t off_level_k = off_level_rows + (size_t)n_occupancy * sizeof(uint64_t);
    const size_t off_row_first = off_level_k + (size_t)n_occupancy * sizeof(uint64_t);
    const size_t off_row_n_elim = off_row_first + (size_t)n * sizeof(uint64_t);
    const size_t off_row_level = off_row_n_elim + (size_t)n * sizeof(uint64_t);
    // The pass-length cursor below is scratch for the bucketing, but it is the
    // same size as `level_offset` and lives exactly as long, so it rides along
    // in the frame rather than asking for a second allocation.
    const size_t off_cursor = off_row_level + (size_t)n * sizeof(uint64_t);
    const size_t total = hybsol_align_up(off_cursor + (size_t)(w.n_levels + 1) * sizeof(uint64_t));

    hybsol_elimination_t *const graph = hybsol_alloc(sys->allocator, total);
    if (graph == NULL)
    {
        walk_release(sys->allocator, &w);
        return HYBSOL_ERROR_OUT_OF_MEMORY;
    }
    *graph = (hybsol_elimination_t){0};

    unsigned char *const base = (unsigned char *)graph;
    graph->allocator = sys->allocator;
    graph->n = n;
    graph->n_columns = w.n_columns;
    graph->n_operations = w.n_operations;
    graph->n_levels = w.n_levels;
    graph->n_occupancy = n_occupancy;
    graph->value_bytes = w.value_bytes;
    graph->signature = hybsol_elimination_signature(sys);
    graph->precision = sys->precision;
    graph->failing_block = w.failing_block;
    graph->row_offset = (uint64_t *)(base + off_row_offset);
    graph->cols = (uint64_t *)(base + off_cols);
    graph->level_offset = (uint64_t *)(base + off_level_offset);
    graph->level_rows = (uint64_t *)(base + off_level_rows);
    graph->level_k = (uint64_t *)(base + off_level_k);
    graph->row_first = (uint64_t *)(base + off_row_first);
    graph->row_n_elim = (uint64_t *)(base + off_row_n_elim);
    graph->row_level = (uint64_t *)(base + off_row_level);
    graph->raw = base;
    graph->raw_bytes = total;

    uint64_t at = 0;
    for (uint64_t i = 0; i < n; ++i)
    {
        graph->row_offset[i] = at;
        memcpy(graph->cols + at, w.rows[i].cols, (size_t)w.rows[i].len * sizeof(uint64_t));
        at += w.rows[i].len;
        // Filled in from the walk's own record of which pass each step ran in.
        graph->row_first[i] = UINT64_MAX;
        graph->row_n_elim[i] = w.row_n_elim[i];
        graph->row_level[i] = w.row_level[i];
    }
    graph->row_offset[n] = at;

    // Each pass lists its rows in ascending index order, which is the order the
    // walk collected them in and therefore the order the operations come out in.
    // Note the passes a row occupies are not consecutive: it waits for its source.
    uint64_t *const cursor = (uint64_t *)(base + off_cursor);
    // The occurrences go into their passes in walk order, which is already
    // ascending by row within a pass -- the ready set is collected that way.
    memset(cursor, 0, ((size_t)w.n_levels + 1) * sizeof(*cursor));
    for (uint64_t i = 0; i < w.n_occurrences; ++i)
        ++cursor[w.occurrences[i].pass + 1];
    memcpy(graph->level_offset, cursor, ((size_t)w.n_levels + 1) * sizeof(*cursor));
    for (uint64_t pass = 0; pass < w.n_levels; ++pass)
        graph->level_offset[pass + 1] += graph->level_offset[pass];
    memcpy(cursor, graph->level_offset, ((size_t)w.n_levels + 1) * sizeof(*cursor));

    for (uint64_t i = 0; i < w.n_occurrences; ++i)
    {
        const occurrence_t *const occ = &w.occurrences[i];
        graph->level_rows[cursor[occ->pass]] = occ->row;
        graph->level_k[cursor[occ->pass]] = occ->step;
        ++cursor[occ->pass];
        if (graph->row_first[occ->row] == UINT64_MAX)
            graph->row_first[occ->row] = occ->pass;
    }

    walk_release(sys->allocator, &w);

    *out = graph;
    return HYBSOL_SUCCESS;
}

void hybsol_elimination_destroy(hybsol_elimination_t *const graph)
{
    if (graph == NULL)
        return;
    hybsol_free(graph->allocator, graph->raw);
}

hybsol_result_t hybsol_elimination_check(hybsol_system_t *const sys)
{
    CUTL_ASSERT(sys != NULL, "The system must not be NULL.");

    if (!hybsol_system_is_valid(sys))
        return HYBSOL_ERROR_SYSTEM_INVALID;

    walk_t w;
    const hybsol_result_t res = elimination_walk(sys, &w);
    if (res != HYBSOL_SUCCESS && w.failing_block != UINT64_MAX)
        sys->failing_block = w.failing_block;
    walk_release(sys->allocator, &w);
    return res;
}

uint64_t hybsol_elimination_n_blocks(const hybsol_elimination_t *const graph)
{
    CUTL_ASSERT(graph != NULL, "The graph must not be NULL.");
    return graph->n;
}

hybsol_precision_t hybsol_elimination_precision(const hybsol_elimination_t *const graph)
{
    CUTL_ASSERT(graph != NULL, "The graph must not be NULL.");
    return graph->precision;
}

uint64_t hybsol_elimination_n_columns(const hybsol_elimination_t *const graph)
{
    CUTL_ASSERT(graph != NULL, "The graph must not be NULL.");
    return graph->n_columns;
}

uint64_t hybsol_elimination_n_operations(const hybsol_elimination_t *const graph)
{
    CUTL_ASSERT(graph != NULL, "The graph must not be NULL.");
    return graph->n_operations;
}

uint64_t hybsol_elimination_n_levels(const hybsol_elimination_t *const graph)
{
    CUTL_ASSERT(graph != NULL, "The graph must not be NULL.");
    return graph->n_levels;
}

size_t hybsol_elimination_value_bytes(const hybsol_elimination_t *const graph)
{
    CUTL_ASSERT(graph != NULL, "The graph must not be NULL.");
    return graph->value_bytes;
}

size_t hybsol_elimination_total_bytes(const hybsol_elimination_t *const graph)
{
    CUTL_ASSERT(graph != NULL, "The graph must not be NULL.");
    return graph->raw_bytes;
}

uint64_t hybsol_elimination_failing_block(const hybsol_elimination_t *const graph)
{
    CUTL_ASSERT(graph != NULL, "The graph must not be NULL.");
    return graph->failing_block;
}

uint64_t hybsol_elimination_row_length(const hybsol_elimination_t *const graph, const uint64_t row)
{
    CUTL_ASSERT(graph != NULL, "The graph must not be NULL.");
    if (row >= graph->n)
        return 0;
    return graph->row_offset[row + 1] - graph->row_offset[row];
}

uint64_t hybsol_elimination_row_n_eliminations(const hybsol_elimination_t *const graph, const uint64_t row)
{
    CUTL_ASSERT(graph != NULL, "The graph must not be NULL.");
    CUTL_ASSERT(row < graph->n, "Block row %llu is outside [0, %llu).", (unsigned long long)row,
                (unsigned long long)graph->n);
    return graph->row_n_elim[row];
}

uint64_t hybsol_elimination_row_first_level(const hybsol_elimination_t *const graph, const uint64_t row)
{
    CUTL_ASSERT(graph != NULL, "The graph must not be NULL.");
    CUTL_ASSERT(row < graph->n, "Block row %llu is outside [0, %llu).", (unsigned long long)row,
                (unsigned long long)graph->n);
    return graph->row_first[row];
}

uint64_t hybsol_elimination_row_level(const hybsol_elimination_t *const graph, const uint64_t row)
{
    CUTL_ASSERT(graph != NULL, "The graph must not be NULL.");
    CUTL_ASSERT(row < graph->n, "Block row %llu is outside [0, %llu).", (unsigned long long)row,
                (unsigned long long)graph->n);
    return graph->row_level[row];
}

hybsol_result_t hybsol_elimination_row_columns(const hybsol_elimination_t *const graph, const uint64_t row,
                                               uint64_t *const out, const uint64_t capacity, uint64_t *const n_written)
{
    CUTL_ASSERT(graph != NULL, "The graph must not be NULL.");
    CUTL_ASSERT(out != NULL, "The output array must not be NULL.");
    CUTL_ASSERT(row < graph->n, "Block row %llu is outside [0, %llu).", (unsigned long long)row,
                (unsigned long long)graph->n);

    const uint64_t length = graph->row_offset[row + 1] - graph->row_offset[row];
    CUTL_ASSERT(capacity >= length, "The row holds %llu columns but only %llu slots were given.",
                (unsigned long long)length, (unsigned long long)capacity);

    memcpy(out, graph->cols + graph->row_offset[row], (size_t)length * sizeof(*out));
    if (n_written != NULL)
        *n_written = length;
    return HYBSOL_SUCCESS;
}

uint64_t hybsol_elimination_level_size(const hybsol_elimination_t *const graph, const uint64_t level)
{
    CUTL_ASSERT(graph != NULL, "The graph must not be NULL.");
    if (level >= graph->n_levels)
        return 0;
    return graph->level_offset[level + 1] - graph->level_offset[level];
}

hybsol_result_t hybsol_elimination_level_rows(const hybsol_elimination_t *const graph, const uint64_t level,
                                              uint64_t *const out, const uint64_t capacity, uint64_t *const n_written)
{
    CUTL_ASSERT(graph != NULL, "The graph must not be NULL.");
    CUTL_ASSERT(out != NULL, "The output array must not be NULL.");
    CUTL_ASSERT(level < graph->n_levels, "Pass %llu is outside [0, %llu).", (unsigned long long)level,
                (unsigned long long)graph->n_levels);

    const uint64_t at = graph->level_offset[level];
    const uint64_t length = graph->level_offset[level + 1] - at;
    CUTL_ASSERT(capacity >= length, "The pass holds %llu rows but only %llu slots were given.",
                (unsigned long long)length, (unsigned long long)capacity);

    memcpy(out, graph->level_rows + at, (size_t)length * sizeof(*out));
    if (n_written != NULL)
        *n_written = length;
    return HYBSOL_SUCCESS;
}
