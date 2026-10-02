/**
 * @file core/decomposition.c
 * The destination a decomposition writes into: laying it out from a graph,
 * filling it with the system's blocks, and walking the graph's passes to
 * factorize it. Nothing allocates once the destination exists -- the graph
 * fixed the pattern, so every block the factorization touches has a slot.
 */

#include "internal.h"

/* ------------------------------------------------------------------------- */
/* Factorization scratch                                                      */
/* ------------------------------------------------------------------------- */

/** Distinguishes a bound workspace from an arbitrary or stale buffer. */
#define HYBSOL_WORKSPACE_MAGIC UINT64_C(0x687962736f6c7731) /* "hybsolw1" */

/**
 * One result slot per block row and one scratch block per thread in one aligned
 * buffer the caller owns: size it with :c:func:`hybsol_workspace_bytes` and
 * nothing allocates before the parallel regions.
 */
typedef struct hybsol_workspace
{
    uint64_t magic;
    /** ``dec->n`` and the thread count the buffer was sized for. */
    uint64_t n;
    uint64_t n_threads;
    /** Bytes each per-thread scratch block holds. */
    size_t scratch_stride;
    /** Total the caller had to supply. */
    size_t total_bytes;
    /** Byte offsets of each array within the buffer. */
    size_t off_results;
    size_t off_scratch;
} hybsol_workspace_t;

/** The largest block either dimension has, which bounds every intermediate. */
static uint64_t max_block_size(const uint64_t *const block_offsets, const uint64_t n)
{
    uint64_t largest = 1;
    for (uint64_t i = 0; i < n; ++i)
    {
        const uint64_t size = block_offsets[i + 1] - block_offsets[i];
        if (size > largest)
            largest = size;
    }
    return largest;
}

/** Bytes of scratch a factorization of this shape needs. */
static size_t workspace_size(const uint64_t n, const uint64_t *const block_offsets, const size_t scalar,
                             const uint64_t threads)
{
    const size_t stride =
        hybsol_align_up((size_t)max_block_size(block_offsets, n) * (size_t)max_block_size(block_offsets, n) * scalar);

    size_t total = sizeof(hybsol_workspace_t);
    total += hybsol_align_up((size_t)n * sizeof(hybsol_result_t));
    total += stride * threads;
    return hybsol_align_up(total);
}

size_t hybsol_workspace_bytes(const hybsol_system_t *const sys, const uint64_t n_threads)
{
    CUTL_ASSERT(sys != NULL, "The system must not be NULL.");

    const uint64_t threads = (uint64_t)hybsol_resolve_threads(n_threads);
    return workspace_size(sys->n, sys->block_offsets, hybsol_scalar_size(sys->precision), threads);
}

/**
 * Point the header at the arrays laid out behind it, matching the arithmetic
 * :c:func:`hybsol_workspace_bytes` used to size the buffer.
 */
static void workspace_bind(hybsol_workspace_t *const ws, const size_t buffer_bytes,
                           const hybsol_decomposition_t *const dec, const uint64_t threads)
{
    ws->magic = HYBSOL_WORKSPACE_MAGIC;
    ws->n = dec->n;
    ws->n_threads = threads;
    ws->scratch_stride =
        hybsol_align_up((size_t)max_block_size(dec->block_offsets, dec->n) *
                        (size_t)max_block_size(dec->block_offsets, dec->n) * hybsol_scalar_size(dec->precision));
    ws->total_bytes = buffer_bytes;

    size_t at = hybsol_align_up(sizeof(hybsol_workspace_t));
    ws->off_results = at;
    at += hybsol_align_up((size_t)dec->n * sizeof(hybsol_result_t));
    ws->off_scratch = at;
}

/** Scratch block belonging to the calling thread; a pure offset into ``ws``. */
static void *workspace_scratch(const hybsol_workspace_t *const ws)
{
    const uint64_t t = (uint64_t)HYBSOL_THREAD_NUM();
    if (ws == NULL || t >= ws->n_threads)
        return NULL;

    const unsigned char *const base = (const unsigned char *)ws;
    return (void *)(base + ws->off_scratch + (size_t)t * ws->scratch_stride);
}

/* ------------------------------------------------------------------------- */
/* The destination                                                            */
/* ------------------------------------------------------------------------- */

void hybsol_decomposition_destroy(hybsol_decomposition_t *const dec)
{
    if (dec == NULL)
        return;
    // A decomposition laid out by hybsol_decomposition_init owns no memory.
    if (dec->raw != NULL)
        hybsol_free(dec->allocator, dec->raw);
}

/** Where each array of a decomposition's frame sits, in layout order. */
typedef struct
{
    size_t total;
    size_t offsets;
    size_t level_offset;
    size_t level_rows;
    size_t level_k;
    size_t row_n_elim;
    size_t row_level;
    size_t rows;
    size_t entries;
    size_t values;
} decomp_layout_t;

/** Lay out a decomposition's frame -- a function of the graph alone, so a caller can size it up front. */
static decomp_layout_t decomp_layout(const hybsol_elimination_t *const graph)
{
    const size_t n = (size_t)graph->n;
    const size_t n_levels = (size_t)graph->n_levels;
    const size_t n_occupancy = (size_t)graph->n_occupancy;

    decomp_layout_t layout;
    layout.offsets = hybsol_align_up(sizeof(hybsol_decomposition_t));
    layout.level_offset = layout.offsets + (n + 1) * sizeof(uint64_t);
    layout.level_rows = layout.level_offset + (n_levels + 1) * sizeof(uint64_t);
    layout.level_k = layout.level_rows + n_occupancy * sizeof(uint64_t);
    layout.row_n_elim = layout.level_k + n_occupancy * sizeof(uint64_t);
    layout.row_level = layout.row_n_elim + n * sizeof(uint64_t);
    layout.rows = layout.row_level + n * sizeof(uint64_t);
    layout.entries = layout.rows + n * sizeof(hybsol_row_t);
    layout.values = hybsol_align_up(layout.entries + (size_t)graph->n_columns * sizeof(hybsol_row_entry_t *));
    layout.total = layout.values + graph->value_bytes;
    return layout;
}

size_t hybsol_decomposition_bytes(const hybsol_elimination_t *const graph)
{
    CUTL_ASSERT(graph != NULL, "The graph must not be NULL.");
    return decomp_layout(graph).total;
}

hybsol_result_t hybsol_decomposition_init(const hybsol_system_t *const sys, const hybsol_elimination_t *const graph,
                                          void *const storage, hybsol_decomposition_t **const out)
{
    CUTL_ASSERT(storage != NULL, "The storage buffer must not be NULL.");
    CUTL_ASSERT(out != NULL, "The output pointer must not be NULL.");
    CUTL_ASSERT(sys != NULL, "The system must not be NULL.");
    CUTL_ASSERT(graph != NULL, "The graph must not be NULL.");
    *out = NULL;

    // The graph describes one particular system; a different one would size the destination wrongly.
    CUTL_ASSERT(graph->n == sys->n, "The graph has %llu blocks but the system has %llu.", (unsigned long long)graph->n,
                (unsigned long long)sys->n);
    CUTL_ASSERT(graph->precision == sys->precision, "The graph describes %s but the system stores %s.",
                graph->precision == HYBSOL_PRECISION_DOUBLE ? "doubles" : "floats",
                sys->precision == HYBSOL_PRECISION_DOUBLE ? "doubles" : "floats");
    CUTL_ASSERT(graph->signature == hybsol_elimination_signature(sys),
                "This graph was computed from a system with different block sizes.");

    const uint64_t n = sys->n;
    const decomp_layout_t layout = decomp_layout(graph);

    unsigned char *const base = (unsigned char *)storage;
    hybsol_decomposition_t *const dec = (hybsol_decomposition_t *)base;
    *dec = (hybsol_decomposition_t){0};

    dec->allocator = sys->allocator;
    dec->n = n;
    dec->n_levels = graph->n_levels;
    dec->n_occupancy = graph->n_occupancy;
    dec->n_operations = graph->n_operations;
    dec->failing_block = UINT64_MAX;
    dec->precision = sys->precision;
    dec->factorized = 0;
    dec->raw = NULL;
    dec->block_offsets = (const uint64_t *)(base + layout.offsets);
    dec->level_offset = (uint64_t *)(base + layout.level_offset);
    dec->level_rows = (uint64_t *)(base + layout.level_rows);
    dec->level_k = (uint64_t *)(base + layout.level_k);
    dec->row_n_elim = (uint64_t *)(base + layout.row_n_elim);
    dec->row_level = (uint64_t *)(base + layout.row_level);
    dec->rows = (hybsol_row_t *)(base + layout.rows);
    dec->entries = (hybsol_row_entry_t **)(base + layout.entries);
    dec->values = base + layout.values;

    memcpy((void *)dec->block_offsets, sys->block_offsets, (size_t)(n + 1) * sizeof(uint64_t));
    memcpy(dec->level_offset, graph->level_offset, (size_t)(graph->n_levels + 1) * sizeof(uint64_t));
    memcpy(dec->level_rows, graph->level_rows, (size_t)dec->n_occupancy * sizeof(uint64_t));
    memcpy(dec->level_k, graph->level_k, (size_t)dec->n_occupancy * sizeof(uint64_t));
    memcpy(dec->row_n_elim, graph->row_n_elim, (size_t)n * sizeof(uint64_t));
    memcpy(dec->row_level, graph->row_level, (size_t)n * sizeof(uint64_t));

    // One carved block per column of the graph's pattern: the system's values where it has them, zeros for fill-in.
    const size_t scalar = hybsol_scalar_size(sys->precision);
    size_t at_values = 0;
    for (uint64_t i = 0; i < n; ++i)
    {
        const uint64_t length = graph->row_offset[i + 1] - graph->row_offset[i];
        const uint64_t at_cols = graph->row_offset[i];

        dec->rows[i].count = length;
        dec->rows[i].capacity = length;
        dec->rows[i].entries = dec->entries + at_cols;

        const uint64_t rows_of = hybsol_block_size(sys, i);
        hybsol_row_entry_t *const *const src_entries = sys->rows[i].entries;
        uint64_t j_src = 0;

        for (uint64_t j = 0; j < length; ++j)
        {
            const uint64_t col = graph->cols[at_cols + j];
            const size_t n_values = (size_t)rows_of * (size_t)hybsol_block_size(sys, col);

            hybsol_row_entry_t *const entry = (hybsol_row_entry_t *)(dec->values + at_values);
            at_values += hybsol_align_up(sizeof(*entry) + n_values * scalar);
            entry->col = col;
            dec->rows[i].entries[j] = entry;

            while (j_src < sys->rows[i].count && src_entries[j_src]->col < col)
                ++j_src;
            if (j_src < sys->rows[i].count && src_entries[j_src]->col == col)
                memcpy(entry->vals, src_entries[j_src]->vals, n_values * scalar);
            else
                memset(entry->vals, 0, n_values * scalar);
        }
    }

    CUTL_ASSERT(at_values == graph->value_bytes, "The destination carved %zu bytes but the graph sized %zu.", at_values,
                graph->value_bytes);

    *out = dec;
    return HYBSOL_SUCCESS;
}

hybsol_result_t hybsol_decomposition_create(const hybsol_system_t *const sys, const hybsol_elimination_t *const graph,
                                            hybsol_decomposition_t **const out)
{
    CUTL_ASSERT(sys != NULL, "The system must not be NULL.");
    CUTL_ASSERT(out != NULL, "The output pointer must not be NULL.");
    CUTL_ASSERT(graph != NULL, "The graph must not be NULL.");

    void *const storage = hybsol_alloc(sys->allocator, hybsol_decomposition_bytes(graph));
    if (storage == NULL)
    {
        *out = NULL;
        return HYBSOL_ERROR_OUT_OF_MEMORY;
    }

    const hybsol_result_t res = hybsol_decomposition_init(sys, graph, storage, out);
    if (res != HYBSOL_SUCCESS)
        hybsol_free(sys->allocator, storage);
    else
        // The decomposition now owns the block it was laid out in.
        (*out)->raw = storage;
    return res;
}

uint64_t hybsol_decomposition_n_blocks(const hybsol_decomposition_t *const dec)
{
    CUTL_ASSERT(dec != NULL, "The decomposition must not be NULL.");
    return dec->n;
}

uint64_t hybsol_decomposition_total_size(const hybsol_decomposition_t *const dec)
{
    CUTL_ASSERT(dec != NULL, "The decomposition must not be NULL.");
    return dec->block_offsets[dec->n];
}

int hybsol_decomposition_is_factorized(const hybsol_decomposition_t *const dec)
{
    CUTL_ASSERT(dec != NULL, "The decomposition must not be NULL.");
    return dec->factorized ? 1 : 0;
}

uint64_t hybsol_decomposition_failing_block(const hybsol_decomposition_t *const dec)
{
    CUTL_ASSERT(dec != NULL, "The decomposition must not be NULL.");
    return dec->failing_block;
}

uint64_t hybsol_decomposition_n_operations(const hybsol_decomposition_t *const dec)
{
    CUTL_ASSERT(dec != NULL, "The decomposition must not be NULL.");
    return dec->n_operations;
}

/* ------------------------------------------------------------------------- */
/* The recorded operations                                                    */
/* ------------------------------------------------------------------------- */

/*
 * The list is not stored: it is a reading of the schedule. Each row contributes one
 * operation per pass it is processed in -- an elimination, except in the pass where it
 * factorizes its diagonal -- walked in pass order, which is the sequence a factorization
 * would have appended.
 */
hybsol_result_t hybsol_decomposition_operations(const hybsol_decomposition_t *const dec, hybsol_operation_t *const out,
                                                const uint64_t capacity, uint64_t *const n_written)
{
    CUTL_ASSERT(dec != NULL, "The decomposition must not be NULL.");
    CUTL_ASSERT(out != NULL, "The output array must not be NULL.");
    CUTL_ASSERT(capacity >= dec->n_operations,
                "The decomposition records %llu operations but only %llu slots were given.",
                (unsigned long long)dec->n_operations, (unsigned long long)capacity);

    uint64_t at = 0;
    for (uint64_t pass = 0; pass < dec->n_levels; ++pass)
    {
        const uint64_t from = dec->level_offset[pass];
        const uint64_t to = dec->level_offset[pass + 1];
        for (uint64_t j = from; j < to; ++j)
        {
            const uint64_t row = dec->level_rows[j];
            const uint64_t step = dec->level_k[j];

            if (step < dec->row_n_elim[row])
            {
                out[at++] = (hybsol_operation_t){
                    .type = HYBSOL_OPERATION_ELIMINATE, .idx_row = row, .idx_col = dec->rows[row].entries[step]->col};
            }
            if (pass == dec->row_level[row])
                out[at++] = (hybsol_operation_t){.type = HYBSOL_OPERATION_INVERT_DIAGONAL, .idx_row = row};
        }
    }

    CUTL_ASSERT(at == dec->n_operations, "Wrote %llu operations but the decomposition records %llu.",
                (unsigned long long)at, (unsigned long long)dec->n_operations);
    if (n_written != NULL)
        *n_written = at;
    return HYBSOL_SUCCESS;
}

/* ------------------------------------------------------------------------- */
/* The factorization                                                          */
/* ------------------------------------------------------------------------- */

/*
 * One pass of the schedule: every row the pass lists does at most one elimination, then factorizes its
 * diagonal if this is its last pass. Each row writes only its own blocks, so a pass needs no lock and
 * no shared result other than the slot each row reports into.
 */
hybsol_result_t hybsol_decomposition_factorize_with_workspace(hybsol_decomposition_t *const dec, void *const workspace,
                                                              const size_t workspace_bytes, const uint64_t n_threads)
{
    CUTL_ASSERT(dec != NULL, "The decomposition must not be NULL.");
    CUTL_ASSERT(workspace != NULL, "The workspace pointer must not be NULL.");
    CUTL_ASSERT(!dec->factorized, "This decomposition has already been factorized.");

    const uint64_t threads = (uint64_t)hybsol_resolve_threads(n_threads);

    workspace_bind((hybsol_workspace_t *)workspace, workspace_bytes, dec, threads);
    const hybsol_workspace_t *const ws = (const hybsol_workspace_t *)workspace;
    const unsigned char *const base = (const unsigned char *)workspace;
    hybsol_result_t *const results = (hybsol_result_t *)(base + ws->off_results);

    for (uint64_t pass = 0; pass < dec->n_levels; ++pass)
    {
        const uint64_t from = dec->level_offset[pass];
        const uint64_t to = dec->level_offset[pass + 1];

#pragma omp parallel for schedule(dynamic) default(none) shared(dec, ws, results, from, to, pass) if (threads > 1)     \
    num_threads(threads)
        for (uint64_t j = from; j < to; ++j)
        {
            const uint64_t row = dec->level_rows[j];
            const uint64_t step = dec->level_k[j];

            // The last elimination and the diagonal factorization share a step: the row is done once
            // its own diagonal comes first.
            hybsol_result_t res = HYBSOL_SUCCESS;
            if (step < dec->row_n_elim[row])
                hybsol_decomposition_eliminate(dec, row, step, workspace_scratch(ws));
            if (pass == dec->row_level[row])
            {
                res = hybsol_decomposition_diagonal_lu(dec, row);
                if (res == HYBSOL_SUCCESS)
                    hybsol_decomposition_diagonal_inverse(dec, row);
            }
            results[row] = res;
        }

        // Settled outside the region: one thread names the offending block, and the rows are final
        // before the next pass reads them.
        for (uint64_t j = from; j < to; ++j)
        {
            if (results[dec->level_rows[j]] == HYBSOL_SUCCESS)
                continue;
            dec->failing_block = dec->level_rows[j];
            return results[dec->level_rows[j]];
        }
    }

    dec->factorized = 1;
    return HYBSOL_SUCCESS;
}

hybsol_result_t hybsol_decomposition_factorize(hybsol_decomposition_t *const dec, const uint64_t n_threads)
{
    CUTL_ASSERT(dec != NULL, "The decomposition must not be NULL.");

    const uint64_t threads = (uint64_t)hybsol_resolve_threads(n_threads);
    const size_t bytes = workspace_size(dec->n, dec->block_offsets, hybsol_scalar_size(dec->precision), threads);

    void *const workspace = hybsol_alloc(dec->allocator, bytes);
    if (workspace == NULL)
        return HYBSOL_ERROR_OUT_OF_MEMORY;

    const hybsol_result_t res = hybsol_decomposition_factorize_with_workspace(dec, workspace, bytes, n_threads);
    hybsol_free(dec->allocator, workspace);
    return res;
}

/* ------------------------------------------------------------------------- */
/* Replay helpers                                                             */
/* ------------------------------------------------------------------------- */

void hybsol_decomposition_apply_operations(const hybsol_decomposition_t *const dec, const uint64_t n_ops,
                                           const hybsol_operation_t *const ops, double *const vec)
{
    CUTL_ASSERT(dec != NULL, "The decomposition must not be NULL.");
    CUTL_ASSERT(vec != NULL, "The solution vector must not be NULL.");
    hybsol_decomposition_replay(dec, n_ops, ops, vec);
}

void hybsol_decomposition_solve_upper(const hybsol_decomposition_t *const dec, double *const vec)
{
    CUTL_ASSERT(dec != NULL, "The decomposition must not be NULL.");
    hybsol_decomposition_back_substitute(dec, vec);
}
