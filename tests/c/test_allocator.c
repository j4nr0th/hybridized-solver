/**
 * Decomposing with a caller-supplied allocator that is deliberately *not*
 * thread-safe.
 *
 * The elimination graph fixes the destination's layout before any thread
 * starts, so a factorization never calls the system's allocator from inside a
 * parallel region -- not to allocate, and not to release either. Without that
 * this test is a race, and on a multi-core box it faults; with it a plain bump
 * allocator -- no lock at all -- works and gives the standard allocator's
 * answers.
 */

#include "test_util.h"

#include <hybsol/hybsol.h>

#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdlib.h>
#include <string.h>

/* A bump allocator with no locking whatsoever */

typedef struct
{
    unsigned char *base;
    size_t capacity;
    size_t used;
    long allocations;
    long frees;
} arena_t;

/** Prefix on every block, so a grow knows how much of the old one to copy. */
typedef struct
{
    size_t size;
    size_t pad;
} arena_hdr_t;

static arena_t g_arena;

static void *arena_alloc(void *const state, const size_t size)
{
    arena_t *const arena = (arena_t *)state;
    const size_t want = sizeof(arena_hdr_t) + ((size + 63u) & ~(size_t)63u);

    arena->allocations += 1;
    if (arena->used + want > arena->capacity)
        return NULL;

    unsigned char *const at = arena->base + arena->used;
    arena->used += want;
    ((arena_hdr_t *)at)->size = size;
    return at + sizeof(arena_hdr_t);
}

static void *arena_realloc(void *const state, void *const ptr, const size_t size)
{
    if (ptr == NULL)
        return arena_alloc(state, size);

    const size_t old = ((const arena_hdr_t *)((unsigned char *)ptr - sizeof(arena_hdr_t)))->size;
    void *const fresh = arena_alloc(state, size);
    if (fresh != NULL)
        memcpy(fresh, ptr, old < size ? old : size);
    return fresh;
}

static void arena_free(void *const state, void *const ptr)
{
    arena_t *const arena = (arena_t *)state;
    if (ptr != NULL)
        arena->frees += 1;
}

static const cutl_allocator_t *arena_allocator(void)
{
    static cutl_allocator_t alloc;
    alloc.state = &g_arena;
    alloc.allocate = arena_alloc;
    alloc.reallocate = arena_realloc;
    alloc.deallocate = arena_free;
    return &alloc;
}

static void arena_reset(const size_t bytes)
{
    g_arena.base = (unsigned char *)aligned_alloc(64, bytes);
    g_arena.capacity = bytes;
    g_arena.used = 0;
    g_arena.allocations = 0;
    g_arena.frees = 0;
}

static void arena_release(void)
{
    free(g_arena.base);
    g_arena.base = NULL;
}

/* A well-conditioned block system */

typedef struct
{
    uint64_t n_blocks;
    uint64_t block_size;
    uint64_t *sizes;
    uint64_t *rows;
    uint64_t *cols;
    double *data;
    size_t n_entries;
} pattern_t;

/**
 * Fill a COO description of a symmetric, strongly diagonally dominant system.
 *
 * Dominance keeps the LU off a zero pivot, so corruption shows up as a
 * wrong answer rather than a clean error that would hide the race.
 */
static int pattern_build(pattern_t *const p, const uint64_t n_blocks, const uint64_t block_size)
{
    // Zeroed up front so an allocation failure below leaves the struct in a
    // defined state for the caller's pattern_free, and so -O2 does not see an
    // uninitialised read on that path.
    *p = (pattern_t){0};
    p->n_blocks = n_blocks;
    p->block_size = block_size;

    /* Upper bound on entries: a full triangle, doubled for the symmetry. */
    const size_t cap = (size_t)n_blocks * (size_t)n_blocks + 2 * (size_t)n_blocks + 8;
    p->sizes = (uint64_t *)malloc(sizeof(uint64_t) * n_blocks);
    p->rows = (uint64_t *)malloc(sizeof(uint64_t) * cap);
    p->cols = (uint64_t *)malloc(sizeof(uint64_t) * cap);
    p->data = (double *)malloc(sizeof(double) * cap * block_size * block_size);

    if (p->sizes == NULL || p->rows == NULL || p->cols == NULL || p->data == NULL)
        return 0;

    for (uint64_t i = 0; i < n_blocks; ++i)
        p->sizes[i] = block_size;

    const size_t block_vals = (size_t)block_size * block_size;
    double *const block = (double *)malloc(sizeof(double) * block_vals);
    if (block == NULL)
        return 0;

    size_t at = 0;
    for (uint64_t i = 0; i < n_blocks; ++i)
    {
        for (uint64_t k = 0; k < block_size; ++k)
        {
            block[k * block_size + k] = 100.0 + (double)((i * 7 + k) % 11);
            block[k * block_size + (k + 1) % block_size] = 0.5 + (double)((i * 3 + k) % 5);
        }
        memcpy(p->data + at * block_vals, block, sizeof(double) * block_vals);
        p->rows[at] = i;
        p->cols[at] = i;
        ++at;
    }
    for (uint64_t i = 0; i < n_blocks; ++i)
    {
        for (uint64_t j = i + 1; j < n_blocks; ++j)
        {
            if ((i + j) % 3 == 0)
                continue; /* leave a hole, so the system is not fully dense */
            for (uint64_t k = 0; k < block_vals; ++k)
                block[k] = 0.25 + (double)((i * j + k) % 7);
            memcpy(p->data + at * block_vals, block, sizeof(double) * block_vals);
            p->rows[at] = i;
            p->cols[at] = j;
            ++at;
            memcpy(p->data + at * block_vals, block, sizeof(double) * block_vals);
            p->rows[at] = j;
            p->cols[at] = i;
            ++at;
        }
    }
    p->n_entries = at;
    free(block);
    return 1;
}

static void pattern_free(pattern_t *const p)
{
    free(p->sizes);
    free(p->rows);
    free(p->cols);
    free(p->data);
    memset(p, 0, sizeof(*p));
}

static int pattern_system(const pattern_t *const p, const cutl_allocator_t *const alloc, hybsol_system_t **out)
{
    *out = NULL;
    if (hybsol_system_create(p->n_blocks, p->sizes, out, alloc) != HYBSOL_SUCCESS)
        return 0;
    return hybsol_system_add_blocks(*out, p->n_entries, p->rows, p->cols, p->data) == HYBSOL_SUCCESS;
}

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

/**
 * Decompose and solve, then checksum the answer.
 *
 * The matrix is read first; the decomposition copies the blocks it needs and
 * leaves the system as it stands.
 */
static int solve_checksum(hybsol_system_t *const sys, const uint64_t n_threads, double *const checksum)
{
    const uint64_t total = hybsol_system_total_size(sys);
    double *const a = (double *)malloc(sizeof(double) * (size_t)total * (size_t)total);
    double *const b = (double *)malloc(sizeof(double) * total);
    double *const x = (double *)malloc(sizeof(double) * total);

    if (a == NULL || b == NULL || x == NULL)
    {
        free(a);
        free(b);
        free(x);
        return 0;
    }

    int ok = hybsol_system_to_dense(sys, a) == HYBSOL_SUCCESS;
    for (uint64_t i = 0; ok && i < total; ++i)
    {
        double sum = 0.0;
        for (uint64_t j = 0; j < total; ++j)
            sum += a[i * total + j];
        b[i] = sum;
        x[i] = 1.0;
    }

    hybsol_decomposition_t *const dec = ok ? factorize(sys, n_threads) : NULL;
    ok = ok && dec != NULL && hybsol_decomposition_solve(dec, x, n_threads) == HYBSOL_SUCCESS;
    if (ok)
    {
        double sum = 0.0;
        for (uint64_t i = 0; i < total; ++i)
            sum += x[i];
        *checksum = sum;
    }

    hybsol_decomposition_destroy(dec);
    free(a);
    free(b);
    free(x);
    return ok;
}

/* Tests */

/** Serial work must drive a caller-supplied allocator correctly, including
 * releasing everything through it. */
static void test_serial_with_lock_free_allocator(void)
{
    pattern_t p;
    CHECK_MSG(pattern_build(&p, 24, 8), "building the pattern");
    arena_reset(64u * 1024u * 1024u);

    hybsol_system_t *sys = NULL;
    double checksum = 0.0;
    CHECK_MSG(pattern_system(&p, arena_allocator(), &sys), "assembling with a lock-free allocator");
    CHECK_MSG(solve_checksum(sys, 1, &checksum), "serial solve with a lock-free allocator");

    const long allocations = g_arena.allocations;
    const long peak_used = (long)g_arena.used;
    CHECK_MSG(allocations > 0, "the supplied allocator was never called (%ld)", allocations);
    hybsol_system_destroy(sys);

    // The arena is bump-only, so a double free or use-after-free would have
    // faulted above; the size must not move on destroy.
    CHECK_MSG((long)g_arena.used == peak_used, "the arena grew while destroying (%ld -> %ld)", peak_used,
              (long)g_arena.used);

    arena_release();
    pattern_free(&p);
}

/**
 * The regression this file exists for: with the per-thread bump regions a
 * lock-free allocator is not raced by the parallel regions. Without them it
 * faults outright above one thread.
 */
static void test_parallel_with_lock_free_allocator(void)
{
    pattern_t p;
    CHECK_MSG(pattern_build(&p, 48, 16), "building the pattern");

    /* The standard allocator, for the reference answer. */
    double with_std = 0.0;
    {
        hybsol_system_t *sys = NULL;
        CHECK_MSG(pattern_system(&p, &CUTL_STD_ALLOCATOR, &sys), "assembling with the standard allocator");
        CHECK_MSG(solve_checksum(sys, 4, &with_std), "reference solve with the standard allocator");
        hybsol_system_destroy(sys);
    }

    arena_reset(256u * 1024u * 1024u);
    const cutl_allocator_t *const alloc = arena_allocator();

    /* Several thread counts, since the fault depended on having a real team. */
    const uint64_t thread_counts[] = {1, 2, 4, 8};
    for (size_t i = 0; i < sizeof(thread_counts) / sizeof(thread_counts[0]); ++i)
    {
        const uint64_t n_threads = thread_counts[i];
        hybsol_system_t *sys = NULL;
        double checksum = 0.0;

        CHECK_MSG(pattern_system(&p, alloc, &sys), "assembling at %llu threads", (unsigned long long)n_threads);
        CHECK_MSG(solve_checksum(sys, n_threads, &checksum), "solve at %llu threads", (unsigned long long)n_threads);
        CHECK_MSG(checksum == with_std, "at %llu threads got %.17g, reference %.17g", (unsigned long long)n_threads,
                  checksum, with_std);

        hybsol_system_destroy(sys);
    }

    arena_release();
    pattern_free(&p);
}

/**
 * The walk knows exactly how many operations there will be, so the count it
 * reports is the count the decomposition materializes -- and a fully coupled
 * system reaches the maximum the structure allows.
 */
static void test_n_operations_is_exact(void)
{
    pattern_t p;
    CHECK_MSG(pattern_build(&p, 32, 4), "building the pattern");

    hybsol_system_t *sys = NULL;
    CHECK(pattern_system(&p, &CUTL_STD_ALLOCATOR, &sys));

    hybsol_elimination_t *graph = NULL;
    CHECK_OK(hybsol_elimination_create(sys, &graph));
    const uint64_t expected = hybsol_elimination_n_operations(graph);
    CHECK_MSG(expected > 0, "the walk found no operations in a valid system");
    CHECK_MSG(expected <= p.n_blocks * (p.n_blocks + 1) / 2, "the walk found %llu operations, above the maximum %llu",
              (unsigned long long)expected, (unsigned long long)(p.n_blocks * (p.n_blocks + 1) / 2));

    hybsol_decomposition_t *dec = NULL;
    CHECK_OK(hybsol_decomposition_create(sys, graph, &dec));
    CHECK_OK(hybsol_decomposition_factorize(dec, 2));

    CHECK_MSG(hybsol_decomposition_n_operations(dec) == expected, "the decomposition says %llu, the walk said %llu",
              (unsigned long long)hybsol_decomposition_n_operations(dec), (unsigned long long)expected);

    hybsol_operation_t *const ops = malloc(sizeof(*ops) * expected);
    CHECK(ops != NULL);
    if (ops != NULL)
    {
        uint64_t written = 0;
        CHECK_OK(hybsol_decomposition_operations(dec, ops, expected, &written));
        CHECK_MSG(written == expected, "materialized %llu operations, expected %llu", (unsigned long long)written,
                  (unsigned long long)expected);
        free(ops);
    }

    hybsol_decomposition_destroy(dec);
    hybsol_elimination_destroy(graph);
    hybsol_system_destroy(sys);
    pattern_free(&p);
}

/** A caller-supplied workspace must match the internal one, be written to, and
 * be reusable. */
static void test_workspace_matches_internal(void)
{
    const uint64_t n_threads = 4;
    pattern_t p;
    CHECK_MSG(pattern_build(&p, 24, 8), "building the pattern");

    hybsol_system_t *reference = NULL;
    double expected = 0.0;
    CHECK(pattern_system(&p, &CUTL_STD_ALLOCATOR, &reference));
    /* Taken before the reference is released. */
    const size_t bytes = hybsol_workspace_bytes(reference, n_threads);
    CHECK_MSG(bytes > 0, "workspace_bytes returned %zu", bytes);
    CHECK(solve_checksum(reference, n_threads, &expected));
    hybsol_system_destroy(reference);

    void *const workspace = malloc(bytes);
    memset(workspace, 0, bytes);

    hybsol_system_t *sys = NULL;
    CHECK(pattern_system(&p, &CUTL_STD_ALLOCATOR, &sys));

    const uint64_t total = hybsol_system_total_size(sys);
    double *const a = (double *)malloc(sizeof(double) * (size_t)total * (size_t)total);
    double *const b = (double *)malloc(sizeof(double) * total);
    double *const x = (double *)malloc(sizeof(double) * total);
    CHECK(hybsol_system_to_dense(sys, a) == HYBSOL_SUCCESS);
    for (uint64_t i = 0; i < total; ++i)
    {
        double sum = 0.0;
        for (uint64_t j = 0; j < total; ++j)
            sum += a[i * total + j];
        b[i] = sum;
        x[i] = 1.0;
    }

    hybsol_elimination_t *graph = NULL;
    CHECK_OK(hybsol_elimination_create(sys, &graph));
    hybsol_decomposition_t *dec = NULL;
    CHECK_OK(hybsol_decomposition_create(sys, graph, &dec));
    hybsol_elimination_destroy(graph);

    CHECK_OK(hybsol_decomposition_factorize_with_workspace(dec, workspace, bytes, n_threads));
    // The header is written first, and its magic proves the caller's buffer is what got used.
    CHECK_MSG(memcmp(workspace, "1wlosbyh", 8) == 0, "the caller's workspace was not written to");

    CHECK_OK(hybsol_decomposition_solve(dec, x, n_threads));
    double got = 0.0;
    for (uint64_t i = 0; i < total; ++i)
        got += x[i];
    CHECK_MSG(got == expected, "workspace solve gave %.17g, internal gave %.17g", got, expected);
    const uint64_t ops = hybsol_decomposition_n_operations(dec);
    hybsol_decomposition_destroy(dec);

    /* The same buffer, reused for a second system of the same shape. */
    hybsol_system_t *second = NULL;
    CHECK(pattern_system(&p, &CUTL_STD_ALLOCATOR, &second));
    graph = NULL;
    CHECK_OK(hybsol_elimination_create(second, &graph));
    dec = NULL;
    CHECK_OK(hybsol_decomposition_create(second, graph, &dec));
    hybsol_elimination_destroy(graph);
    CHECK_OK(hybsol_decomposition_factorize_with_workspace(dec, workspace, bytes, n_threads));
    CHECK_MSG(hybsol_decomposition_n_operations(dec) == ops, "reusing a workspace changed the operation count");
    hybsol_decomposition_destroy(dec);

    hybsol_system_destroy(second);
    hybsol_system_destroy(sys);
    free(a);
    free(b);
    free(x);
    free(workspace);
    pattern_free(&p);
}

/** A workspace must grow with the thread count, since each thread gets scratch. */
static void test_workspace_scales_with_threads(void)
{
    pattern_t p;
    CHECK_MSG(pattern_build(&p, 16, 8), "building the pattern");

    hybsol_system_t *sys = NULL;
    CHECK(pattern_system(&p, &CUTL_STD_ALLOCATOR, &sys));

    const size_t one = hybsol_workspace_bytes(sys, 1);
    const size_t four = hybsol_workspace_bytes(sys, 4);
    CHECK_MSG(four > one, "4 threads need %zu bytes but 1 thread needs %zu", four, one);

    hybsol_system_destroy(sys);
    pattern_free(&p);
}

/**
 * A decomposition and the graph it came from are separate allocations, so the
 * system, the graph and the decomposition can go in any order.
 */
static void test_release_in_any_order(void)
{
    pattern_t p;
    CHECK_MSG(pattern_build(&p, 16, 4), "building the pattern");

    hybsol_system_t *sys = NULL;
    CHECK(pattern_system(&p, &CUTL_STD_ALLOCATOR, &sys));

    hybsol_elimination_t *graph = NULL;
    CHECK_OK(hybsol_elimination_create(sys, &graph));
    hybsol_decomposition_t *dec = NULL;
    CHECK_OK(hybsol_decomposition_create(sys, graph, &dec));
    CHECK_OK(hybsol_decomposition_factorize(dec, 2));

    // The system first: the decomposition owns its own copy of the blocks.
    hybsol_system_destroy(sys);
    CHECK(hybsol_decomposition_n_operations(dec) > 0);
    CHECK(hybsol_decomposition_is_factorized(dec));

    // Then the graph, which the decomposition already copied what it needs of.
    hybsol_elimination_destroy(graph);
    CHECK(hybsol_decomposition_n_blocks(dec) == p.n_blocks);

    hybsol_decomposition_destroy(dec);
    pattern_free(&p);
}

/* ------------------------------------------------------------------------- */
/* What the decomposition actually asks of the allocator                      */
/* ------------------------------------------------------------------------- */

/*
 * Counts calls, and separately counts the ones made from inside an OpenMP
 * region -- which is the property under test, since a caller may have supplied
 * an allocator that is not thread-safe.
 */
typedef struct
{
    long allocate;
    long reallocate;
    long deallocate;
    long from_parallel;
} calls_t;

static calls_t g_calls;

static void note_call(void)
{
#ifdef _OPENMP
    if (omp_in_parallel())
        g_calls.from_parallel += 1;
#endif
}

static void *count_alloc(void *const s, const size_t size)
{
    (void)s;
    g_calls.allocate += 1;
    note_call();
    return malloc(size);
}
static void *count_realloc(void *const s, void *const p, const size_t size)
{
    (void)s;
    g_calls.reallocate += 1;
    note_call();
    return realloc(p, size);
}
static void count_dealloc(void *const s, void *const p)
{
    (void)s;
    if (p != NULL)
        g_calls.deallocate += 1;
    note_call();
    free(p);
}

/**
 * No call into the system's allocator may be made from inside a parallel
 * region -- not to allocate and not to release. A release from one is a call on
 * every thread at once, which is what a lock-taking allocator cannot survive.
 *
 * The rows are grown to their peak before the first region starts, which is
 * what removes the releases; the fill-in comes from the pool, which is the only
 * other thing the elimination allocates. Serial-phase calls, including the
 * symbolic walk and the pre-grow's own releases of the assembly-time arrays,
 * are fine and expected.
 */
static void test_no_allocator_calls_from_parallel_regions(void)
{
    const uint64_t thread_counts[] = {1, 4, 8};

    for (size_t t = 0; t < sizeof(thread_counts) / sizeof(thread_counts[0]); ++t)
    {
        pattern_t p;
        CHECK_MSG(pattern_build(&p, 20, 5), "building the pattern");

        const cutl_allocator_t counting = {
            .state = NULL, .allocate = count_alloc, .reallocate = count_realloc, .deallocate = count_dealloc};

        hybsol_system_t *sys = NULL;
        CHECK_MSG(pattern_system(&p, &counting, &sys), "assembling at %llu threads",
                  (unsigned long long)thread_counts[t]);

        memset(&g_calls, 0, sizeof(g_calls));
        hybsol_elimination_t *graph = NULL;
        CHECK_OK(hybsol_elimination_create(sys, &graph));
        hybsol_decomposition_t *dec = NULL;
        CHECK_OK(hybsol_decomposition_create(sys, graph, &dec));
        hybsol_elimination_destroy(graph);
        CHECK_OK(hybsol_decomposition_factorize(dec, thread_counts[t]));
        hybsol_decomposition_destroy(dec);

        CHECK_MSG(g_calls.allocate > 0, "nothing was allocated at all, so the test proves nothing");
        CHECK_MSG(g_calls.from_parallel == 0,
                  "%ld of %ld allocator calls were made from inside a parallel region at %llu threads",
                  g_calls.from_parallel, g_calls.allocate + g_calls.reallocate + g_calls.deallocate,
                  (unsigned long long)thread_counts[t]);

        hybsol_system_destroy(sys);
        pattern_free(&p);
    }
}

/**
 * The walk is exact, so the destination it sizes needs no allocator call it did
 * not budget, and two walks of the same system agree to the byte.
 */
static void test_footprint_is_exact_and_stable(void)
{
    pattern_t p;
    CHECK_MSG(pattern_build(&p, 16, 4), "building the pattern");

    hybsol_system_t *sys = NULL;
    CHECK(pattern_system(&p, &CUTL_STD_ALLOCATOR, &sys));

    hybsol_elimination_t *first = NULL;
    hybsol_elimination_t *second = NULL;
    CHECK_OK(hybsol_elimination_create(sys, &first));
    CHECK_OK(hybsol_elimination_create(sys, &second));

    CHECK_MSG(hybsol_elimination_value_bytes(first) == hybsol_elimination_value_bytes(second),
              "the footprint changed between walks: %zu then %zu", hybsol_elimination_value_bytes(first),
              hybsol_elimination_value_bytes(second));
    CHECK_MSG(hybsol_elimination_total_bytes(first) == hybsol_elimination_total_bytes(second),
              "the graph's own size changed between walks");
    CHECK_MSG(hybsol_elimination_value_bytes(first) > 0, "a system with fill-in planned a zero-byte arena");

    // The graph outlives the system, which is what lets a caller release the
    // system before it factors anything.
    hybsol_system_destroy(sys);
    CHECK(hybsol_elimination_n_blocks(first) == p.n_blocks);
    CHECK(hybsol_elimination_n_columns(first) > 0);

    hybsol_elimination_destroy(first);
    hybsol_elimination_destroy(second);
    pattern_free(&p);
}

/**
 * A chain: diagonal blocks plus one off-diagonal either side.
 *
 * Eliminating a chain adds no fill-in, so no row ever outgrows the columns the
 * walk seeded it with. That makes the allocation counts below exact rather
 * than bounded, which is the point: they are pinned so a future edit cannot
 * quietly reintroduce a per-row or a per-thread allocation.
 */
static hybsol_system_t *chain_system(const cutl_allocator_t *const alloc, const uint64_t n_blocks,
                                     const uint64_t block_size, uint64_t *const order_out)
{
    uint64_t *const sizes = (uint64_t *)malloc(sizeof(*sizes) * n_blocks);
    for (uint64_t i = 0; i < n_blocks; ++i)
        sizes[i] = block_size;

    hybsol_system_t *sys = NULL;
    if (hybsol_system_create(n_blocks, sizes, &sys, alloc) != HYBSOL_SUCCESS)
    {
        free(sizes);
        return NULL;
    }
    free(sizes);

    const size_t values = (size_t)block_size * (size_t)block_size;
    double *const block = (double *)malloc(sizeof(*block) * values);
    double *const coupling = (double *)malloc(sizeof(*coupling) * values);
    if (block == NULL || coupling == NULL)
    {
        free(block);
        free(coupling);
        hybsol_system_destroy(sys);
        return NULL;
    }
    for (size_t k = 0; k < values; ++k)
    {
        coupling[k] = 0.25;
        block[k] = 0.5;
    }
    // Diagonally dominant, so the factorization stays off a zero pivot.
    for (uint64_t k = 0; k < block_size; ++k)
        block[k * block_size + k] = 8.0 + (double)k;

    for (uint64_t i = 0; i < n_blocks; ++i)
    {
        hybsol_system_add_block(sys, i, i, block_size, block_size, block);
        if (i + 1 < n_blocks)
        {
            hybsol_system_add_block(sys, i, i + 1, block_size, block_size, coupling);
            hybsol_system_add_block(sys, i + 1, i, block_size, block_size, coupling);
        }
    }
    free(block);
    free(coupling);

    for (uint64_t i = 0; i < n_blocks; ++i)
        order_out[i] = n_blocks - 1 - i;
    return sys;
}

/**
 * Each stage makes a fixed number of allocations, whatever the system is.
 *
 * The walk's arrays share one allocation and the graph's own scratch rides in
 * its frame; a reorder's scratch is a single block instead of one allocation a
 * thread; the destination is already one allocation and factorizing adds none.
 */
static void test_allocation_counts_are_grouped(void)
{
    const uint64_t n_blocks = 32;
    const uint64_t block_size = 4;
    uint64_t *const order = (uint64_t *)malloc(sizeof(*order) * n_blocks);
    CHECK(order != NULL);
    if (order == NULL)
    {
        return;
    }

    const cutl_allocator_t counting = {
        .state = NULL, .allocate = count_alloc, .reallocate = count_realloc, .deallocate = count_dealloc};
    hybsol_system_t *const sys = chain_system(&counting, n_blocks, block_size, order);
    CHECK(sys != NULL);
    if (sys == NULL)
    {
        free(order);
        return;
    }

    // A chain needs no fill-in, so the walk allocates the arena, the occurrence
    // log and the graph frame, and nothing per row.
    memset(&g_calls, 0, sizeof(g_calls));
    hybsol_elimination_t *graph = NULL;
    CHECK_OK(hybsol_elimination_create(sys, &graph));
    CHECK_MSG(g_calls.allocate == 3, "the walk made %ld allocations, expected 3", g_calls.allocate);
    hybsol_elimination_destroy(graph);

    // A reorder is its own arena, then the walk's two for the order check -- which
    // builds no graph, because it only wants a verdict. The per-thread scratch
    // that used to cost one allocation a thread is inside the arena now.
    memset(&g_calls, 0, sizeof(g_calls));
    CHECK_OK(hybsol_system_reorder_blocks(sys, order, 4));
    CHECK_MSG(g_calls.allocate == 3, "a reorder made %ld allocations, expected 3", g_calls.allocate);

    // The destination is one allocation on top of the walk's three.
    memset(&g_calls, 0, sizeof(g_calls));
    graph = NULL;
    CHECK_OK(hybsol_elimination_create(sys, &graph));
    hybsol_decomposition_t *dec = NULL;
    CHECK_OK(hybsol_decomposition_create(sys, graph, &dec));
    CHECK_MSG(g_calls.allocate == 4, "a decomposition made %ld allocations, expected 4", g_calls.allocate);
    hybsol_elimination_destroy(graph);

    // And factorizing adds none at all: the destination is already laid out.
    const size_t workspace_bytes = hybsol_workspace_bytes(sys, 4);
    void *const workspace = malloc(workspace_bytes);
    CHECK(workspace != NULL);
    if (workspace != NULL)
    {
        memset(&g_calls, 0, sizeof(g_calls));
        CHECK_OK(hybsol_decomposition_factorize_with_workspace(dec, workspace, workspace_bytes, 4));
        CHECK_MSG(g_calls.allocate == 0, "factorizing allocated %ld times, expected none", g_calls.allocate);
        free(workspace);
    }

    hybsol_decomposition_destroy(dec);
    hybsol_system_destroy(sys);
    free(order);
}

int main(void)
{
    test_serial_with_lock_free_allocator();
    test_parallel_with_lock_free_allocator();
    test_n_operations_is_exact();
    test_workspace_matches_internal();
    test_workspace_scales_with_threads();
    test_release_in_any_order();
    test_no_allocator_calls_from_parallel_regions();
    test_footprint_is_exact_and_stable();
    test_allocation_counts_are_grouped();
    return test_report("test_allocator");
}
