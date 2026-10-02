/**
 * Decomposing with a caller-supplied allocator that is deliberately *not*
 * thread-safe.
 *
 * A decomposition allocates its fill-in from a pool the library sizes exactly
 * up front, so it never calls the system's allocator from inside a parallel
 * region -- not to allocate, and not to release either. Without that this test
 * is a race, and on a multi-core box it faults; with it a plain bump allocator
 * -- no lock at all -- works and gives the standard allocator's answers.
 */

#include "test_util.h"

#include <hybsol/decompose.h>
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

/**
 * Decompose and solve, then checksum the answer.
 *
 * The matrix is read first: decompose overwrites the blocks in place with the
 * LU factors.
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

    ok = ok && hybsol_system_decompose(sys, n_threads) == HYBSOL_SUCCESS &&
         hybsol_system_solve(sys, x) == HYBSOL_SUCCESS;
    if (ok)
    {
        double sum = 0.0;
        for (uint64_t i = 0; i < total; ++i)
            sum += x[i];
        *checksum = sum;
    }

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
        CHECK_MSG(hybsol_system_is_decomposed(sys) != 0, "decomposed at %llu threads", (unsigned long long)n_threads);
        CHECK_MSG(checksum == with_std, "at %llu threads got %.17g, reference %.17g", (unsigned long long)n_threads,
                  checksum, with_std);

        hybsol_system_destroy(sys);
    }

    arena_release();
    pattern_free(&p);
}

/** The operation list is sized to a proven bound, so a decomposition must stay
 * inside it rather than grow. */
static void test_operation_bound_holds(void)
{
    pattern_t p;
    CHECK_MSG(pattern_build(&p, 32, 4), "building the pattern");

    hybsol_system_t *sys = NULL;
    double checksum = 0.0;
    CHECK(pattern_system(&p, &CUTL_STD_ALLOCATOR, &sys));
    CHECK(solve_checksum(sys, 2, &checksum));

    const uint64_t n = p.n_blocks;
    const uint64_t bound = hybsol_system_operation_bound(sys);
    const uint64_t recorded = hybsol_system_n_operations(sys);

    CHECK_MSG(bound == n * (n + 1) / 2, "bound should be n(n+1)/2 = %llu, got %llu",
              (unsigned long long)(n * (n + 1) / 2), (unsigned long long)bound);
    CHECK_MSG(recorded <= bound, "recorded %llu operations but the bound is %llu", (unsigned long long)recorded,
              (unsigned long long)bound);
    CHECK_MSG(recorded > 0, "a valid decomposition recorded no operations");

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

    CHECK_OK(hybsol_system_decompose_with_workspace(sys, workspace, bytes, n_threads));
    // The header is written first, and its magic proves the caller's buffer is what got used.
    CHECK_MSG(memcmp(workspace, "1wlosbyh", 8) == 0, "the caller's workspace was not written to");

    CHECK_OK(hybsol_system_solve(sys, x));
    double got = 0.0;
    for (uint64_t i = 0; i < total; ++i)
        got += x[i];
    CHECK_MSG(got == expected, "workspace solve gave %.17g, internal gave %.17g", got, expected);
    const uint64_t ops = hybsol_system_n_operations(sys);

    /* The same buffer, reused for a second system of the same shape. */
    hybsol_system_t *second = NULL;
    CHECK(pattern_system(&p, &CUTL_STD_ALLOCATOR, &second));
    CHECK_OK(hybsol_system_decompose_with_workspace(second, workspace, bytes, n_threads));
    CHECK_MSG(hybsol_system_n_operations(second) == ops, "reusing a workspace changed the operation count");

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

/** Copying and destroying a decomposed system must not trip over pooled memory. */
static void test_copy_after_decompose(void)
{
    pattern_t p;
    CHECK_MSG(pattern_build(&p, 16, 4), "building the pattern");

    hybsol_system_t *sys = NULL;
    double checksum = 0.0;
    CHECK(pattern_system(&p, &CUTL_STD_ALLOCATOR, &sys));
    CHECK(solve_checksum(sys, 2, &checksum));

    hybsol_system_t *copy = NULL;
    CHECK_OK(hybsol_system_copy(sys, &copy));
    CHECK_MSG(hybsol_system_n_operations(copy) == hybsol_system_n_operations(sys), "copy lost operations");
    CHECK_MSG(hybsol_system_is_decomposed(copy) != 0, "copy is not marked decomposed");

    /* Both must be releasable, in either order. */
    hybsol_system_destroy(copy);
    hybsol_system_destroy(sys);
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
        CHECK_OK(hybsol_system_decompose(sys, thread_counts[t]));

        CHECK_MSG(g_calls.allocate > 0, "nothing was allocated at all, so the test proves nothing");
        CHECK_MSG(g_calls.from_parallel == 0,
                  "%ld of %ld allocator calls were made from inside a parallel region at %llu threads",
                  g_calls.from_parallel, g_calls.allocate + g_calls.reallocate + g_calls.deallocate,
                  (unsigned long long)thread_counts[t]);

        hybsol_system_destroy(sys);
        pattern_free(&p);
    }
}

/** The plan is exact, so a decomposition needs no allocator call it did not budget. */
static void test_fill_plan_is_exact_and_stable(void)
{
    pattern_t p;
    CHECK_MSG(pattern_build(&p, 16, 4), "building the pattern");

    hybsol_system_t *sys = NULL;
    CHECK(pattern_system(&p, &CUTL_STD_ALLOCATOR, &sys));

    hybsol_fill_plan_t first, second;
    CHECK_OK(hybsol_fill_plan(sys, &first));
    CHECK_OK(hybsol_fill_plan(sys, &second));
    CHECK_MSG(first.pool_bytes == second.pool_bytes, "the plan changed between calls: %zu then %zu", first.pool_bytes,
              second.pool_bytes);
    CHECK_MSG(first.pool_bytes > 0, "a system with fill-in planned a zero-byte pool");
    CHECK_MSG(first.operations == p.n_blocks * (p.n_blocks + 1) / 2, "operation bound was %llu, expected %llu",
              (unsigned long long)first.operations, (unsigned long long)(p.n_blocks * (p.n_blocks + 1) / 2));

    // An already decomposed system has no meaningful plan left to give.
    CHECK_OK(hybsol_system_decompose(sys, 2));
    CHECK_RESULT(hybsol_fill_plan(sys, &first), HYBSOL_ERROR_ALREADY_DECOMPOSED);

    hybsol_system_destroy(sys);
    pattern_free(&p);
}

int main(void)
{
    test_serial_with_lock_free_allocator();
    test_parallel_with_lock_free_allocator();
    test_operation_bound_holds();
    test_workspace_matches_internal();
    test_workspace_scales_with_threads();
    test_copy_after_decompose();
    test_no_allocator_calls_from_parallel_regions();
    test_fill_plan_is_exact_and_stable();
    return test_report("test_allocator");
}
