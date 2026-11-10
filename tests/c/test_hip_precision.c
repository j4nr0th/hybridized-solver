/** @file test_hip_precision.c
 * The HIP selection harness: time and accuracy of each arithmetic mode,
 * the table that picks the default.
 *
 * One seeded (fixed-LCG, per the repo's test style) banded SPD system per
 * block size -- 128 blocks of 8, 16 and 32 -- and one row per (size, mode):
 * the median of 7 create+factorize+solve rounds, the relative residual of
 * that solve against the assembled matrix, and the conditioning bound.
 * native and ozaki gate on residual <= max(cond*1e-10,
 * 1e-8); ozaki2 is asymmetric slicing, a refinement partner by design, so
 * its accuracy is printed and never gated. The timings drive
 * ``HYBSOL_HIP_DEFAULT_FOR_GFX12`` in src/hip/device.c; the printed table
 * belongs in that commit's message. Each row carries a load probe -- one
 * trivial GPU round trip, in milliseconds, at the start and the end of
 * the mode's rounds -- because these seconds are measured while the GPU
 * may also serve other workloads; rerun at idle for authoritative
 * absolute seconds. Without a device, every case is skipped and the exit
 * status stays zero.
 */
#include "test_util.h"

#include <hybsol/hybsol.h>
#include <hybsol/hip.h>

#include <hip/hip_runtime_api.h>

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N_BLOCKS 128
#define MAX_BLOCK_SIZE 32
#define MAX_DIM (N_BLOCKS * MAX_BLOCK_SIZE)
#define ROUNDS 7

/** Draw the next number of a fixed-seed generator, so a row reproduces. */
static uint64_t rng_next(uint64_t *const state)
{
    *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
    return *state >> 11;
}

/**
 * A seeded banded SPD system: ``n_blocks`` blocks of ``size``, present only
 * on the block-diagonal tridiagonal, every block's entries random, and each
 * diagonal entry raised by one plus its row's off-diagonal absolute sum --
 * the builder test_parity_in_both_precisions uses in test_hip.c, so
 * Gershgorin again bounds the
 * spectrum into [1, 1 + 2R] for the largest such sum R. ``dense`` (which
 * must hold MAX_DIM * MAX_DIM doubles) receives the assembled matrix.
 */
static hybsol_system_t *build_banded(uint64_t *const state, const uint64_t size, double *const dense)
{
    const uint64_t dim = (uint64_t)N_BLOCKS * size;
    memset(dense, 0, (size_t)dim * (size_t)dim * sizeof(double));
    for (uint64_t i = 0; i < (uint64_t)N_BLOCKS; ++i)
    {
        for (uint64_t j = i; j < (uint64_t)N_BLOCKS; ++j)
        {
            if (j > i + 1u)
            {
                continue;
            }
            for (uint64_t a = 0; a < size; ++a)
            {
                for (uint64_t b = 0; b < size; ++b)
                {
                    const double value = (double)(rng_next(state) % 1000) * 0.001 - 0.5;
                    dense[(i * size + a) * dim + j * size + b] = value;
                    if (i != j)
                    {
                        dense[(j * size + b) * dim + i * size + a] = value;
                    }
                }
            }
        }
    }
    for (uint64_t i = 0; i < dim; ++i)
    {
        double sum = 1.0;
        for (uint64_t j = 0; j < dim; ++j)
        {
            sum += fabs(dense[i * dim + j]);
        }
        dense[i * dim + i] += sum;
    }

    uint64_t sizes[N_BLOCKS];
    for (uint64_t i = 0; i < (uint64_t)N_BLOCKS; ++i)
    {
        sizes[i] = size;
    }

    hybsol_system_t *sys = NULL;
    CHECK_OK(hybsol_system_create_with_precision((uint64_t)N_BLOCKS, sizes, HYBSOL_PRECISION_DOUBLE, &sys,
                                                 &CUTL_STD_ALLOCATOR));
    for (uint64_t i = 0; i < (uint64_t)N_BLOCKS; ++i)
    {
        for (uint64_t j = (i > 0u ? i - 1u : 0u); j <= i + 1u && j < (uint64_t)N_BLOCKS; ++j)
        {
            double block[MAX_BLOCK_SIZE * MAX_BLOCK_SIZE];
            for (uint64_t a = 0; a < size; ++a)
            {
                for (uint64_t b = 0; b < size; ++b)
                {
                    block[a * size + b] = dense[(i * size + a) * dim + j * size + b];
                }
            }
            CHECK_OK(hybsol_system_add_block(sys, i, j, size, size, block));
        }
    }
    CHECK(hybsol_system_is_valid(sys));
    return sys;
}

/** The Gershgorin conditioning bound of the built system: 1 + 2 * the widest row sum. */
static double conditioning_bound(const double *const dense, const uint64_t dim)
{
    double worst_row = 0.0;
    for (uint64_t i = 0; i < dim; ++i)
    {
        double row = 0.0;
        for (uint64_t j = 0; j < dim; ++j)
        {
            row += fabs(dense[i * dim + j]);
        }
        row -= fabs(dense[i * dim + i]);
        if (row > worst_row)
        {
            worst_row = row;
        }
    }
    return 1.0 + 2.0 * worst_row;
}

/** The relative infinity residual of ``x`` against the dense system and ``rhs``. */
static double residual_of(const double *const dense, const uint64_t dim, const double *const rhs,
                          const double *const x)
{
    double norm = 0.0;
    for (uint64_t i = 0; i < dim; ++i)
    {
        if (fabs(rhs[i]) > norm)
        {
            norm = fabs(rhs[i]);
        }
    }
    double worst = 0.0;
    for (uint64_t i = 0; i < dim; ++i)
    {
        double row = 0.0;
        for (uint64_t j = 0; j < dim; ++j)
        {
            row += dense[i * dim + j] * x[j];
        }
        const double difference = fabs(row - rhs[i]);
        if (difference > worst)
        {
            worst = difference;
        }
    }
    return worst / norm;
}

/** One mode's row: its arith argument and whether the residual gates. */
typedef struct
{
    const char *name;
    hybsol_hip_arith_t arith;
    int gated;
} mode_case_t;

/** Median of a handful of timings. */
static double median_seconds(double *const seconds)
{
    for (int i = 1; i < ROUNDS; ++i)
    {
        const double key = seconds[i];
        int j = i - 1;
        while (j >= 0 && seconds[j] > key)
        {
            seconds[j + 1] = seconds[j];
            --j;
        }
        seconds[j + 1] = key;
    }
    return seconds[ROUNDS / 2];
}

/**
 * Contention probe: one trivial GPU round trip, timed in milliseconds.
 * Nothing about the modes depends on it -- it just says how loaded the
 * device was around the mode's rounds, so a row read under load can be
 * re-read with the load column in mind.
 */
static double probe_ms(void)
{
    void *buffer = NULL;
    if (hipMalloc(&buffer, 64) != hipSuccess)
    {
        return -1.0;
    }
    struct timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    (void)hipMemset(buffer, 0, 64);
    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    hipFree(buffer);
    return (double)(t1.tv_sec - t0.tv_sec) * 1e3 + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-6;
}

static void run_mode(const hybsol_hip_arith_t arith, const int gated, const char *const name,
                     const hybsol_system_t *const sys, const hybsol_elimination_t *const graph,
                     hybsol_hip_device_t *const device, const double *const dense, const uint64_t dim,
                     const double *const rhs, const double cond)
{
    double seconds[ROUNDS];
    double residual = -1.0;
    hybsol_result_t first_res = HYBSOL_SUCCESS;
    const double probe_before = probe_ms();

    for (int round = 0; round < ROUNDS; ++round)
    {
        hybsol_decomposition_t *dec = NULL;
        const hybsol_result_t res
            = hybsol_hip_decomposition_create_on_device(sys, graph, HYBSOL_PRECISION_DOUBLE, arith, device, &dec);
        if (res != HYBSOL_SUCCESS)
        {
            first_res = res;
            break;
        }
        struct timespec t0;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        const hybsol_result_t fact_res = hybsol_decomposition_factorize(dec, 0);
        double *const x = (double *)malloc((size_t)dim * sizeof(double));
        memcpy(x, rhs, (size_t)dim * sizeof(double));
        const hybsol_result_t solve_res = hybsol_decomposition_solve(dec, x, 0);
        struct timespec t1;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        hybsol_decomposition_destroy(dec);

        if (fact_res != HYBSOL_SUCCESS || solve_res != HYBSOL_SUCCESS || x == NULL)
        {
            free(x);
            first_res = (fact_res != HYBSOL_SUCCESS) ? fact_res : solve_res;
            break;
        }
        const double elapsed = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
        seconds[round] = elapsed;
        residual = residual_of(dense, dim, rhs, x);
        free(x);
    }
    const double probe_after = probe_ms();

    if (first_res != HYBSOL_SUCCESS)
    {
        printf("  %-14s refused: %s (%s)\n", name, hybsol_result_str(first_res), hybsol_hip_last_error());
        return;
    }

    const double secs = median_seconds(seconds);
    const double gate = cond * 1e-10 > 1e-8 ? cond * 1e-10 : 1e-8;
    printf("  %-14s %8.4f s   residual %.3e   cond %.2f   gate %s   load %.3f/%.3f ms\n", name, secs, residual, cond,
           gated ? (residual <= gate ? "PASS" : "FAIL") : "not gated", probe_before, probe_after);
    if (gated)
    {
        CHECK_MSG(residual <= gate, "%s: the residual %.3e is over the gate %.3e (cond <= %.2f)", name, residual,
                  gate, cond);
    }
}

/** One block size: the system, then every mode's timed, gated row. */
static void run_size(const uint64_t size, hybsol_hip_device_t *const device)
{
    uint64_t rng = 0x243f6a8885a308d3ULL + size;
    double *const dense = (double *)malloc((size_t)MAX_DIM * (size_t)MAX_DIM * sizeof(double));
    double x_scratch[MAX_DIM];
    CHECK(dense != NULL);
    if (dense == NULL)
    {
        return;
    }
    hybsol_system_t *const sys = build_banded(&rng, size, dense);
    if (sys == NULL)
    {
        free(dense);
        return;
    }
    const uint64_t dim = hybsol_system_total_size(sys);

    static const mode_case_t modes[] = {
        { "native", HYBSOL_HIP_ARITH_NATIVE, 1 },
        { "ozaki_i8", HYBSOL_HIP_ARITH_OZAKI, 1 },
        { "ozaki2", HYBSOL_HIP_ARITH_OZAKI2, 0 },
    };

    hybsol_elimination_t *graph = NULL;
    CHECK_OK(hybsol_elimination_create(sys, &graph, NULL));

    const double cond = conditioning_bound(dense, dim);
    double *const rhs = x_scratch;
    for (uint64_t i = 0; i < dim; ++i)
    {
        rhs[i] = 1.0 + 0.25 * (double)i;
    }

    printf("block size %llu, %llu blocks, %llu unknowns, cond <= %.2f\n", (unsigned long long)size,
           (unsigned long long)N_BLOCKS, (unsigned long long)dim, cond);
    for (size_t m = 0; m < sizeof(modes) / sizeof(modes[0]); ++m)
    {
        run_mode(modes[m].arith, modes[m].gated, modes[m].name, sys, graph, device, dense, dim, rhs, cond);
    }

    hybsol_elimination_destroy(graph);
    hybsol_system_destroy(sys);
    free(dense);
}

/** The first device that can take a decomposition, or ``NULL``; same rule as test_hip.c. */
static hybsol_hip_device_t *acquire_device(uint64_t *const index_out)
{
    for (uint64_t i = 0; i < hybsol_hip_device_count(); ++i)
    {
        hybsol_hip_device_info_t info;
        if (hybsol_hip_device_info(i, &info) != HYBSOL_SUCCESS || !info.supports_double)
        {
            continue;
        }
        hybsol_hip_device_t *device = NULL;
        if (hybsol_hip_device_acquire(i, &device) == HYBSOL_SUCCESS)
        {
            *index_out = i;
            return device;
        }
    }
    return NULL;
}

int main(void)
{
    if (hybsol_hip_device_count() == 0)
    {
        printf("no HIP device; skipping the precision harness\n");
        return test_report("test_hip_precision");
    }

    uint64_t device_index = 0;
    hybsol_hip_device_t *const device = acquire_device(&device_index);
    if (device == NULL)
    {
        printf("no HIP device that can do double arithmetic; skipping the precision harness\n");
        return test_report("test_hip_precision");
    }

    hybsol_hip_device_info_t info;
    if (hybsol_hip_device_info(device_index, &info) == HYBSOL_SUCCESS)
    {
        printf("HIP precision harness on device %llu ('%s')\n", (unsigned long long)device_index, info.name);
    }

    static const uint64_t sizes[] = { 8, 16, 32 };
    for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); ++s)
    {
        run_size(sizes[s], device);
    }

    hybsol_hip_device_release(device);
    return test_report("test_hip_precision");
}
