/** @file test_gpu.c
 * The OpenCL backend: a device decomposition must agree with the CPU one.
 *
 * Every case is a comparison rather than an absolute number -- a device
 * factorization is the same arithmetic with a different summation order, so
 * the checks are the tolerances that order deserves, and the reference is the
 * CPU path the rest of the suite already trusts.
 *
 * With no OpenCL device -- no runtime, no driver, or a build without the
 * backend -- every case is skipped and the test reports success: a machine that
 * cannot run the kernels is not a machine the test can fail on.
 *
 * The vendor's OpenCL runtime leaks a few kilobytes of its own at exit, which
 * LeakSanitizer reports against ``libOpenCL.so``: run this under the
 * sanitizers with ``ASAN_OPTIONS=detect_leaks=0`` to see ours, or without
 * them to see everything.
 */
#include "test_util.h"

#include <hybsol/hybsol.h>
#include <hybsol/opencl.h>

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#define MAX_BLOCKS 6
#define MAX_SIZE 3
#define MAX_DIM (MAX_BLOCKS * MAX_SIZE)

/** Draw the next number of a fixed-seed generator, so a failure reproduces. */
static uint64_t rng_next(uint64_t *const state)
{
    *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
    return *state >> 11;
}

/** What a device decomposition has to match: the CPU solve of the same system. */
static void compare_solves(const hybsol_decomposition_t *const gpu, const hybsol_decomposition_t *const cpu,
                           const double tolerance, const char *const what)
{
    const uint64_t total = hybsol_decomposition_total_size(cpu);
    double *const vec = (double *)malloc((size_t)total * sizeof(double));
    double *const reference = (double *)malloc((size_t)total * sizeof(double));
    CHECK(vec != NULL && reference != NULL);
    if (vec == NULL || reference == NULL)
    {
        free(vec);
        free(reference);
        return;
    }

    for (uint64_t i = 0; i < total; ++i)
    {
        vec[i] = 1.0 + 0.25 * (double)i;
        reference[i] = vec[i];
    }
    CHECK_OK(hybsol_decomposition_solve(cpu, reference, 1));
    CHECK_OK(hybsol_decomposition_solve(gpu, vec, 0));

    double worst = 0.0;
    for (uint64_t i = 0; i < total; ++i)
    {
        const double difference = fabs(vec[i] - reference[i]);
        if (difference > worst)
        {
            worst = difference;
        }
    }
    CHECK_MSG(worst <= tolerance, "%s: the device solve differs from the CPU's by %.3e, which is over %.3e", what,
              worst, tolerance);

    free(vec);
    free(reference);
}

/**
 * A system with a random pattern in ``precision``, diagonally dominant so the
 * factorization is stable enough for the tolerances here.
 */
static hybsol_system_t *build_system(uint64_t *const state, const hybsol_precision_t precision, const uint64_t n_blocks,
                                     uint64_t *const sizes, double dense[MAX_DIM][MAX_DIM])
{
    uint64_t offsets[MAX_BLOCKS + 1] = {0};
    for (uint64_t i = 0; i < n_blocks; ++i)
    {
        sizes[i] = 1 + rng_next(state) % MAX_SIZE;
        offsets[i + 1] = offsets[i] + sizes[i];
    }
    const uint64_t dim = offsets[n_blocks];

    bool present[MAX_BLOCKS][MAX_BLOCKS] = {{false}};
    for (uint64_t i = 0; i < n_blocks; ++i)
    {
        present[i][i] = true;
    }
    for (uint64_t i = 0; i < n_blocks; ++i)
    {
        for (uint64_t j = i + 1; j < n_blocks; ++j)
        {
            if (rng_next(state) % 10 < 4)
            {
                continue;
            }
            present[i][j] = present[j][i] = true;
            for (uint64_t a = 0; a < sizes[i]; ++a)
            {
                for (uint64_t b = 0; b < sizes[j]; ++b)
                {
                    const double value = (double)(rng_next(state) % 1000) * 0.001 - 0.5;
                    dense[offsets[i] + a][offsets[j] + b] = value;
                    dense[offsets[j] + b][offsets[i] + a] = value;
                }
            }
        }
    }

    for (uint64_t i = 0; i < dim; ++i)
    {
        double sum = 1.0;
        for (uint64_t j = 0; j < dim; ++j)
        {
            sum += fabs(dense[i][j]);
        }
        dense[i][i] += sum;
    }

    hybsol_system_t *sys = NULL;
    CHECK_OK(hybsol_system_create_with_precision(n_blocks, sizes, precision, &sys, &CUTL_STD_ALLOCATOR));
    for (uint64_t i = 0; i < n_blocks; ++i)
    {
        for (uint64_t j = 0; j < n_blocks; ++j)
        {
            if (!present[i][j])
            {
                continue;
            }
            double block[MAX_SIZE * MAX_SIZE];
            float narrow[MAX_SIZE * MAX_SIZE];
            for (uint64_t a = 0; a < sizes[i]; ++a)
            {
                for (uint64_t b = 0; b < sizes[j]; ++b)
                {
                    block[a * sizes[j] + b] = dense[offsets[i] + a][offsets[j] + b];
                    narrow[a * sizes[j] + b] = (float)block[a * sizes[j] + b];
                }
            }
            CHECK_OK(precision == HYBSOL_PRECISION_SINGLE
                         ? hybsol_system_add_block_f32(sys, i, j, sizes[i], sizes[j], narrow)
                         : hybsol_system_add_block(sys, i, j, sizes[i], sizes[j], block));
        }
    }
    CHECK(hybsol_system_is_valid(sys));
    return sys;
}

/** The first device that can take a decomposition, or ``NULL``. */
static hybsol_opencl_device_t *acquire_device(void)
{
    for (uint64_t i = 0; i < hybsol_opencl_device_count(); ++i)
    {
        hybsol_opencl_device_info_t info;
        if (hybsol_opencl_device_info(i, &info) != HYBSOL_SUCCESS || !info.supports_double)
        {
            continue;
        }
        hybsol_opencl_device_t *device = NULL;
        if (hybsol_opencl_device_acquire(i, &device) == HYBSOL_SUCCESS)
        {
            return device;
        }
    }
    return NULL;
}

/** What the runtime reports must be filled in: a name, a vendor, a kind. */
static void test_device_enumeration(void)
{
    const uint64_t count = hybsol_opencl_device_count();
    CHECK_MSG(count > 0, "no OpenCL device to test with");

    for (uint64_t i = 0; i < count; ++i)
    {
        hybsol_opencl_device_info_t info;
        CHECK_OK(hybsol_opencl_device_info(i, &info));
        CHECK_MSG(info.name[0] != '\0', "device %llu has no name", (unsigned long long)i);
        CHECK_MSG(info.vendor[0] != '\0', "device %llu has no vendor", (unsigned long long)i);
        CHECK_MSG(info.kind != HYBSOL_OPENCL_DEVICE_UNKNOWN, "device %llu has no kind", (unsigned long long)i);
    }

    // An index past the end is refused, not invented.
    hybsol_opencl_device_info_t info;
    CHECK_RESULT(hybsol_opencl_device_info(count, &info), HYBSOL_ERROR_NO_DEVICE);
}

/** The same factorizations on both paths, in both precisions, agree. */
static void test_parity_in_both_precisions(hybsol_opencl_device_t *const device)
{
    for (int single = 0; single < 2; ++single)
    {
        const hybsol_precision_t precision = single ? HYBSOL_PRECISION_SINGLE : HYBSOL_PRECISION_DOUBLE;
        double dense[MAX_DIM][MAX_DIM];
        uint64_t sizes[MAX_BLOCKS];
        uint64_t state = 0x9e3779b97f4a7c15ULL + (uint64_t)single;
        hybsol_system_t *const sys = build_system(&state, precision, 5, sizes, dense);
        CHECK(sys != NULL);
        if (sys == NULL)
        {
            continue;
        }

        hybsol_elimination_t *graph = NULL;
        CHECK_OK(hybsol_elimination_create(sys, &graph, NULL));

        hybsol_decomposition_t *cpu = NULL;
        CHECK_OK(hybsol_decomposition_create_with_precision(sys, graph, precision, &cpu));
        CHECK_OK(hybsol_decomposition_factorize(cpu, 1));
        CHECK(hybsol_decomposition_device(cpu) == UINT64_MAX);

        hybsol_decomposition_t *gpu = NULL;
        CHECK_OK(hybsol_opencl_decomposition_create_on_device(sys, graph, precision, device, &gpu));
        CHECK_OK(hybsol_decomposition_factorize(gpu, 0));
        CHECK_MSG(hybsol_decomposition_device(gpu) != UINT64_MAX, "the device decomposition reports no device");
        CHECK(hybsol_decomposition_is_factorized(gpu));
        CHECK(hybsol_decomposition_n_operations(gpu) == hybsol_decomposition_n_operations(cpu));

        compare_solves(gpu, cpu, single ? 1e-4 : 1e-9, single ? "single-precision parity" : "double-precision parity");

        hybsol_decomposition_destroy(gpu);
        hybsol_decomposition_destroy(cpu);
        hybsol_elimination_destroy(graph);
        hybsol_system_destroy(sys);
    }
}

/** A double system may produce single factors on a device, and the reverse. */
static void test_mixed_precision_on_device(hybsol_opencl_device_t *const device)
{
    static const struct
    {
        hybsol_precision_t system_precision;
        hybsol_precision_t factor_precision;
        double tolerance;
        const char *what;
    } cases[] = {
        {HYBSOL_PRECISION_DOUBLE, HYBSOL_PRECISION_SINGLE, 1e-4, "double system, single factors"},
        {HYBSOL_PRECISION_SINGLE, HYBSOL_PRECISION_DOUBLE, 1e-9, "single system, double factors"},
    };

    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); ++c)
    {
        double dense[MAX_DIM][MAX_DIM];
        uint64_t sizes[MAX_BLOCKS];
        uint64_t state = 12345 + (uint64_t)c;
        hybsol_system_t *const sys = build_system(&state, cases[c].system_precision, 4, sizes, dense);
        CHECK(sys != NULL);
        if (sys == NULL)
        {
            continue;
        }

        hybsol_elimination_t *graph = NULL;
        CHECK_OK(hybsol_elimination_create(sys, &graph, NULL));

        hybsol_decomposition_t *cpu = NULL;
        CHECK_OK(hybsol_decomposition_create_with_precision(sys, graph, cases[c].factor_precision, &cpu));
        CHECK_OK(hybsol_decomposition_factorize(cpu, 1));

        hybsol_decomposition_t *gpu = NULL;
        CHECK_OK(hybsol_opencl_decomposition_create_on_device(sys, graph, cases[c].factor_precision, device, &gpu));
        CHECK_OK(hybsol_decomposition_factorize(gpu, 0));

        compare_solves(gpu, cpu, cases[c].tolerance, cases[c].what);

        hybsol_decomposition_destroy(gpu);
        hybsol_decomposition_destroy(cpu);
        hybsol_elimination_destroy(graph);
        hybsol_system_destroy(sys);
    }
}

/** The block a singular diagonal names is the block the CPU names. */
static void test_singular_block_agrees(hybsol_opencl_device_t *const device)
{
    const uint64_t sizes[1] = {2};
    hybsol_system_t *sys = NULL;

    // A rank-one diagonal block: nonzero, so the walk accepts the order, but
    // its second pivot comes out exactly zero. Only the factorization finds it.
    const double rank_one[4] = {1.0, 2.0, 2.0, 4.0};
    CHECK_OK(hybsol_system_create(1, sizes, &sys, &CUTL_STD_ALLOCATOR));
    CHECK_OK(hybsol_system_add_block(sys, 0, 0, 2, 2, rank_one));
    CHECK(hybsol_system_is_valid(sys));

    hybsol_elimination_t *graph = NULL;
    CHECK_OK(hybsol_elimination_create(sys, &graph, NULL));

    hybsol_decomposition_t *cpu = NULL;
    CHECK_OK(hybsol_decomposition_create(sys, graph, &cpu));
    CHECK_RESULT(hybsol_decomposition_factorize(cpu, 1), HYBSOL_ERROR_SINGULAR);

    hybsol_decomposition_t *gpu = NULL;
    CHECK_OK(hybsol_opencl_decomposition_create_on_device(sys, graph, HYBSOL_PRECISION_DOUBLE, device, &gpu));
    CHECK_RESULT(hybsol_decomposition_factorize(gpu, 0), HYBSOL_ERROR_SINGULAR);
    CHECK_MSG(hybsol_decomposition_failing_block(gpu) == hybsol_decomposition_failing_block(cpu),
              "the device named block %llu, the CPU named %llu",
              (unsigned long long)hybsol_decomposition_failing_block(gpu),
              (unsigned long long)hybsol_decomposition_failing_block(cpu));
    CHECK(!hybsol_decomposition_is_factorized(gpu));

    hybsol_decomposition_destroy(gpu);
    hybsol_decomposition_destroy(cpu);
    hybsol_elimination_destroy(graph);
    hybsol_system_destroy(sys);
}

/** Replaying the operations and back-substituting is the same solve. */
static void test_replay_matches_solve(hybsol_opencl_device_t *const device)
{
    double dense[MAX_DIM][MAX_DIM];
    uint64_t sizes[MAX_BLOCKS];
    uint64_t state = 777;
    hybsol_system_t *const sys = build_system(&state, HYBSOL_PRECISION_DOUBLE, 4, sizes, dense);
    CHECK(sys != NULL);
    if (sys == NULL)
    {
        return;
    }

    hybsol_elimination_t *graph = NULL;
    CHECK_OK(hybsol_elimination_create(sys, &graph, NULL));

    hybsol_decomposition_t *gpu = NULL;
    CHECK_OK(hybsol_opencl_decomposition_create_on_device(sys, graph, HYBSOL_PRECISION_DOUBLE, device, &gpu));
    CHECK_OK(hybsol_decomposition_factorize(gpu, 0));

    const uint64_t total = hybsol_decomposition_total_size(gpu);
    const uint64_t n_ops = hybsol_decomposition_n_operations(gpu);
    CHECK(n_ops > 0);

    hybsol_operation_t *const ops = (hybsol_operation_t *)malloc(sizeof(*ops) * n_ops);
    double *const solved = (double *)malloc((size_t)total * sizeof(double));
    double *const replayed = (double *)malloc((size_t)total * sizeof(double));
    CHECK(ops != NULL && solved != NULL && replayed != NULL);
    if (ops != NULL && solved != NULL && replayed != NULL)
    {
        uint64_t written = 0;
        CHECK_OK(hybsol_decomposition_operations(gpu, ops, n_ops, &written));
        CHECK(written == n_ops);

        for (uint64_t i = 0; i < total; ++i)
        {
            solved[i] = 0.5 + (double)i;
            replayed[i] = solved[i];
        }
        CHECK_OK(hybsol_decomposition_solve(gpu, solved, 0));
        hybsol_decomposition_apply_operations(gpu, written, ops, replayed);
        hybsol_decomposition_solve_upper(gpu, replayed);

        for (uint64_t i = 0; i < total; ++i)
        {
            CHECK_NEAR(replayed[i], solved[i], 1e-12);
        }
    }

    free(ops);
    free(solved);
    free(replayed);
    hybsol_decomposition_destroy(gpu);
    hybsol_elimination_destroy(graph);
    hybsol_system_destroy(sys);
}

/** Acquire twice, release twice: the device outlives one owner's release. */
static void test_device_lifetime(hybsol_opencl_device_t *const device)
{
    hybsol_opencl_device_t *second = NULL;
    CHECK_OK(hybsol_opencl_device_acquire(0, &second));
    CHECK_MSG(second == device, "acquiring the same device twice gave two handles");
    hybsol_opencl_device_release(second);

    double dense[MAX_DIM][MAX_DIM];
    uint64_t sizes[MAX_BLOCKS];
    uint64_t state = 31;
    hybsol_system_t *sys = build_system(&state, HYBSOL_PRECISION_DOUBLE, 3, sizes, dense);
    CHECK(sys != NULL);
    if (sys == NULL)
    {
        return;
    }

    hybsol_elimination_t *graph = NULL;
    CHECK_OK(hybsol_elimination_create(sys, &graph, NULL));
    hybsol_decomposition_t *gpu = NULL;
    CHECK_OK(hybsol_opencl_decomposition_create_on_device(sys, graph, HYBSOL_PRECISION_DOUBLE, device, &gpu));
    CHECK_OK(hybsol_decomposition_factorize(gpu, 0));
    hybsol_decomposition_destroy(gpu);
    hybsol_elimination_destroy(graph);
    hybsol_system_destroy(sys);

    // Still usable after that decomposition went away.
    sys = build_system(&state, HYBSOL_PRECISION_DOUBLE, 3, sizes, dense);
    CHECK(sys != NULL);
    if (sys == NULL)
    {
        return;
    }
    hybsol_elimination_t *g2 = NULL;
    CHECK_OK(hybsol_elimination_create(sys, &g2, NULL));
    hybsol_decomposition_t *d2 = NULL;
    CHECK_OK(hybsol_opencl_decomposition_create_on_device(sys, g2, HYBSOL_PRECISION_DOUBLE, device, &d2));
    CHECK_OK(hybsol_decomposition_factorize(d2, 0));
    hybsol_decomposition_destroy(d2);
    hybsol_elimination_destroy(g2);
    hybsol_system_destroy(sys);
}

int main(void)
{
    if (hybsol_opencl_device_count() == 0)
    {
        printf("no OpenCL device; skipping the device tests\n");
        return test_report("test_gpu");
    }

    test_device_enumeration();

    hybsol_opencl_device_t *const device = acquire_device();
    CHECK_MSG(device != NULL, "a device with double arithmetic was expected");
    if (device == NULL)
    {
        printf("no OpenCL device that can do double arithmetic; skipping the device tests\n");
        return test_report("test_gpu");
    }

    test_parity_in_both_precisions(device);
    test_mixed_precision_on_device(device);
    test_singular_block_agrees(device);
    test_replay_matches_solve(device);
    test_device_lifetime(device);

    hybsol_opencl_device_release(device);
    return test_report("test_gpu");
}
