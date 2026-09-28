#pragma once

/** @file test_util.h
 * Minimal CHECK-based test harness shared by the native test executables.
 *
 * Each test is a plain executable that calls ``CHECK``/``CHECK_MSG`` and
 * finishes with ``TEST_REPORT``; a non-zero exit status makes CTest fail.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int test_checks = 0;
static int test_failures = 0;

/** Assert that ``cond`` holds, without further explanation. */
#define CHECK(cond)                                                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        ++test_checks;                                                                                                 \
        if (!(cond))                                                                                                   \
        {                                                                                                              \
            ++test_failures;                                                                                           \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond);                                   \
        }                                                                                                              \
    } while (0)

/** Assert that ``cond`` holds, printing ``...`` when it does not. */
#define CHECK_MSG(cond, ...)                                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        ++test_checks;                                                                                                 \
        if (!(cond))                                                                                                   \
        {                                                                                                              \
            ++test_failures;                                                                                           \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n    ", __FILE__, __LINE__, #cond);                               \
            fprintf(stderr, __VA_ARGS__);                                                                              \
            fputc('\n', stderr);                                                                                       \
        }                                                                                                              \
    } while (0)

/** Assert that ``a`` and ``b`` agree to within ``tol``. */
#define CHECK_NEAR(a, b, tol)                                                                                          \
    CHECK_MSG(fabs((a) - (b)) <= (tol), "expected %g ~= %g (tol %g)", (double)(a), (double)(b), (double)(tol))

/** Assert that ``res`` is :c:enumerator:`HYBSOL_SUCCESS`. */
#define CHECK_OK(res) CHECK_MSG((res) == HYBSOL_SUCCESS, "expected success, got %s", hybsol_result_str(res))

/** Assert that ``res`` is exactly ``expected``. */
#define CHECK_RESULT(res, expected)                                                                                    \
    CHECK_MSG((res) == (expected), "expected %s, got %s", hybsol_result_str(expected), hybsol_result_str(res))

/** Print the outcome of a test executable and return its exit status. */
static int test_report(const char *const name)
{
    if (test_failures != 0)
    {
        fprintf(stderr, "%s: %d of %d checks FAILED\n", name, test_failures, test_checks);
        return 1;
    }
    printf("%s: %d checks passed\n", name, test_checks);
    return 0;
}
