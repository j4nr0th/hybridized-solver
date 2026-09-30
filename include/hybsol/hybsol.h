/**
 * @file hybsol/hybsol.h
 * Convenience umbrella header for the hybsol C API.
 *
 * Include this single header to get the whole public API:
 *
 * .. code-block:: c
 *
 *    #include <hybsol/hybsol.h>
 *
 * .. topic:: Preconditions
 *
 *    Conditions the caller can be asked to guarantee — index ranges,
 *    non-``NULL`` output pointers, matching shapes, matching precision
 *    spelling, a permutation, a present diagonal block — are *preconditions*.
 *    They are checked with cutl's ``CUTL_ASSERT``, which aborts with a
 *    diagnostic rather than returning a code, and each function documents them
 *    alongside its parameters (typically "asserted …"). Set ``CUTL_ASSERTS``
 *    when building the core to enable the checks; see the build options for
 *    the default.
 *
 *    Conditions that are outcomes of the data rather than mistakes stay
 *    returned :c:type:`hybsol_result_t` codes: allocation failure, a singular
 *    matrix, a system that does not decompose, and the other genuine runtime
 *    states. Callers that must not abort — the Python bindings, most
 *    obviously — check these conditions themselves and raise instead.
 */

#ifndef HYBSOL_HYBSOL_H
#define HYBSOL_HYBSOL_H

#include <hybsol/block_system.h>
#include <hybsol/decompose.h>
#include <hybsol/matrix.h>
#include <hybsol/ordering.h>
#include <hybsol/types.h>

#endif /* HYBSOL_HYBSOL_H */
