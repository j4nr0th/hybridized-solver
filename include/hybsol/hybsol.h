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
 *    non-``NULL`` output pointers, matching shapes and precision spelling, a
 *    permutation, a structurally valid system, an unfactorized decomposition —
 *    are *preconditions*, checked with cutl's ``CUTL_ASSERT`` rather than returned.
 *    Set ``CUTL_ASSERTS`` when building the core to enable them.
 *
 *    What is left as a :c:type:`hybsol_result_t` code is an outcome of the data
 *    rather than a mistake: allocation failure, a singular matrix, a block order
 *    that admits no factorization, and the other genuine runtime states.
 *    Callers that must not abort — the Python bindings, most obviously — check
 *    these themselves and raise instead.
 */

#ifndef HYBSOL_HYBSOL_H
#define HYBSOL_HYBSOL_H

#include <hybsol/backend.h>
#include <hybsol/block_system.h>
#include <hybsol/decomposition.h>
#include <hybsol/elimination.h>
#include <hybsol/matrix.h>
#include <hybsol/ordering.h>
#include <hybsol/types.h>

#endif /* HYBSOL_HYBSOL_H */
