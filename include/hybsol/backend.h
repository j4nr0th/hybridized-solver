/**
 * @file hybsol/backend.h
 * The contract a compute backend implements: a per-decomposition vtable the
 * core calls wherever it would otherwise do the arithmetic itself.
 *
 * A backend is in-tree and links the library's internals, so a new runtime is
 * one new library against this header, with no change to the core.
 */

#ifndef HYBSOL_BACKEND_H
#define HYBSOL_BACKEND_H

#include <hybsol/decomposition.h>
#include <hybsol/types.h>

/**
 * The operations the core hands a decomposition's work to.
 *
 * Every function takes the backend's own state first — whatever its create
 * equivalent attached to the decomposition — and ``destroy`` is called
 * exactly once, when the decomposition is destroyed.
 */
typedef struct hybsol_backend
{
    /** Short name of the backend, such as ``"opencl"``. For diagnostics. */
    const char *name;
    /**
     * Factorize in place, ignoring ``n_threads`` if the backend has a scheduler of its own.
     *
     * Returns :c:enumerator:`HYBSOL_ERROR_SINGULAR` with the decomposition's ``failing_block``
     * set when a diagonal has no pivot, as the CPU factorization does.
     */
    hybsol_result_t (*factorize)(void *state, uint64_t n_threads);
    /** Solve ``A x = b`` in ``vec``, replacing it with the solution; ``vec`` is double whatever the factors are. */
    hybsol_result_t (*solve)(void *state, double *vec, uint64_t n_threads);
    /** Apply a recorded operation list to ``vec``, in order, as :c:func:`hybsol_decomposition_apply_operations`. */
    void (*apply_operations)(void *state, uint64_t n_ops, const hybsol_operation_t *ops, double *vec);
    /** Solve ``U x = y`` by block back-substitution, as ``solve_upper`` does. */
    void (*solve_upper)(void *state, double *vec);
    /** The index of the device the factors live on, or ``UINT64_MAX`` when they are in host memory. */
    uint64_t (*device)(const void *state);
    /** Release everything the backend attached to the decomposition, before the core frees the frame itself. */
    void (*destroy)(void *state);
} hybsol_backend_t;

#endif /* HYBSOL_BACKEND_H */
