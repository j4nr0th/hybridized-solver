/**
 * @file hybsol/backend.h
 * The contract a compute backend implements. Nothing here is OpenCL, CUDA or
 * anything else: a backend is a per-decomposition vtable, and the core calls
 * through it wherever a factorization's arithmetic would otherwise be.
 *
 * A backend keeps its blocks wherever it likes -- the CPU ones live in the
 * decomposition's own frame, a device one usually on the device -- and the core
 * reaches them only through the operations below. A decomposition that has one
 * filled in is still queried, created and destroyed the same way; what changes
 * is that :c:func:`hybsol_decomposition_factorize`,
 * :c:func:`hybsol_decomposition_solve` and the replay helpers hand the work
 * over instead of computing it.
 *
 * Backends are in-tree: they link :c:enumerator:`HYBSOL` internals to carve
 * their own frame and stage the system's blocks into their own layout. A
 * backend for a new runtime is one new library against :c:file:`hybsol.h` and
 * :c:file:`hybsol/backend.h`, with no change to the core.
 */

#ifndef HYBSOL_BACKEND_H
#define HYBSOL_BACKEND_H

#include <hybsol/decomposition.h>
#include <hybsol/types.h>

/**
 * The operations the core hands a decomposition's work to.
 *
 * Every function takes the backend's own state -- whatever its
 * ``create`` equivalent attached to the decomposition -- as its first
 * argument. The core passes the state's pointer through unchanged and calls
 * ``destroy`` exactly once, when the decomposition is destroyed, whether or not
 * the factorization succeeded.
 *
 * ``state`` is declared ``void *`` rather than the backend's own type so that
 * the core needs to know nothing about it; a backend that wants type safety
 * casts it back to its own state struct, which is the only thing that pointer
 * ever is.
 */
typedef struct hybsol_backend
{
    /** Short name of the backend, such as ``"opencl"``. For diagnostics. */
    const char *name;
    /**
     * Factorize in place.
     *
     * Called by :c:func:`hybsol_decomposition_factorize` with whatever
     * thread count the caller asked for, which a backend is free to ignore --
     * it has a scheduler of its own.
     *
     * On failure it leaves ``failing_block`` set on the decomposition where it
     * applies, the way the CPU factorization names the block whose diagonal hit
     * a zero pivot.
     */
    hybsol_result_t (*factorize)(void *state, uint64_t n_threads);
    /**
     * Solve ``A x = b`` for ``b`` in ``vec``, replacing it with the solution.
     *
     * ``vec`` holds doubles whatever the factors are stored in, as everywhere
     * else in the API.
     */
    hybsol_result_t (*solve)(void *state, double *vec, uint64_t n_threads);
    /**
     * Apply a recorded operation list to a vector, front to back. The same
     * contract as :c:func:`hybsol_decomposition_apply_operations`, which does
     * not require the decomposition to be factorized.
     */
    void (*apply_operations)(void *state, uint64_t n_ops, const hybsol_operation_t *ops, double *vec);
    /** Solve ``U x = y`` by block back-substitution, as ``solve_upper`` does. */
    void (*solve_upper)(void *state, double *vec);
    /**
     * The index this decomposition's factors live on, out of the range
     * :c:func:`hybsol_device_count` would report for the backend, or
     * ``UINT64_MAX`` when the factors are not on a device.
     */
    uint64_t (*device)(const void *state);
    /**
     * Release everything the backend attached to the decomposition. Called by
     * :c:func:`hybsol_decomposition_destroy` before the core releases the
     * frame itself.
     */
    void (*destroy)(void *state);
} hybsol_backend_t;

#endif /* HYBSOL_BACKEND_H */
