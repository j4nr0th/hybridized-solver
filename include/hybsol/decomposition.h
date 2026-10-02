/**
 * @file hybsol/decomposition.h
 * The factorized form of a system: the destination a decomposition writes into,
 * and the solves that read it back.
 *
 * A decomposition owns a copy of every block its system holds plus every block
 * the fill-in adds, laid out in one region whose size
 * :c:func:`hybsol_decomposition_bytes` fixes exactly. The system it came from
 * is untouched by the factorization, so it can be solved again under a different
 * block order, copied, or released while the decomposition lives on.
 */

#ifndef HYBSOL_DECOMPOSITION_H
#define HYBSOL_DECOMPOSITION_H

#include <hybsol/elimination.h>
#include <hybsol/matrix.h>
#include <hybsol/types.h>

/** The kind of block operation recorded during a decomposition. */
typedef enum hybsol_operation_type
{
    /** Solve with the LU factors of the diagonal block ``idx_row``. */
    HYBSOL_OPERATION_INVERT_DIAGONAL = 0,
    /** Subtract ``block(idx_row, idx_col)`` times the ``idx_col`` block row from ``idx_row``. */
    HYBSOL_OPERATION_ELIMINATE = 1,
} hybsol_operation_type_t;

/**
 * A single step of the block decomposition.
 *
 * For :c:enumerator:`HYBSOL_OPERATION_INVERT_DIAGONAL` only ``idx_row`` is
 * meaningful; for :c:enumerator:`HYBSOL_OPERATION_ELIMINATE`, ``idx_row`` is
 * the row being updated and ``idx_col`` the row it is eliminated with.
 */
typedef struct hybsol_operation
{
    hybsol_operation_type_t type;
    uint64_t idx_row;
    /** ``0`` for :c:enumerator:`HYBSOL_OPERATION_INVERT_DIAGONAL`. */
    uint64_t idx_col;
} hybsol_operation_t;

/**
 * A decomposed system.
 *
 * The type is opaque: :c:func:`hybsol_decomposition_create` builds one and
 * :c:func:`hybsol_decomposition_destroy` releases it.
 */
typedef struct hybsol_decomposition hybsol_decomposition_t;

/**
 * Bytes of scratch :c:func:`hybsol_decomposition_factorize_with_workspace` needs.
 *
 * The buffer is opaque; a caller only ever allocates this many bytes, hands the
 * pointer over and reads the result back out of ``dec``. Reusing one buffer
 * across factorizations is fine.
 *
 * Preconditions: ``sys`` non-``NULL``. Only its block sizes matter, so this may
 * be called before any blocks are added. ``factor_precision`` is the type the
 * decomposition will factor in -- which need not be the system's own: see
 * :c:func:`hybsol_decomposition_create_with_precision`.
 */
size_t hybsol_workspace_bytes(const hybsol_system_t *sys, hybsol_precision_t factor_precision, uint64_t n_threads);

/**
 * Get the bytes :c:func:`hybsol_decomposition_init` writes into, for a
 * decomposition whose factors are stored in ``factor_precision``.
 *
 * A function of the graph alone, so a caller can carve one region for a graph,
 * a decomposition and its workspace and release them together.
 */
size_t hybsol_decomposition_bytes(const hybsol_elimination_t *graph, hybsol_precision_t factor_precision);

/**
 * Lay a decomposition out in caller-owned storage and copy the system's blocks
 * into it.
 *
 * The pattern is the one :c:func:`hybsol_elimination_create` found, so every
 * block a factorization can ever need has a slot before the first thread starts,
 * and the factorization itself never allocates. The system is only read: its own
 * blocks are left exactly as they were assembled.
 *
 * The decomposition copies the schedule it needs out of ``graph`` and holds no
 * reference to it, so the graph may be destroyed as soon as this returns.
 * ``storage`` must stay valid and untouched until
 * :c:func:`hybsol_decomposition_destroy` returns; a decomposition built this way
 * frees nothing.
 *
 * Preconditions: ``storage`` at least :c:func:`hybsol_decomposition_bytes` bytes;
 * ``sys``, ``graph``, ``storage`` and ``out`` non-``NULL``; ``graph`` asserted
 * to have been built from ``sys``.
 *
 * Allocates nothing, so the only way it can fail is a violated precondition:
 * it always returns :c:enumerator:`HYBSOL_SUCCESS` with ``*out`` set.
 */
hybsol_result_t hybsol_decomposition_init(const hybsol_system_t *sys, const hybsol_elimination_t *graph, void *storage,
                                          hybsol_decomposition_t **out);

/**
 * Lay a decomposition out in caller-owned storage with factors in ``precision``,
 * copying the system's blocks in: :c:func:`hybsol_decomposition_init` for a
 * factorization that need not match the system's own precision.
 *
 * The blocks convert element by element on the way in -- double factors narrow
 * to single, single factors widen to double -- and the decomposition factors and
 * stores everything in ``precision`` from then on. Filling in more accurate
 * factors than the system holds therefore only makes the factorization itself
 * exact; the fill-in and cancellation errors remain those of the system's own
 * values.
 *
 * The same preconditions as :c:func:`hybsol_decomposition_init`, with ``storage``
 * at least :c:func:`hybsol_decomposition_bytes` bytes for ``precision``.
 */
hybsol_result_t hybsol_decomposition_init_with_precision(const hybsol_system_t *sys, const hybsol_elimination_t *graph,
                                                         hybsol_precision_t precision, void *storage,
                                                         hybsol_decomposition_t **out);

/**
 * Allocate the destination of a decomposition and copy the system's blocks into
 * it: :c:func:`hybsol_decomposition_init` in memory of its own.
 *
 * Preconditions: ``sys``, ``graph`` and ``out`` non-``NULL``; ``graph``
 * asserted to still describe ``sys`` -- same block count, precision, block
 * offsets and sparsity pattern. A graph from before blocks were added would
 * otherwise skip them and factorize a different matrix than the caller holds.
 *
 * Returns :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY` on failure, leaving
 * ``*out`` ``NULL``.
 */
hybsol_result_t hybsol_decomposition_create(const hybsol_system_t *sys, const hybsol_elimination_t *graph,
                                            hybsol_decomposition_t **out);

/**
 * Allocate a decomposition with factors in ``precision`` and copy the system's
 * blocks into it: :c:func:`hybsol_decomposition_init_with_precision` in memory
 * of its own.
 *
 * The same preconditions as :c:func:`hybsol_decomposition_create`; returns
 * :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY` on failure, leaving ``*out`` ``NULL``.
 */
hybsol_result_t hybsol_decomposition_create_with_precision(const hybsol_system_t *sys,
                                                           const hybsol_elimination_t *graph,
                                                           hybsol_precision_t precision, hybsol_decomposition_t **out);

/**
 * Release a decomposition and whatever memory it owns. A decomposition built by
 * :c:func:`hybsol_decomposition_init` frees nothing; ``NULL`` does nothing.
 */
void hybsol_decomposition_destroy(hybsol_decomposition_t *dec);

/** Get ``n``, the number of blocks per dimension. */
uint64_t hybsol_decomposition_n_blocks(const hybsol_decomposition_t *dec);

/** Get the sum of all block sizes, the dimension of the matrix it solves. */
uint64_t hybsol_decomposition_total_size(const hybsol_decomposition_t *dec);

/** Returns ``1`` once :c:func:`hybsol_decomposition_factorize` has succeeded. */
int hybsol_decomposition_is_factorized(const hybsol_decomposition_t *dec);

/**
 * Get the block whose diagonal could not be factorized, or ``UINT64_MAX`` when
 * the last factorization did not fail on a block.
 */
uint64_t hybsol_decomposition_failing_block(const hybsol_decomposition_t *dec);

/**
 * Get the length of the list :c:func:`hybsol_decomposition_operations` writes.
 *
 * Known as soon as the graph is, so it is available before the factorization
 * runs; the list itself is materialized on demand.
 */
uint64_t hybsol_decomposition_n_operations(const hybsol_decomposition_t *dec);

/**
 * Write the operations a factorization performs, in order.
 *
 * Deterministic: the same system gives the same operations in the same order
 * every time, so two lists compare position by position. They must be replayed
 * front to back to apply the lower-triangular factor.
 *
 * Preconditions: ``capacity >=`` :c:func:`hybsol_decomposition_n_operations`,
 * ``out`` non-``NULL``. ``n_written`` may be ``NULL``.
 */
hybsol_result_t hybsol_decomposition_operations(const hybsol_decomposition_t *dec, hybsol_operation_t *out,
                                                uint64_t capacity, uint64_t *n_written);

/**
 * Factorize the decomposition in place, using scratch the caller supplies.
 *
 * The transient scratch is ``workspace_bytes``, sized by
 * :c:func:`hybsol_workspace_bytes`. No allocation happens inside a parallel
 * region, so the decomposition's allocator need not be thread-safe.
 *
 * Preconditions: ``dec`` non-``NULL`` and not yet factorized (asserted),
 * ``workspace`` non-``NULL`` and at least
 * :c:func:`hybsol_workspace_bytes` bytes for the same system and thread count.
 * ``n_threads`` of ``0`` selects the OpenMP default and ``1`` runs serially.
 *
 * Returns :c:enumerator:`HYBSOL_ERROR_SINGULAR` if a diagonal block has no
 * pivot, in which case :c:func:`hybsol_decomposition_failing_block` names it.
 */
hybsol_result_t hybsol_decomposition_factorize_with_workspace(hybsol_decomposition_t *dec, void *workspace,
                                                              size_t workspace_bytes, uint64_t n_threads);

/**
 * Factorize the decomposition in place, allocating the scratch that
 * :c:func:`hybsol_decomposition_factorize_with_workspace` would take from the
 * caller. Callers factorizing repeatedly, or that care where the memory comes
 * from, should size a buffer once and use that spelling instead.
 *
 * Preconditions: ``dec`` non-``NULL`` and not yet factorized (asserted).
 * ``n_threads`` of ``0`` selects the OpenMP default and ``1`` runs serially.
 *
 * Returns :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY` if the scratch cannot be
 * allocated, otherwise as
 * :c:func:`hybsol_decomposition_factorize_with_workspace`.
 */
hybsol_result_t hybsol_decomposition_factorize(hybsol_decomposition_t *dec, uint64_t n_threads);

/**
 * Apply a list of operations to a vector, applying ``L^{-1}``.
 *
 * ``vec`` has length :c:func:`hybsol_decomposition_total_size` and is modified
 * in place.
 *
 * Preconditions: ``dec`` and ``vec`` non-``NULL``, ``ops`` holding ``n_ops``
 * operations that :c:func:`hybsol_decomposition_operations` could have produced.
 * The decomposition need not be factorized: this replays a list, so it applies
 * whatever multipliers that list names.
 */
void hybsol_decomposition_apply_operations(const hybsol_decomposition_t *dec, uint64_t n_ops,
                                           const hybsol_operation_t *ops, double *vec);

/**
 * Solve the upper-triangular system by block back-substitution.
 *
 * Assumes the diagonal blocks are the identity (which is what the factorization
 * produces) and that the vector has already had the forward substitution applied
 * to it. Serial: a row depends on every higher column it holds, which is a
 * different dependency from the elimination's passes.
 *
 * Preconditions: ``dec`` non-``NULL``. The decomposition need not be
 * factorized, but ``vec`` must already hold the forward-substituted right-hand
 * side for the answer to mean anything.
 */
void hybsol_decomposition_solve_upper(const hybsol_decomposition_t *dec, double *vec);

/**
 * Solve ``A x = b`` for ``b`` stored in ``vec``, replacing it with the solution.
 *
 * The forward substitution runs one elimination pass at a time, parallel across
 * the block rows of a pass; the back substitution that follows is serial. The
 * arithmetic per block row does not depend on the thread count, so the answer is
 * the same however many threads are used, and the vector may be solved any
 * number of times.
 *
 * Preconditions: ``dec`` non-``NULL`` and factorized (asserted -- run
 * :c:func:`hybsol_decomposition_is_factorized` first if the caller must not
 * abort), ``vec`` non-``NULL``. ``n_threads`` of ``0`` selects the OpenMP
 * default and ``1`` runs serially.
 *
 * Returns :c:enumerator:`HYBSOL_SUCCESS`; the arithmetic either succeeds or
 * trips an assertion above.
 */
hybsol_result_t hybsol_decomposition_solve(const hybsol_decomposition_t *dec, double *vec, uint64_t n_threads);

#endif /* HYBSOL_DECOMPOSITION_H */
