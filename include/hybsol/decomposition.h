/**
 * @file hybsol/decomposition.h
 * The factorized form of a system: the destination a decomposition writes into,
 * and the solves that read it back.
 *
 * A decomposition owns a copy of every block its system holds plus every block
 * the fill-in adds, laid out in one allocation sized exactly by
 * :c:func:`hybsol_elimination_create`. The system it came from is untouched by
 * the factorization, so it can be solved again under a different block order,
 * copied, or released while the decomposition lives on.
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
    /** Which operation this describes. */
    hybsol_operation_type_t type;
    /** Row the operation acts on. */
    uint64_t idx_row;
    /** Source row; ``0`` for :c:enumerator:`HYBSOL_OPERATION_INVERT_DIAGONAL`. */
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
 * The buffer is opaque; a caller only ever allocates this many bytes, hands
 * the pointer over and reads the result back out of ``dec``. Reusing one
 * buffer across factorizations is fine.
 *
 * :param sys: The system that will be decomposed. Only its block sizes and
 *     precision matter, so this can be called before any blocks are added.
 * :param n_threads: The thread count that will be used.
 * :returns: The number of bytes to supply; ``0`` if ``sys`` is ``NULL``.
 */
size_t hybsol_workspace_bytes(const hybsol_system_t *sys, uint64_t n_threads);

/**
 * Allocate the destination of a decomposition and copy the system's blocks into it.
 *
 * The pattern is the one :c:func:`hybsol_elimination_create` found, so every
 * block a factorization can ever need has a slot before the first thread
 * starts, and the factorization itself never allocates. The system is only
 * read: its own blocks are left exactly as they were assembled.
 *
 * The decomposition copies the schedule it needs out of ``graph`` and holds no
 * reference to it, so the graph may be destroyed as soon as this returns.
 *
 * :param sys: The system to copy the blocks from. Must not be ``NULL``.
 * :param graph: A graph of ``sys``; asserted to match its block structure and
 *     precision. Must not be ``NULL``.
 * :param out: Receives the decomposition. Set to ``NULL`` on failure. Must not
 *     be ``NULL``.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS` or
 *     :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY`.
 */
hybsol_result_t hybsol_decomposition_create(const hybsol_system_t *sys, const hybsol_elimination_t *graph,
                                            hybsol_decomposition_t **out);

/**
 * Release a decomposition and all memory it owns.
 *
 * :param dec: The decomposition to destroy; ``NULL`` is allowed and does nothing.
 */
void hybsol_decomposition_destroy(hybsol_decomposition_t *dec);

/**
 * Get the number of blocks per dimension.
 *
 * :param dec: The decomposition.
 * :returns: ``n``, the same as :c:func:`hybsol_system_n_blocks`.
 */
uint64_t hybsol_decomposition_n_blocks(const hybsol_decomposition_t *dec);

/**
 * Get the number of rows of the matrix the decomposition solves.
 *
 * :param dec: The decomposition.
 * :returns: The sum of all block sizes.
 */
uint64_t hybsol_decomposition_total_size(const hybsol_decomposition_t *dec);

/**
 * Check whether the decomposition has been factorized.
 *
 * :param dec: The decomposition.
 * :returns: ``1`` if :c:func:`hybsol_decomposition_factorize` has succeeded.
 */
int hybsol_decomposition_is_factorized(const hybsol_decomposition_t *dec);

/**
 * Get the block whose diagonal could not be factorized.
 *
 * :param dec: The decomposition.
 * :returns: The zero-based block index, or ``UINT64_MAX`` when the last
 *     factorization did not fail on a block.
 */
uint64_t hybsol_decomposition_failing_block(const hybsol_decomposition_t *dec);

/**
 * Get the number of operations the decomposition records.
 *
 * Known as soon as the graph is, so it is available before the factorization
 * runs; the list itself is materialized on demand.
 *
 * :param dec: The decomposition.
 * :returns: The length of :c:func:`hybsol_decomposition_operations`.
 */
uint64_t hybsol_decomposition_n_operations(const hybsol_decomposition_t *dec);

/**
 * Write the operations a factorization performs, in order.
 *
 * The list is deterministic: the same system and the same thread count give
 * the same operations in the same order every time, so it can be compared
 * position by position. The operations must be replayed front to back to apply
 * the lower-triangular factor.
 *
 * :param dec: The decomposition.
 * :param out: Destination for ``capacity`` operations. Must not be ``NULL``.
 * :param capacity: Number of operations ``out`` can hold; asserted large
 *     enough for :c:func:`hybsol_decomposition_n_operations`.
 * :param n_written: Receives the number of operations written. May be ``NULL``.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`.
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
 * :param dec: The decomposition to factorize. Must not be ``NULL``.
 * :param workspace: Buffer of at least ``workspace_bytes``; must not be
 *     ``NULL``. Its contents are overwritten.
 * :param workspace_bytes: How large ``workspace`` is, as reported by
 *     :c:func:`hybsol_workspace_bytes`.
 * :param n_threads: Number of OpenMP threads; ``0`` selects the OpenMP
 *     default (usually every core) and ``1`` runs the factorization serially.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`,
 *     :c:enumerator:`HYBSOL_ERROR_ALREADY_DECOMPOSED`,
 *     :c:enumerator:`HYBSOL_ERROR_SINGULAR`,
 *     :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY` or
 *     :c:enumerator:`HYBSOL_ERROR_INTERNAL`. For a singular block
 *     :c:func:`hybsol_decomposition_failing_block` names it.
 */
hybsol_result_t hybsol_decomposition_factorize_with_workspace(hybsol_decomposition_t *dec, void *workspace,
                                                              size_t workspace_bytes, uint64_t n_threads);

/**
 * Factorize the decomposition in place.
 *
 * Allocates the scratch that
 * :c:func:`hybsol_decomposition_factorize_with_workspace` would take from the
 * caller. Callers factorizing repeatedly, or that care where the memory comes
 * from, should size a buffer once and use that spelling instead.
 *
 * :param dec: The decomposition to factorize. Must not be ``NULL``.
 * :param n_threads: Number of OpenMP threads; ``0`` selects the OpenMP
 *     default and ``1`` runs the factorization serially.
 * :returns: The result codes of
 *     :c:func:`hybsol_decomposition_factorize_with_workspace`.
 */
hybsol_result_t hybsol_decomposition_factorize(hybsol_decomposition_t *dec, uint64_t n_threads);

/**
 * Apply a list of operations to a vector, applying ``L^{-1}``.
 *
 * :param dec: The decomposed system.
 * :param n_ops: Number of operations in ``ops``.
 * :param ops: Operations to apply, in order.
 * :param vec: Vector of length :c:func:`hybsol_decomposition_total_size`,
 *     modified in place.
 */
void hybsol_decomposition_apply_operations(const hybsol_decomposition_t *dec, uint64_t n_ops,
                                           const hybsol_operation_t *ops, double *vec);

/**
 * Solve the upper-triangular system by block back-substitution.
 *
 * Assumes the diagonal blocks are the identity (which is what the
 * factorization produces) and that the vector has already had the forward
 * substitution applied to it. Serial: a row depends on every higher column it
 * holds, which is a different dependency from the elimination's passes.
 *
 * :param dec: The decomposed system.
 * :param vec: Vector of length :c:func:`hybsol_decomposition_total_size`,
 *     replaced by the solution.
 */
void hybsol_decomposition_solve_upper(const hybsol_decomposition_t *dec, double *vec);

/**
 * Solve ``A x = b`` for ``b`` stored in ``vec``.
 *
 * The forward substitution runs one pass at a time over the elimination's
 * passes, parallel across the block rows of a pass; the back substitution that
 * follows is serial. The arithmetic per block row does not depend on the
 * thread count, so the answer is the same however many threads are used.
 *
 * The vector may be solved any number of times.
 *
 * :param dec: The decomposed system. Must not be ``NULL``.
 * :param vec: Right-hand side of length
 *     :c:func:`hybsol_decomposition_total_size`, replaced by the solution.
 *     Must not be ``NULL``.
 * :param n_threads: Number of OpenMP threads for the forward substitution;
 *     ``0`` selects the OpenMP default and ``1`` runs it serially.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS` or
 *     :c:enumerator:`HYBSOL_ERROR_NOT_DECOMPOSED`.
 */
hybsol_result_t hybsol_decomposition_solve(const hybsol_decomposition_t *dec, double *vec, uint64_t n_threads);

#endif /* HYBSOL_DECOMPOSITION_H */
