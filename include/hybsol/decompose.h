/**
 * @file hybsol/decompose.h
 * Block LU decomposition and the triangular solves that use it.
 *
 * :c:func:`hybsol_system_decompose` factorizes the system once and records
 * the sequence of block operations it performed. :c:func:`hybsol_system_solve`
 * replays those operations against a right-hand side and finishes with a
 * block back-substitution.
 *
 * After a successful decomposition the system is frozen: no further blocks
 * may be added, eliminated or reordered.
 */

#ifndef HYBSOL_DECOMPOSE_H
#define HYBSOL_DECOMPOSE_H

#include <hybsol/block_system.h>
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
 * Bytes of scratch :c:func:`hybsol_system_decompose_with_workspace` needs.
 *
 * The buffer is opaque; a caller only ever allocates this many bytes, hands
 * the pointer over and reads the result back out of ``sys``. Reusing one
 * buffer across decompositions is fine.
 *
 * :param sys: The system that will be decomposed. Only its block sizes and
 *     precision matter, so this can be called before any blocks are added.
 * :param n_threads: The thread count that will be used.
 * :returns: The number of bytes to supply; ``0`` if ``sys`` is ``NULL``.
 */
size_t hybsol_workspace_bytes(const hybsol_system_t *sys, uint64_t n_threads);

/**
 * The most operations a decomposition of ``sys`` will record.
 *
 * One operation per row plus at most one elimination per ``(target, source)``
 * pair, so a dense system reaches this exactly. Sizing the recorded list to it
 * is what lets the decomposition append without locking or growing.
 */
uint64_t hybsol_system_operation_bound(const hybsol_system_t *sys);

/**
 * Decompose the system in place, using scratch the caller supplies.
 *
 * This is :c:func:`hybsol_system_decompose` without the internal allocation:
 * every byte it needs up front is ``workspace_bytes``, sized by
 * :c:func:`hybsol_workspace_bytes`, and nothing is allocated from the system's
 * allocator once the parallel regions start.
 *
 * That holds even when the system's allocator is not thread-safe, because each
 * thread fills a region of its own. The fill-in the decomposition produces
 * lives in those regions and is released with the system, not here.
 *
 * :param sys: The system to decompose. Must not be ``NULL``.
 * :param workspace: Buffer of at least ``workspace_bytes``; must not be
 *     ``NULL``. Its contents are overwritten.
 * :param workspace_bytes: How large ``workspace`` is, as reported by
 *     :c:func:`hybsol_workspace_bytes`.
 * :param n_threads: Number of OpenMP threads; ``0`` selects the OpenMP
 *     default (usually every core) and ``1`` runs the decomposition serially.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`,
 *     :c:enumerator:`HYBSOL_ERROR_SYSTEM_INVALID`,
 *     :c:enumerator:`HYBSOL_ERROR_ALREADY_DECOMPOSED`,
 *     :c:enumerator:`HYBSOL_ERROR_SINGULAR`,
 *     :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY` or
 *     :c:enumerator:`HYBSOL_ERROR_INTERNAL`.
 */
hybsol_result_t hybsol_system_decompose_with_workspace(hybsol_system_t *sys, void *workspace, size_t workspace_bytes,
                                                       uint64_t n_threads);

/**
 * Decompose the system in place.
 *
 * The system must be valid (see :c:func:`hybsol_system_is_valid`). On
 * success the diagonal blocks hold LU factors instead of their original
 * values and the recorded operation list can be replayed with
 * :c:func:`hybsol_system_apply_operations`.
 *
 * The work is embarrassingly parallel up to the point where a row's
 * diagonal is complete; ``n_threads`` controls how many OpenMP threads are
 * used for that phase.
 *
 * This allocates the scratch :c:func:`hybsol_system_decompose_with_workspace`
 * would otherwise take from the caller, from ``sys``'s allocator. Callers that
 * decompose repeatedly, or that care where the memory comes from, should size
 * a buffer once with :c:func:`hybsol_workspace_bytes` and use that spelling.
 *
 * :param sys: The system to decompose.
 * :param n_threads: Number of OpenMP threads; ``0`` selects the OpenMP
 *     default (usually every core) and ``1`` runs the decomposition serially.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`,
 *     :c:enumerator:`HYBSOL_ERROR_SYSTEM_INVALID`,
 *     :c:enumerator:`HYBSOL_ERROR_ALREADY_DECOMPOSED`,
 *     :c:enumerator:`HYBSOL_ERROR_SINGULAR`,
 *     :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY` or
 *     :c:enumerator:`HYBSOL_ERROR_INTERNAL`.
 */
hybsol_result_t hybsol_system_decompose(hybsol_system_t *sys, uint64_t n_threads);

/**
 * Check whether the system has been decomposed.
 *
 * :param sys: The system.
 * :returns: ``1`` if :c:func:`hybsol_system_decompose` has succeeded, ``0`` otherwise.
 */
int hybsol_system_is_decomposed(const hybsol_system_t *sys);

/**
 * Get the number of recorded operations.
 *
 * :param sys: The system.
 * :returns: The length of :c:func:`hybsol_system_operations`; ``0`` if the
 *     system has not been decomposed.
 */
uint64_t hybsol_system_n_operations(const hybsol_system_t *sys);

/**
 * Get the recorded operations.
 *
 * The pointer is owned by the system and stays valid until it is destroyed
 * or decomposed again. The operations must be replayed front to back to apply
 * the lower-triangular factor.
 *
 * Order is fixed *between* elimination passes, because a later pass reads what
 * an earlier one produced. Within a single pass the operations commute -- the
 * targets are distinct, and every source was finished before the pass started
 * -- so which thread records which of them first is not defined, and running a
 * decomposition twice with more than one thread may hand back the same
 * operations in a different order. Treat the list as significant only as a
 * whole, not position by position.
 *
 * :param sys: The system.
 * :returns: An array of :c:func:`hybsol_system_n_operations` entries, or
 *     ``NULL`` if the system has not been decomposed.
 */
const hybsol_operation_t *hybsol_system_operations(const hybsol_system_t *sys);

/**
 * Replace a diagonal block with its LU factorization.
 *
 * :param sys: The system.
 * :param idx: Block row index; asserted in range and to have a diagonal block.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`,
 *     :c:enumerator:`HYBSOL_ERROR_SINGULAR` or
 *     :c:enumerator:`HYBSOL_ERROR_ALREADY_DECOMPOSED`.
 */
hybsol_result_t hybsol_system_decompose_diagonal(hybsol_system_t *sys, uint64_t idx);

/**
 * Solve the diagonal block's factors against the rest of its row.
 *
 * After this, every block ``i > idx`` in the row satisfies
 * ``block(idx, idx) @ block(idx, i) == original``, i.e. the row has been
 * scaled by the inverse of its diagonal block. Requires a preceding call to
 * :c:func:`hybsol_system_decompose_diagonal` for the same ``idx``.
 *
 * :param sys: The system.
 * :param idx: Block row index; asserted in range and to have a diagonal block.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS` or
 *     :c:enumerator:`HYBSOL_ERROR_NOT_DECOMPOSED`.
 */
hybsol_result_t hybsol_system_apply_diagonal_inverse(hybsol_system_t *sys, uint64_t idx);

/**
 * Solve with a previously decomposed diagonal block.
 *
 * :param sys: The system.
 * :param idx: Block row index; asserted in range and to have a diagonal block.
 * :param b: Right-hand side with ``size(idx)`` rows. Must not be ``NULL``.
 * :param x: Destination with the same shape as ``b``; may alias ``b``. Must
 *     not be ``NULL``.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS` or
 *     :c:enumerator:`HYBSOL_ERROR_NOT_DECOMPOSED`.
 */
hybsol_result_t hybsol_system_solve_diagonal(hybsol_system_t *sys, uint64_t idx, const hybsol_matrix_t *b,
                                             const hybsol_matrix_t *x);

/**
 * The single-precision spelling of :c:func:`hybsol_system_solve_diagonal`.
 *
 * ``b`` and ``x`` are floats here, and the system is asserted to have been
 * created with :c:enumerator:`HYBSOL_PRECISION_SINGLE` — the double spelling
 * on such a system asserts rather than widening the factors.
 *
 * :param sys: The system; asserted to store ``float``.
 * :param idx: Block row index; asserted in range and to have a diagonal block.
 * :param b: Right-hand side with ``block_size(idx)`` rows. Must not be ``NULL``.
 * :param x: Destination for the solution, shaped like ``b``. Must not be ``NULL``.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS` or
 *     :c:enumerator:`HYBSOL_ERROR_NOT_DECOMPOSED`.
 */
hybsol_result_t hybsol_system_solve_diagonal_f32(hybsol_system_t *sys, uint64_t idx, const hybsol_fmatrix_t *b,
                                                 const hybsol_fmatrix_t *x);

/**
 * Apply the recorded operations to a vector, applying ``L^{-1}``.
 *
 * :param sys: The decomposed system.
 * :param n_ops: Number of operations in ``ops``.
 * :param ops: Operations to apply, in order.
 * :param vec: Vector of length :c:func:`hybsol_system_total_size`, modified in place.
 */
void hybsol_system_apply_operations(const hybsol_system_t *sys, uint64_t n_ops, const hybsol_operation_t *ops,
                                    double *vec);

/**
 * Solve the upper-triangular system by block back-substitution.
 *
 * Assumes the diagonal blocks are the identity (which is what the
 * decomposition produces) and that the vector has already had
 * :c:func:`hybsol_system_apply_operations` applied to it.
 *
 * :param sys: The decomposed system.
 * :param vec: Vector of length :c:func:`hybsol_system_total_size`, replaced by the solution.
 */
void hybsol_system_solve_upper(const hybsol_system_t *sys, double *vec);

/**
 * Solve ``A x = b`` for ``b`` stored in ``vec``.
 *
 * Convenience wrapper performing :c:func:`hybsol_system_apply_operations`
 * followed by :c:func:`hybsol_system_solve_upper`.
 *
 * :param sys: The decomposed system.
 * :param vec: Right-hand side of length :c:func:`hybsol_system_total_size`,
 *     replaced by the solution. Must not be ``NULL``.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS` or
 *     :c:enumerator:`HYBSOL_ERROR_NOT_DECOMPOSED`.
 */
hybsol_result_t hybsol_system_solve(hybsol_system_t *sys, double *vec);

#endif /* HYBSOL_DECOMPOSE_H */
