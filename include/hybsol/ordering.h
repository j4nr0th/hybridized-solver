/**
 * @file hybsol/ordering.h
 * Coloring-based reordering of the block degrees of freedom.
 *
 * The idea is to group blocks so that no two blocks of the same group share
 * a non-zero off-diagonal block ("coloring"), then order the unknowns group
 * by group. This concentrates the non-zero structure and makes the
 * factorization in :c:func:`hybsol_decomposition_factorize` cheaper.
 */

#ifndef HYBSOL_ORDERING_H
#define HYBSOL_ORDERING_H

#include <hybsol/block_system.h>
#include <hybsol/types.h>

/** How the coloring produced by :c:func:`hybsol_system_compute_reordering` is built. */
typedef enum hybsol_ordering_strategy
{
    /** Always pick the lowest-numbered color that is still available. */
    HYBSOL_ORDERING_FIRST = 0,
    /** Prefer the color used by the most blocks so far. */
    HYBSOL_ORDERING_GREEDY,
    /** Prefer the color used by the fewest blocks so far. */
    HYBSOL_ORDERING_BALANCED,
} hybsol_ordering_strategy_t;

/**
 * Compute a **coloring** of the blocks: entry ``i`` of ``out_ordering`` gets the
 * new index of old block ``i``, and the result is always a permutation of
 * ``[0, n_blocks)``. ``max_colors`` of ``0`` lets the library pick one (the
 * number of blocks in the densest row, which is always sufficient).
 *
 * This is **not** a factorization ordering. It is not valid input to
 * :c:func:`hybsol_system_reorder_blocks` when the goal is
 * :c:func:`hybsol_decomposition_factorize`: in an augmented system the
 * multiplier region is structurally zero, so it colors as a perfect independent
 * set and can be scheduled ahead of the blocks it depends on. Choose your own
 * permutation there, with every block that has a lower connection ahead of the
 * blocks it couples to.
 *
 * Preconditions: ``out_ordering`` non-``NULL``, ``strategy`` one of
 * :c:type:`hybsol_ordering_strategy_t`.
 *
 * Returns :c:enumerator:`HYBSOL_ERROR_MAX_COLORS` if ``max_colors`` was too
 * small, leaving ``out_ordering`` unspecified, or
 * :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY`.
 */
hybsol_result_t hybsol_system_compute_reordering(const hybsol_system_t *sys, hybsol_ordering_strategy_t strategy,
                                                 uint64_t max_colors, uint64_t *out_ordering);

/**
 * Reorder the blocks of the system according to a permutation.
 *
 * Row ``i`` moves to row ``new_order[i]``, column ``i`` to column
 * ``new_order[i]``, and the block sizes are permuted to match, so the system
 * stays structurally identical up to the relabelling.
 *
 * The permutation is applied before it is checked, and is not rolled back. What
 * the check rejects is the order, not the shuffle: on
 * :c:enumerator:`HYBSOL_ERROR_INVALID_ORDERING` the new order would have had to
 * factorize a block whose diagonal is identically zero, and ``*failing_block``
 * names it. ``failing_block`` may be ``NULL``.
 *
 * Preconditions: ``new_order`` is asserted to be a permutation of
 * ``[0, n_blocks)``, and ``sys`` is asserted structurally valid.
 */
hybsol_result_t hybsol_system_reorder_blocks(hybsol_system_t *sys, const uint64_t *new_order, uint64_t n_threads,
                                             uint64_t *failing_block);

/**
 * Apply a block ordering to a vector.
 *
 * Block ``i`` of ``in`` is written to the slot that
 * :c:func:`hybsol_system_reorder_blocks` moved it to, so the result is the
 * vector of the reordered system. ``sys`` must already have been reordered with
 * ``new_order``; the offsets it holds are used to locate both slices.
 *
 * ``new_order`` is ``n_blocks`` entries forming a permutation, ``in`` a vector
 * in the *old* ordering of length :c:func:`hybsol_system_total_size`, and
 * ``out`` a destination of the same length that must not alias ``in``.
 *
 * Preconditions: ``sys``, ``new_order``, ``in`` and ``out`` non-``NULL``.
 */
void hybsol_system_reorder_vector(const hybsol_system_t *sys, const uint64_t *new_order, const double *in, double *out);

/**
 * Undo :c:func:`hybsol_system_reorder_vector`.
 *
 * Block ``new_order[i]`` of ``in`` is written back to slot ``i``, restoring
 * the vector of the system as it was before the reordering.
 *
 * ``new_order`` is ``n_blocks`` entries forming a permutation, ``in`` a vector
 * in the *new* ordering of length :c:func:`hybsol_system_total_size`, and
 * ``out`` a destination of the same length that must not alias ``in``.
 *
 * Preconditions: ``sys``, ``new_order``, ``in`` and ``out`` non-``NULL``.
 */
void hybsol_system_unorder_vector(const hybsol_system_t *sys, const uint64_t *new_order, const double *in, double *out);

#endif /* HYBSOL_ORDERING_H */
