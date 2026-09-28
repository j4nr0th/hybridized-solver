/**
 * @file hybsol/ordering.h
 * Coloring-based reordering of the block degrees of freedom.
 *
 * The idea is to group blocks so that no two blocks of the same group share
 * a non-zero off-diagonal block ("coloring"), then order the unknowns group
 * by group. This concentrates the non-zero structure and makes the
 * elimination in :c:func:`hybsol_system_decompose` cheaper.
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
 * Compute a coloring-based reordering of the blocks.
 *
 * :param sys: The system to analyze; it is not modified.
 * :param strategy: Which coloring strategy to use.
 * :param max_colors: Upper bound on the number of colors, or ``0`` to let
 *     the library pick one (the number of blocks in the densest row, which
 *     is always sufficient).
 * :param out_ordering: Destination for ``n_blocks`` entries. Entry ``i``
 *     holds the new index of old block ``i``; the result is always a
 *     permutation of ``[0, n_blocks)``.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS` if a valid ordering was written,
 *     :c:enumerator:`HYBSOL_ERROR_MAX_COLORS` if ``max_colors`` was too
 *     small (``out_ordering`` is then left unspecified),
 *     :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY` or
 *     :c:enumerator:`HYBSOL_ERROR_INVALID_ARGUMENT` if ``out_ordering`` is
 *     too small because ``max_colors`` was ``0`` on an empty system.
 */
hybsol_result_t hybsol_system_compute_reordering(const hybsol_system_t *sys, hybsol_ordering_strategy_t strategy,
                                                 uint64_t max_colors, uint64_t *out_ordering);

/**
 * Reorder the blocks of the system according to a permutation.
 *
 * Row ``i`` moves to row ``new_order[i]``, column ``i`` to column
 * ``new_order[i]``, and block sizes are permuted to match, so the system
 * stays structurally identical up to the relabelling.
 *
 * :param sys: The system to reorder.
 * :param new_order: ``n_blocks`` entries forming a permutation; entry ``i``
 *     is the new index of old block ``i``.
 * :param n_threads: Number of OpenMP threads; ``0`` selects the OpenMP
 *     default and ``1`` runs serially.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`,
 *     :c:enumerator:`HYBSOL_ERROR_INVALID_ARGUMENT` (duplicate index),
 *     :c:enumerator:`HYBSOL_ERROR_ALREADY_DECOMPOSED` or
 *     :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY`.
 */
hybsol_result_t hybsol_system_reorder_blocks(hybsol_system_t *sys, const uint64_t *new_order, uint64_t n_threads);

/**
 * Apply a block ordering to a vector.
 *
 * Produces the vector that corresponds to the reordered system: the slice
 * of ``in`` belonging to block ``i`` is written to the slice of ``out``
 * belonging to block ``new_order[i]``.
 *
 * :param sys: The system the ordering was computed for.
 * :param new_order: ``n_blocks`` entries forming a permutation.
 * :param in: Input vector of length :c:func:`hybsol_system_total_size`.
 * :param out: Destination of the same length; must not alias ``in``.
 */
void hybsol_system_reorder_vector(const hybsol_system_t *sys, const uint64_t *new_order, const double *in, double *out);

/**
 * Undo :c:func:`hybsol_system_reorder_vector`.
 *
 * :param sys: The system the ordering was computed for.
 * :param new_order: ``n_blocks`` entries forming a permutation.
 * :param in: Input vector of length :c:func:`hybsol_system_total_size`.
 * :param out: Destination of the same length; must not alias ``in``.
 */
void hybsol_system_unorder_vector(const hybsol_system_t *sys, const uint64_t *new_order, const double *in, double *out);

#endif /* HYBSOL_ORDERING_H */
