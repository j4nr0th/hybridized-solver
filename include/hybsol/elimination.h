/**
 * @file hybsol/elimination.h
 * The symbolic result of eliminating a system: the graph the numeric
 * factorization and the solve both walk.
 *
 * Computing it modifies nothing, and its result answers "how expensive will
 * this be, and will it work at all" before any memory for the factorization is
 * committed.
 */

#ifndef HYBSOL_ELIMINATION_H
#define HYBSOL_ELIMINATION_H

#include <hybsol/block_system.h>
#include <hybsol/types.h>

/**
 * The elimination graph of a system.
 *
 * The type is opaque: :c:func:`hybsol_elimination_create` builds one and
 * :c:func:`hybsol_elimination_destroy` releases it. It holds, for every block
 * row, the columns that row ends up holding once the fill-in is complete, the
 * passes at which the row is processed, and the exact number of bytes a
 * decomposition of the system will need.
 */
typedef struct hybsol_elimination hybsol_elimination_t;

/**
 * Walk ``sys``'s elimination graph without computing any values.
 *
 * The walk carries only column indices, so it costs a pass over the graph
 * rather than over the numbers: for a sparse system a constant-factor fraction
 * of the factorization rather than a separate order of growth.
 *
 * The result describes ``sys`` as it stands at the call. It is not a
 * reservation: assembling more blocks afterwards invalidates it.
 * :c:func:`hybsol_decomposition_create` asserts that the graph still describes
 * the system -- the same block count, precision, block offsets and sparsity
 * pattern -- so a stale graph is refused rather than read as written. Discard
 * the graph when the system changes anyway; the assertion is a backstop, not a
 * licence to keep one.
 *
 * Preconditions: ``sys`` satisfies :c:func:`hybsol_system_is_valid` (asserted,
 * so a caller wanting a diagnosis runs that first), ``out`` non-``NULL``.
 *
 * Returns :c:enumerator:`HYBSOL_ERROR_INVALID_ORDERING` or
 * :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY`; on
 * :c:enumerator:`HYBSOL_ERROR_INVALID_ORDERING`, ``*failing_block`` receives
 * the block whose diagonal is identically zero, and ``*out`` is ``NULL``.
 * ``failing_block`` may be ``NULL``; it is written on both success and failure.
 */
hybsol_result_t hybsol_elimination_create(const hybsol_system_t *sys, hybsol_elimination_t **out,
                                          uint64_t *failing_block);

/** Release a graph. ``NULL`` is allowed and does nothing. */
void hybsol_elimination_destroy(hybsol_elimination_t *graph);

/** Get ``n``, the number of blocks per dimension. */
uint64_t hybsol_elimination_n_blocks(const hybsol_elimination_t *graph);

/** Get the precision of the analyzed system. */
hybsol_precision_t hybsol_elimination_precision(const hybsol_elimination_t *graph);

/**
 * Get the total number of blocks the final pattern holds: the system's own
 * blocks plus every block the fill-in adds. Equals the sum of
 * :c:func:`hybsol_elimination_row_length` over all rows.
 */
uint64_t hybsol_elimination_n_columns(const hybsol_elimination_t *graph);

/**
 * Get the number of operations a decomposition records: one elimination per
 * (row, source) pair the walk found, plus one diagonal solve per block row.
 *
 * Exact, so an operation list sized to it never grows and never needs a lock to
 * append to.
 */
uint64_t hybsol_elimination_n_operations(const hybsol_elimination_t *graph);

/**
 * Get the number of passes the elimination takes, at least ``1``.
 *
 * Passes are level-synchronous: everything a pass does is independent of
 * everything else in it, and every pass depends only on passes before it. Rows
 * within a pass are listed by ascending index, so the order is deterministic.
 */
uint64_t hybsol_elimination_n_levels(const hybsol_elimination_t *graph);

/**
 * Get the bytes of block storage a decomposition of this system needs.
 *
 * Every block of the final pattern is carved contiguously and rounded up to the
 * library's alignment, so this is exact rather than an estimate.
 */
size_t hybsol_elimination_value_bytes(const hybsol_elimination_t *graph);

/** Get the bytes the whole graph occupies, its own allocation. */
size_t hybsol_elimination_total_bytes(const hybsol_elimination_t *graph);

/**
 * Get the block whose diagonal is identically zero, which no factorization of
 * the current block order can get past, or ``UINT64_MAX`` when the walk did not
 * reject the order.
 */
uint64_t hybsol_elimination_failing_block(const hybsol_elimination_t *graph);

/**
 * Get the number of blocks a block row holds once the fill-in is complete.
 *
 * Returns ``0`` for a ``row`` outside ``[0, n_blocks)``.
 */
uint64_t hybsol_elimination_row_length(const hybsol_elimination_t *graph, uint64_t row);

/**
 * Get how many eliminations a block row performs: row ``i`` is eliminated with
 * each of the columns it holds below its own diagonal, in ascending order, and
 * its diagonal is factorized once it has none left. ``0`` for a row that is
 * diagonal-first from the outset.
 */
uint64_t hybsol_elimination_row_n_eliminations(const hybsol_elimination_t *graph, uint64_t row);

/**
 * Get the first pass a block row is processed in, and
 * :c:func:`hybsol_elimination_row_level` the last. A row is processed in as
 * many passes as it has eliminations, or in one pass if it has none, but those
 * passes are *not* consecutive: a row sits out any pass in which the row it is
 * eliminated with has not finished yet.
 *
 * Preconditions: ``row`` in range.
 */
uint64_t hybsol_elimination_row_first_level(const hybsol_elimination_t *graph, uint64_t row);

/**
 * Get the last pass a block row is processed in: the pass in which it
 * factorizes its diagonal, having eliminated once in each of the passes it was
 * processed in before that. The last of those eliminations shares the pass with
 * the diagonal factorization.
 *
 * Preconditions: ``row`` in range.
 */
uint64_t hybsol_elimination_row_level(const hybsol_elimination_t *graph, uint64_t row);

/**
 * Get the column indices a block row holds once the fill-in is complete, in
 * increasing order.
 *
 * Preconditions: ``row`` in range, ``capacity >=`` the row's length, ``out``
 * non-``NULL``. ``n_written`` may be ``NULL``.
 */
hybsol_result_t hybsol_elimination_row_columns(const hybsol_elimination_t *graph, uint64_t row, uint64_t *out,
                                               uint64_t capacity, uint64_t *n_written);

/**
 * Get the number of block rows processed in a pass.
 *
 * Returns ``0`` for a ``level`` outside ``[0, n_levels)``.
 */
uint64_t hybsol_elimination_level_size(const hybsol_elimination_t *graph, uint64_t level);

/**
 * Get the block rows processed in a pass, in ascending index order.
 *
 * A row appears in exactly
 * :c:func:`hybsol_elimination_row_n_eliminations` passes, or in one pass if it
 * has none. Those are the passes from
 * :c:func:`hybsol_elimination_row_first_level` to
 * :c:func:`hybsol_elimination_row_level` with gaps, since a row waits for the
 * row it is eliminated with.
 *
 * Preconditions: ``level`` in range, ``capacity >=`` the pass's length, ``out``
 * non-``NULL``. ``n_written`` may be ``NULL``.
 */
hybsol_result_t hybsol_elimination_level_rows(const hybsol_elimination_t *graph, uint64_t level, uint64_t *out,
                                              uint64_t capacity, uint64_t *n_written);

#endif /* HYBSOL_ELIMINATION_H */
