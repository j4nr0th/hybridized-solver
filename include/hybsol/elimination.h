/**
 * @file hybsol/elimination.h
 * The symbolic result of eliminating a system: the graph the numeric
 * factorization and the solve both walk.
 *
 * Computing it touches no block value and modifies nothing, so it can be run
 * on a system that is still being assembled, and its result answers "how
 * expensive will this be, and will it work at all" before any memory for the
 * factorization is committed.
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
 * rather than over the numbers. For a sparse system that is a constant-factor
 * fraction of the factorization rather than a separate order of growth.
 *
 * The result describes ``sys`` as it stands at the call. It is not a
 * reservation: assembling more blocks afterwards invalidates it, and
 * :c:func:`hybsol_decomposition_create` refuses a graph that does not match
 * the system it is given.
 *
 * :param sys: The system to analyze. Must not be ``NULL``.
 * :param out: Receives the graph. Set to ``NULL`` on failure. Must not be ``NULL``.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`,
 *     :c:enumerator:`HYBSOL_ERROR_SYSTEM_INVALID` if ``sys`` does not
 *     satisfy the solver's structural assumptions,
 *     :c:enumerator:`HYBSOL_ERROR_INVALID_ORDERING` if its current block order
 *     would have to factorize an identically zero diagonal, or
 *     :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY`. For the ordering case
 *     :c:func:`hybsol_system_failing_block` names the block, since there is no
 *     graph to ask; the walk itself reads nothing it writes.
 */
hybsol_result_t hybsol_elimination_create(hybsol_system_t *sys, hybsol_elimination_t **out);

/**
 * Release a graph. ``NULL`` is allowed and does nothing.
 *
 * :param graph: The graph to release.
 */
void hybsol_elimination_destroy(hybsol_elimination_t *graph);

/**
 * Get the number of blocks per dimension.
 *
 * :param graph: The graph.
 * :returns: ``n`` in the sense of :c:func:`hybsol_system_create`.
 */
uint64_t hybsol_elimination_n_blocks(const hybsol_elimination_t *graph);

/**
 * Get the precision the analyzed system stores its blocks in.
 *
 * :param graph: The graph.
 * :returns: The precision ``sys`` was created with.
 */
hybsol_precision_t hybsol_elimination_precision(const hybsol_elimination_t *graph);

/**
 * Get the total number of blocks the graph's final pattern holds.
 *
 * That is the system's own blocks plus every block the fill-in adds.
 *
 * :param graph: The graph.
 * :returns: The sum of :c:func:`hybsol_elimination_row_length` over all rows.
 */
uint64_t hybsol_elimination_n_columns(const hybsol_elimination_t *graph);

/**
 * Get the number of operations a decomposition of this system records.
 *
 * One elimination per (row, source) pair the walk found, plus one diagonal
 * solve per block row. The count is exact, so an operation list sized to it
 * never grows and never needs a lock to append to.
 *
 * :param graph: The graph.
 * :returns: The number of operations :c:func:`hybsol_decomposition_operations`
 *     will write.
 */
uint64_t hybsol_elimination_n_operations(const hybsol_elimination_t *graph);

/**
 * Get the number of passes the elimination takes.
 *
 * Passes are level-synchronous: everything a pass does is independent of
 * everything else in it, and every pass depends only on passes before it. Rows
 * within a pass are listed by ascending index, so the order is deterministic.
 *
 * :param graph: The graph.
 * :returns: The number of passes; at least ``1``.
 */
uint64_t hybsol_elimination_n_levels(const hybsol_elimination_t *graph);

/**
 * Get the bytes of block storage a decomposition of this system needs.
 *
 * Every block of the final pattern is carved contiguously and rounded up to the
 * library's alignment, so this is exact rather than an estimate.
 *
 * :param graph: The graph.
 * :returns: The size of the destination's value arena, in bytes.
 */
size_t hybsol_elimination_value_bytes(const hybsol_elimination_t *graph);

/**
 * Get the bytes the whole graph occupies.
 *
 * :param graph: The graph.
 * :returns: The size of the graph's own allocation, in bytes.
 */
size_t hybsol_elimination_total_bytes(const hybsol_elimination_t *graph);

/**
 * Get the block whose diagonal could not be factorized.
 *
 * Set when the walk finds a block row that is diagonal-first but whose
 * diagonal block is identically zero, which no factorization of the current
 * block order can get past.
 *
 * :param graph: The graph.
 * :returns: The zero-based block index, or ``UINT64_MAX`` when the walk did not
 *     reject the order.
 */
uint64_t hybsol_elimination_failing_block(const hybsol_elimination_t *graph);

/**
 * Get the number of blocks a block row holds once the fill-in is complete.
 *
 * :param graph: The graph.
 * :param row: Block row index; asserted in range.
 * :returns: The length of :c:func:`hybsol_elimination_row_columns`; ``0`` if
 *     ``row`` is out of range.
 */
uint64_t hybsol_elimination_row_length(const hybsol_elimination_t *graph, uint64_t row);

/**
 * Get how many eliminations a block row performs.
 *
 * Row ``i`` is eliminated with each of the columns it holds below its own
 * diagonal, in ascending order, and its diagonal is factorized once it has none
 * left.
 *
 * :param graph: The graph.
 * :param row: Block row index; asserted in range.
 * :returns: The number of eliminations; ``0`` for a row that is diagonal-first
 *     from the outset.
 */
uint64_t hybsol_elimination_row_n_eliminations(const hybsol_elimination_t *graph, uint64_t row);

/**
 * Get the first pass a block row is processed in.
 *
 * A row is processed once per pass from this one until
 * :c:func:`hybsol_elimination_row_level`, eliminating in all but the last.
 *
 * :param graph: The graph.
 * :param row: Block row index; asserted in range.
 * :returns: The first pass, in ``[0, hybsol_elimination_n_levels)``.
 */
uint64_t hybsol_elimination_row_first_level(const hybsol_elimination_t *graph, uint64_t row);

/**
 * Get the last pass a block row is processed in.
 *
 * In this pass the row factorizes its diagonal; every earlier pass it spends
 * eliminating.
 *
 * :param graph: The graph.
 * :param row: Block row index; asserted in range.
 * :returns: The last pass, in ``[0, hybsol_elimination_n_levels)``.
 */
uint64_t hybsol_elimination_row_level(const hybsol_elimination_t *graph, uint64_t row);

/**
 * Get the column indices a block row holds once the fill-in is complete.
 *
 * The indices come back in increasing order. A block the system already stores
 * is present, and so is every block the fill-in adds.
 *
 * :param graph: The graph.
 * :param row: Block row index; asserted in range.
 * :param out: Destination for ``capacity`` indices. Must not be ``NULL``.
 * :param capacity: Number of indices ``out`` can hold; asserted large enough
 *     for the row.
 * :param n_written: Receives the number of indices the row holds. May be ``NULL``.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`.
 */
hybsol_result_t hybsol_elimination_row_columns(const hybsol_elimination_t *graph, uint64_t row, uint64_t *out,
                                               uint64_t capacity, uint64_t *n_written);

/**
 * Get the number of block rows processed in a pass.
 *
 * :param graph: The graph.
 * :param level: Pass index; asserted less than :c:func:`hybsol_elimination_n_levels`.
 * :returns: The length of :c:func:`hybsol_elimination_level_rows`.
 */
uint64_t hybsol_elimination_level_size(const hybsol_elimination_t *graph, uint64_t level);

/**
 * Get the block rows processed in a pass.
 *
 * The rows come back in ascending index order, and a row appears in every pass
 * from :c:func:`hybsol_elimination_row_first_level` to
 * :c:func:`hybsol_elimination_row_level`.
 *
 * :param graph: The graph.
 * :param level: Pass index; asserted less than :c:func:`hybsol_elimination_n_levels`.
 * :param out: Destination for ``capacity`` indices. Must not be ``NULL``.
 * :param capacity: Number of indices ``out`` can hold; asserted large enough
 *     for the pass.
 * :param n_written: Receives the number of indices the pass holds. May be ``NULL``.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`.
 */
hybsol_result_t hybsol_elimination_level_rows(const hybsol_elimination_t *graph, uint64_t level, uint64_t *out,
                                              uint64_t capacity, uint64_t *n_written);

#endif /* HYBSOL_ELIMINATION_H */
