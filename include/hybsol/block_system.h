/**
 * @file hybsol/block_system.h
 * The block-structured linear system the solver operates on.
 *
 * A system is a square ``n x n`` grid of dense blocks laid out on an
 * underlying ``size x size`` matrix, ``size`` being the sum of the block
 * sizes. Only the sparsity pattern is stored, sorted by column within each
 * row.
 *
 * A system is pure storage: nothing here consumes it. A factorization reads it
 * and writes into a :c:type:`hybsol_decomposition_t` of its own, so the same
 * system can be decomposed repeatedly, under different block orders.
 */

#ifndef HYBSOL_BLOCK_SYSTEM_H
#define HYBSOL_BLOCK_SYSTEM_H

#include <hybsol/matrix.h>
#include <hybsol/types.h>

/**
 * A block-structured sparse linear system.
 *
 * The type is opaque: :c:func:`hybsol_system_create` builds one and
 * :c:func:`hybsol_system_destroy` releases it.
 */
typedef struct hybsol_system hybsol_system_t;

/**
 * Create an empty double-precision system with the given block sizes.
 *
 * The result has no blocks at all; fill it with
 * :c:func:`hybsol_system_add_block` or :c:func:`hybsol_system_add_blocks`.
 *
 * Preconditions: ``n_blocks >= 1``, every ``block_sizes[i] > 0``, ``out`` and
 * ``allocator`` non-``NULL``. Pass ``&CUTL_STD_ALLOCATOR`` for plain
 * ``malloc``/``realloc``/``free``; any other allocator must stay valid until
 * :c:func:`hybsol_system_destroy` returns.
 *
 * Returns :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY` on failure, leaving
 * ``*out`` ``NULL``.
 */
hybsol_result_t hybsol_system_create(uint64_t n_blocks, HYBSOL_IN(uint64_t, block_sizes, n_blocks),
                                     hybsol_system_t **out, const hybsol_allocator_t *allocator);

/**
 * Create an empty system storing its blocks in a chosen type.
 *
 * ``HYBSOL_PRECISION_SINGLE`` halves the memory and, where FP32 is faster,
 * the time of the factorization, at the cost of a forward error of roughly
 * ``cond * 1e-7`` instead of ``cond * eps``. Vectors are unaffected: the
 * solve and ordering functions keep taking doubles and convert at the
 * boundary.
 *
 * Same preconditions as :c:func:`hybsol_system_create`, plus ``precision``
 * being one of :c:type:`hybsol_precision_t`.
 */
hybsol_result_t hybsol_system_create_with_precision(uint64_t n_blocks, HYBSOL_IN(uint64_t, block_sizes, n_blocks),
                                                    hybsol_precision_t precision, hybsol_system_t **out,
                                                    const hybsol_allocator_t *allocator);

/** Report how ``sys`` stores its blocks. */
hybsol_precision_t hybsol_system_precision(const hybsol_system_t *sys);

/**
 * Release a system and all memory it owns. ``NULL`` is allowed and does
 * nothing.
 *
 * :c:type:`hybsol_decomposition_t` values built from it stay valid: they hold
 * their own copy of every block.
 */
void hybsol_system_destroy(hybsol_system_t *sys);

/**
 * Deep-copy a system's blocks and pattern.
 *
 * Returns :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY` on failure, leaving
 * ``*out`` ``NULL``.
 */
hybsol_result_t hybsol_system_copy(const hybsol_system_t *sys, hybsol_system_t **out);

/** Get ``n``, the number of blocks per dimension. */
uint64_t hybsol_system_n_blocks(const hybsol_system_t *sys);

/** Get the sum of all block sizes, i.e. the dimension of the matrix. */
uint64_t hybsol_system_total_size(const hybsol_system_t *sys);

/**
 * Get the number of rows of block ``idx``; block ``(idx, j)`` has this many
 * rows and ``block_size(j)`` columns.
 *
 * Preconditions: ``idx`` is asserted in ``[0, n)``.
 */
uint64_t hybsol_system_block_size(const hybsol_system_t *sys, uint64_t idx);

/**
 * Get the block offsets of the underlying matrix.
 *
 * ``n_blocks + 1`` entries, starting at ``0``, strictly increasing, ending at
 * :c:func:`hybsol_system_total_size`. The array is owned by ``sys`` and stays
 * valid until it is destroyed or reordered.
 */
const uint64_t *hybsol_system_block_offsets(const hybsol_system_t *sys);

/**
 * Check that the system satisfies the solver's structural assumptions: every
 * row must contain its diagonal block, and every block below the diagonal
 * must be mirrored above it. Rows may otherwise be arbitrarily sparse.
 *
 * This is the check :c:func:`hybsol_elimination_create` asserts, so a caller
 * that wants a diagnosis rather than an abort runs it first.
 *
 * Returns ``1`` if valid, ``0`` otherwise.
 */
int hybsol_system_is_valid(const hybsol_system_t *sys);

/** Get the number of stored blocks in block row ``row``; ``row`` is asserted in range. */
uint64_t hybsol_system_row_count(const hybsol_system_t *sys, uint64_t row);

/**
 * Get the column indices of the blocks stored in a row, increasing as stored.
 *
 * Preconditions: ``row`` in range, ``capacity >=`` the row's length, ``out``
 * non-``NULL``. ``n_written`` may be ``NULL``.
 */
hybsol_result_t hybsol_system_row_indices(const hybsol_system_t *sys, uint64_t row, uint64_t *out, uint64_t capacity,
                                          uint64_t *n_written);

/** Returns ``1`` if block ``(row, col)`` is stored, ``0`` if not or out of range. */
int hybsol_system_has_block(const hybsol_system_t *sys, uint64_t row, uint64_t col);

/**
 * Get a view of a stored block.
 *
 * The view points into the system and is only valid until it is modified or
 * destroyed. Writing through it is allowed but must not change the block's
 * shape.
 *
 * Preconditions: ``row`` and ``col`` in range, ``(row, col)`` stored, ``sys``
 * double-precision, ``out`` non-``NULL``.
 */
hybsol_result_t hybsol_system_get_block(hybsol_system_t *sys, uint64_t row, uint64_t col, hybsol_matrix_t *out);

/**
 * The single-precision spelling of :c:func:`hybsol_system_get_block`: ``sys``
 * must store floats, and the view is a :c:type:`hybsol_fmatrix_t`. Hand it a
 * double system and it asserts rather than widening the block silently.
 */
hybsol_result_t hybsol_system_get_block_f32(hybsol_system_t *sys, uint64_t row, uint64_t col, hybsol_fmatrix_t *out);

/**
 * Find the smallest column index stored in a row. ``row`` is asserted in
 * range, ``out`` may be ``NULL``.
 *
 * Returns :c:enumerator:`HYBSOL_ERROR_EMPTY_ROW` if the row stores nothing.
 */
hybsol_result_t hybsol_system_first_column(const hybsol_system_t *sys, uint64_t row, uint64_t *out);

/**
 * Find the smallest column index in a row strictly greater than ``col``.
 *
 * Returns :c:enumerator:`HYBSOL_ERROR_EMPTY_ROW` if the row stores nothing,
 * or :c:enumerator:`HYBSOL_ERROR_NO_MORE_COLUMNS` once the scan runs off the
 * end.
 */
hybsol_result_t hybsol_system_next_column(const hybsol_system_t *sys, uint64_t row, uint64_t col, uint64_t *out);

/**
 * Report which block rows have nothing stored to the left of their diagonal:
 * ``out[i]`` is ``1`` when row ``i``'s first stored column is at or beyond the
 * diagonal. ``out`` must have ``n_blocks`` entries.
 */
void hybsol_system_no_lower_connections(const hybsol_system_t *sys, uint8_t *out);

/**
 * Reserve room for ``capacity`` blocks in a row.
 *
 * Assembly loops that know how many blocks each row will receive can call this
 * once per row instead of growing the row repeatedly.
 *
 * Returns :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY` on failure.
 */
hybsol_result_t hybsol_system_reserve(hybsol_system_t *sys, uint64_t row, uint64_t capacity);

/**
 * Add or accumulate into a single block.
 *
 * A block already present has the values summed into it; otherwise a new entry
 * is inserted at the correct position in the row.
 *
 * Preconditions: ``row`` and ``col`` in range, ``n_rows``/``n_cols`` equal to
 * the sizes of those blocks, ``vals`` non-``NULL``.
 *
 * Returns :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY` on failure.
 */
hybsol_result_t hybsol_system_add_block(hybsol_system_t *sys, uint64_t row, uint64_t col, uint64_t n_rows,
                                        uint64_t n_cols, const double *vals);

/** The ``_f32`` spelling of :c:func:`hybsol_system_add_block`; ``sys`` must store ``float``. */
hybsol_result_t hybsol_system_add_block_f32(hybsol_system_t *sys, uint64_t row, uint64_t col, uint64_t n_rows,
                                            uint64_t n_cols, const float *vals);

/**
 * Add many blocks in a single pass.
 *
 * Block ``k`` occupies ``block_size(rows[k]) * block_size(cols[k])``
 * consecutive elements of ``data``, in that order. Duplicate ``(row, col)``
 * pairs accumulate, exactly as repeated :c:func:`hybsol_system_add_block`
 * calls would. The rows are counted first, so each one's pointer array is grown
 * a single time.
 *
 * Preconditions: every index in ``rows`` and ``cols`` in range, ``data``
 * non-``NULL`` unless ``n_entries`` is ``0``.
 *
 * Returns :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY` on failure.
 */
hybsol_result_t hybsol_system_add_blocks(hybsol_system_t *sys, uint64_t n_entries, HYBSOL_IN(uint64_t, rows, n_entries),
                                         HYBSOL_IN(uint64_t, cols, n_entries), const double *data);

/** The ``_f32`` spelling of :c:func:`hybsol_system_add_blocks`; ``sys`` must store ``float``. */
hybsol_result_t hybsol_system_add_blocks_f32(hybsol_system_t *sys, uint64_t n_entries,
                                             HYBSOL_IN(uint64_t, rows, n_entries), HYBSOL_IN(uint64_t, cols, n_entries),
                                             const float *data);

/**
 * Get writable storage for a block, creating it zero-filled on first use.
 *
 * The shape is implied by the system, so the caller fills the returned view
 * directly with no temporary. A present block is returned as it stands, never
 * cleared and never accumulated, so repeated calls hand back the same buffer.
 *
 * The pointer outlives any number of calls that only add blocks: a stored block
 * is its own allocation and adding to a row moves only the row's pointer array.
 * It also survives :c:func:`hybsol_system_reorder_blocks`, which relabels and
 * re-sorts the same entries rather than rebuilding them.
 *
 * It does not survive :c:func:`hybsol_system_eliminate_row_with`, which rewrites
 * the target row and allocates new entries for the fill-in, nor
 * :c:func:`hybsol_system_destroy`.
 *
 * Preconditions: ``row`` and ``col`` in range, ``out`` non-``NULL``.
 */
hybsol_result_t hybsol_system_block_storage(hybsol_system_t *sys, uint64_t row, uint64_t col, hybsol_matrix_t *out);

/** The ``_f32`` spelling of :c:func:`hybsol_system_block_storage`; ``sys`` must store ``float``. */
hybsol_result_t hybsol_system_block_storage_f32(hybsol_system_t *sys, uint64_t row, uint64_t col,
                                                hybsol_fmatrix_t *out);

/**
 * Multiply a whole block row from the left: every stored block at or after
 * ``start_col`` is replaced by ``mat @ block``. Blocks before ``start_col``
 * are untouched, which lets a caller skip columns already eliminated.
 *
 * Preconditions: ``row`` in range, ``mat`` asserted ``size(row)`` square.
 */
hybsol_result_t hybsol_system_multiply_row(hybsol_system_t *sys, uint64_t row, uint64_t start_col,
                                           const hybsol_matrix_t *mat);

/** The ``_f32`` spelling of :c:func:`hybsol_system_multiply_row`; ``sys`` must store ``float``. */
hybsol_result_t hybsol_system_multiply_row_f32(hybsol_system_t *sys, uint64_t row, uint64_t start_col,
                                               const hybsol_fmatrix_t *mat);

/**
 * Eliminate a block row using another one: ``row_tgt := row_tgt - mat @ row_src``.
 *
 * Entries of either row with a column index at or below ``row_src`` are assumed
 * already eliminated and are left alone. One of the two rows must hold a block
 * in column ``row_src``, or there is nothing to eliminate with; a later
 * allocation failure empties the target row rather than leaving it
 * half-updated.
 *
 * Preconditions: ``row_tgt`` and ``row_src`` in range, ``mat`` non-``NULL`` and
 * asserted ``size(row_tgt) x size(row_src)``.
 */
hybsol_result_t hybsol_system_eliminate_row_with(hybsol_system_t *sys, uint64_t row_tgt, uint64_t row_src,
                                                 const hybsol_matrix_t *mat);

/** The ``_f32`` spelling of :c:func:`hybsol_system_eliminate_row_with`; ``sys`` must store ``float``. */
hybsol_result_t hybsol_system_eliminate_row_with_f32(hybsol_system_t *sys, uint64_t row_tgt, uint64_t row_src,
                                                     const hybsol_fmatrix_t *mat);

/**
 * Eliminate a block row using the block stored at ``(row_tgt, row_src)``, i.e.
 * :c:func:`hybsol_system_eliminate_row_with` with the multiplier taken from
 * the system itself.
 */
hybsol_result_t hybsol_system_eliminate_row(hybsol_system_t *sys, uint64_t row_tgt, uint64_t row_src);

/**
 * Apply the system to a single vector, ``y = A x``.
 *
 * This is the operator the elimination solves against, and the one worth having
 * fast: it walks the stored blocks rather than a dense form, so a sparse system
 * costs the sum of its blocks rather than the square of its dimension. The
 * factorizations' correctness rests on being able to measure ``rhs - A x``
 * against the system, which is what this is for.
 *
 * Both vectors are ``size x 1``. ``x`` may be narrower than the system's
 * precision and is widened as it is read; ``y`` is written, not accumulated
 * into, and must not be ``x``.
 *
 * The work is split across block rows, each of which writes only its own slice
 * of ``y``, so the system is only ever read and a caller may keep using it
 * meanwhile.
 */
hybsol_result_t hybsol_system_matvec(const hybsol_system_t *sys, const hybsol_matrix_t *x, hybsol_matrix_t *y,
                                     uint64_t n_threads);

/**
 * Apply the system to several right-hand sides, ``y = A x``.
 *
 * The same walk as :c:func:`hybsol_system_matvec`, done for every column of
 * ``x`` at once, so the pattern is traversed once rather than once per column.
 * With one column the two are identical. ``x`` and ``y`` are
 * ``total_size x k`` row-major, ``y`` is written rather than accumulated into,
 * and must not overlap ``x``.
 */
hybsol_result_t hybsol_system_matmat(const hybsol_system_t *sys, const hybsol_matrix_t *x, hybsol_matrix_t *y,
                                     uint64_t n_threads);

/**
 * Write the system as a dense ``total_size x total_size`` matrix, missing
 * blocks as zeros. ``out`` must not be ``NULL``.
 */
hybsol_result_t hybsol_system_to_dense(const hybsol_system_t *sys, double *out);

/** The ``_f32`` spelling of :c:func:`hybsol_system_to_dense`, writing floats; ``sys`` must store ``float``. */
hybsol_result_t hybsol_system_to_dense_f32(const hybsol_system_t *sys, float *out);

#endif /* HYBSOL_BLOCK_SYSTEM_H */
