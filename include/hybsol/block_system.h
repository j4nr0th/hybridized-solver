/**
 * @file hybsol/block_system.h
 * The block-structured linear system the solver operates on.
 *
 * A system is a square ``n x n`` grid of dense blocks laid out on an
 * underlying ``size x size`` matrix, ``size`` being the sum of the block
 * sizes. Only the sparsity pattern is stored, sorted by column within each
 * row.
 *
 * The pattern must be symmetric below the diagonal and every row must contain
 * its diagonal block; :c:func:`hybsol_system_is_valid` checks this.
 */

#ifndef HYBSOL_BLOCK_SYSTEM_H
#define HYBSOL_BLOCK_SYSTEM_H

#include <hybsol/matrix.h>
#include <hybsol/types.h>

/**
 * A block-structured sparse linear system.
 *
 * The type is opaque: use :c:func:`hybsol_system_create` to obtain an
 * instance and :c:func:`hybsol_system_destroy` to release it.
 */
typedef struct hybsol_system hybsol_system_t;

/**
 * Create an empty system with the given block sizes.
 *
 * The resulting system has no blocks at all; populate it with
 * :c:func:`hybsol_system_add_block` or :c:func:`hybsol_system_add_blocks`.
 *
 * :param n_blocks: Number of blocks per dimension; asserted at least 1.
 * :param block_sizes: Array of ``n_blocks`` sizes; asserted non-zero.
 * :param out: Receives the new system. Must not be ``NULL``.
 * :param allocator: Allocator every allocation of this system goes through.
 *     Must not be ``NULL``; pass ``&CUTL_STD_ALLOCATOR`` for plain
 *     ``malloc``/``realloc``/``free``. The allocator must stay valid until
 *     :c:func:`hybsol_system_destroy` returns.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS` or
 *     :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY`.
 */
hybsol_result_t hybsol_system_create(uint64_t n_blocks, HYBSOL_IN(uint64_t, block_sizes, n_blocks),
                                     hybsol_system_t **out, const hybsol_allocator_t *allocator);

/**
 * Create an empty system that stores its blocks in a chosen type.
 *
 * ``HYBSOL_PRECISION_SINGLE`` halves the memory and, where FP32 is faster,
 * the time of the factorization, at the cost of a forward error of roughly
 * ``cond * 1e-7`` instead of ``cond * eps``. Vectors are unaffected: the
 * solve and ordering functions keep taking doubles and convert at the
 * boundary.
 *
 * Precision cannot be changed afterwards, and every function carrying values
 * has a matching spelling; using the wrong one asserts rather than converting
 * behind the caller's back.
 *
 * :param n_blocks: Number of blocks per dimension; asserted at least 1.
 * :param block_sizes: Array of ``n_blocks`` sizes; asserted non-zero.
 * :param precision: Storage type, asserted one of :c:type:`hybsol_precision_t`.
 * :param out: Receives the new system. Must not be ``NULL``.
 * :param allocator: Allocator every allocation of this system goes through.
 *     Must not be ``NULL``; pass ``&CUTL_STD_ALLOCATOR`` for plain
 *     ``malloc``/``realloc``/``free``. The allocator must stay valid until
 *     :c:func:`hybsol_system_destroy` returns.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS` or
 *     :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY`.
 */
hybsol_result_t hybsol_system_create_with_precision(uint64_t n_blocks, HYBSOL_IN(uint64_t, block_sizes, n_blocks),
                                                    hybsol_precision_t precision, hybsol_system_t **out,
                                                    const hybsol_allocator_t *allocator);

/**
 * Report how a system stores its blocks.
 *
 * :param sys: The system.
 * :returns: The precision chosen when ``sys`` was created.
 */
hybsol_precision_t hybsol_system_precision(const hybsol_system_t *sys);

/**
 * Release a system and all memory it owns.
 *
 * :param sys: The system to destroy; ``NULL`` is allowed and does nothing.
 */
void hybsol_system_destroy(hybsol_system_t *sys);

/**
 * Deep-copy a system, including its decomposition if it has one.
 *
 * :param sys: The system to copy.
 * :param out: Receives the copy on success. Set to ``NULL`` on failure.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS` or
 *     :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY`.
 */
hybsol_result_t hybsol_system_copy(const hybsol_system_t *sys, hybsol_system_t **out);

/**
 * Get the number of blocks per dimension.
 *
 * :param sys: The system.
 * :returns: ``n`` in the sense of the constructor.
 */
uint64_t hybsol_system_n_blocks(const hybsol_system_t *sys);

/**
 * Get the total number of rows (and columns) of the underlying matrix.
 *
 * :param sys: The system.
 * :returns: The sum of all block sizes.
 */
uint64_t hybsol_system_total_size(const hybsol_system_t *sys);

/**
 * Get the size of a block.
 *
 * Block ``(row, col)`` has this many rows and ``block_size(col)`` columns.
 *
 * :param sys: The system.
 * :param idx: Block index.
 * :returns: The number of rows of block ``idx``; ``0`` if ``idx`` is out of
 *     range.
 */
uint64_t hybsol_system_block_size(const hybsol_system_t *sys, uint64_t idx);

/**
 * Get the block offsets of the underlying matrix.
 *
 * The returned array has ``n_blocks + 1`` entries, starts at ``0``, is
 * strictly increasing and ends at :c:func:`hybsol_system_total_size`. It
 * stays valid until the system is destroyed or reordered.
 *
 * :param sys: The system.
 * :returns: The offset array owned by ``sys``.
 */
const uint64_t *hybsol_system_block_offsets(const hybsol_system_t *sys);

/**
 * Check that the system satisfies the solver's structural assumptions.
 *
 * Every row must contain its diagonal block, and every block below the
 * diagonal must be mirrored above it. Rows may otherwise be arbitrarily
 * sparse.
 *
 * :param sys: The system.
 * :returns: ``1`` if valid, ``0`` otherwise.
 */
int hybsol_system_is_valid(const hybsol_system_t *sys);

/**
 * Get the number of stored blocks in a row.
 *
 * :param sys: The system.
 * :param row: Block row index.
 * :returns: The number of stored blocks; ``0`` if ``row`` is out of range.
 */
uint64_t hybsol_system_row_count(const hybsol_system_t *sys, uint64_t row);

/**
 * Get the column indices of the blocks stored in a row.
 *
 * The indices come back in increasing order, as stored.
 *
 * :param sys: The system.
 * :param row: Block row index; asserted in range.
 * :param out: Destination for ``capacity`` indices. Must not be ``NULL``.
 * :param capacity: Number of indices ``out`` can hold; asserted large enough
 *     for the row.
 * :param n_written: Receives the number of indices the row holds. May be
 *     ``NULL``.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`.
 */
hybsol_result_t hybsol_system_row_indices(const hybsol_system_t *sys, uint64_t row, uint64_t *out, uint64_t capacity,
                                          uint64_t *n_written);

/**
 * Check whether a block is present.
 *
 * :param sys: The system.
 * :param row: Block row index.
 * :param col: Block column index.
 * :returns: ``1`` if the block is stored, ``0`` if not (or if an index is
 *     out of range).
 */
int hybsol_system_has_block(const hybsol_system_t *sys, uint64_t row, uint64_t col);

/**
 * Get a view of a stored block.
 *
 * The returned view points into the system and is only valid until the
 * system is modified or destroyed. Writing through it is allowed but must
 * not change the block's shape.
 *
 * :param sys: The system.
 * :param row: Block row index; asserted in range.
 * :param col: Block column index; asserted in range and present in the row.
 * :param out: Receives the view. Must not be ``NULL``.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`.
 */
hybsol_result_t hybsol_system_get_block(hybsol_system_t *sys, uint64_t row, uint64_t col, hybsol_matrix_t *out);

/**
 * The single-precision spelling of :c:func:`hybsol_system_get_block`.
 *
 * Hand it a system that stores doubles and it asserts rather than widening
 * the blocks silently; :c:func:`hybsol_system_get_block` is its double twin.
 *
 * :param sys: The system.
 * :param row: Block row index; asserted in range.
 * :param col: Block column index; asserted in range and present in the row.
 * :param out: Receives the view. Must not be ``NULL``.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`.
 */
hybsol_result_t hybsol_system_get_block_f32(hybsol_system_t *sys, uint64_t row, uint64_t col, hybsol_fmatrix_t *out);

/**
 * Find the first column index stored in a row.
 *
 * :param sys: The system.
 * :param row: Block row index.
 * :param out: Receives the smallest stored column index of the row.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`, or
 *     :c:enumerator:`HYBSOL_ERROR_EMPTY_ROW` if the row stores no blocks.
 */
hybsol_result_t hybsol_system_first_column(const hybsol_system_t *sys, uint64_t row, uint64_t *out);

/**
 * Find the first column index in a row that is strictly greater than ``col``.
 *
 * :param sys: The system.
 * :param row: Block row index.
 * :param col: Column index to start looking after.
 * :param out: Receives the next stored column index.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`,
 *     :c:enumerator:`HYBSOL_ERROR_EMPTY_ROW` or
 *     :c:enumerator:`HYBSOL_ERROR_NO_MORE_COLUMNS`.
 */
hybsol_result_t hybsol_system_next_column(const hybsol_system_t *sys, uint64_t row, uint64_t col, uint64_t *out);

/**
 * Report which rows have no blocks to the left of their diagonal.
 *
 * :param sys: The system.
 * :param out: Destination for ``n_blocks`` flags; ``1`` means the row's
 *     first stored column is at or beyond the diagonal.
 */
void hybsol_system_no_lower_connections(const hybsol_system_t *sys, uint8_t *out);

/**
 * Reserve space for a number of blocks in a row.
 *
 * Assembly loops that know how many blocks each row will receive can call
 * this once per row to avoid repeated reallocation.
 *
 * :param sys: The system.
 * :param row: Block row index; asserted in range.
 * :param capacity: Number of blocks the row should be able to hold.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`,
 *     :c:enumerator:`HYBSOL_ERROR_ALREADY_DECOMPOSED` or
 *     :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY`.
 */
hybsol_result_t hybsol_system_reserve(hybsol_system_t *sys, uint64_t row, uint64_t capacity);

/**
 * Add (or accumulate into) a single block.
 *
 * If the block is already present the values are summed into it, otherwise a
 * new entry is inserted at the correct position in the row.
 *
 * :param sys: The system.
 * :param row: Block row index; asserted in range.
 * :param col: Block column index; asserted in range.
 * :param n_rows: Number of rows of ``vals``; asserted equal to the size of ``row``.
 * :param n_cols: Number of columns of ``vals``; asserted equal to the size of ``col``.
 * :param vals: Row-major values, read but not modified. Must not be ``NULL``.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`,
 *     :c:enumerator:`HYBSOL_ERROR_ALREADY_DECOMPOSED` or
 *     :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY`.
 */
hybsol_result_t hybsol_system_add_block(hybsol_system_t *sys, uint64_t row, uint64_t col, uint64_t n_rows,
                                        uint64_t n_cols, const double *vals);

/**
 * The single-precision spelling of :c:func:`hybsol_system_add_block`, taking
 * ``float`` values and requiring a system created with
 * :c:enumerator:`HYBSOL_PRECISION_SINGLE`.
 *
 * :param sys: The system; asserted to store ``float``.
 * :param row: Block row index; asserted in range.
 * :param col: Block column index; asserted in range.
 * :param n_rows: Number of rows of ``vals``; asserted equal to the size of ``row``.
 * :param n_cols: Number of columns of ``vals``; asserted equal to the size of ``col``.
 * :param vals: Row-major values, read but not modified. Must not be ``NULL``.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`,
 *     :c:enumerator:`HYBSOL_ERROR_ALREADY_DECOMPOSED` or
 *     :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY`.
 */
hybsol_result_t hybsol_system_add_block_f32(hybsol_system_t *sys, uint64_t row, uint64_t col, uint64_t n_rows,
                                            uint64_t n_cols, const float *vals);

/**
 * Add many blocks in a single pass.
 *
 * Block ``k`` occupies ``block_size(rows[k]) * block_size(cols[k])``
 * consecutive doubles of ``data``, in that order. Duplicate ``(row, col)``
 * pairs accumulate, exactly as repeated calls to
 * :c:func:`hybsol_system_add_block` would. The index arrays are grouped and
 * sorted once, so each row allocates its storage a single time.
 *
 * :param sys: The system.
 * :param n_entries: Number of blocks being added.
 * :param rows: Block row indices, ``n_entries`` entries; asserted in range.
 * :param cols: Block column indices, ``n_entries`` entries; asserted in range.
 * :param data: Row-major block values concatenated in the order implied by
 *     ``rows`` and ``cols``. Must not be ``NULL``.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`,
 *     :c:enumerator:`HYBSOL_ERROR_ALREADY_DECOMPOSED` or
 *     :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY`.
 */
hybsol_result_t hybsol_system_add_blocks(hybsol_system_t *sys, uint64_t n_entries, HYBSOL_IN(uint64_t, rows, n_entries),
                                         HYBSOL_IN(uint64_t, cols, n_entries), const double *data);

/**
 * The single-precision spelling of :c:func:`hybsol_system_add_blocks`.
 *
 * :param sys: The system; asserted to store ``float``.
 * :param n_entries: Number of blocks to add.
 * :param rows: Block row indices; asserted in range.
 * :param cols: Block column indices; asserted in range.
 * :param data: Concatenated, row-major block values. Must not be ``NULL``.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`,
 *     :c:enumerator:`HYBSOL_ERROR_ALREADY_DECOMPOSED` or
 *     :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY`.
 */
hybsol_result_t hybsol_system_add_blocks_f32(hybsol_system_t *sys, uint64_t n_entries,
                                             HYBSOL_IN(uint64_t, rows, n_entries), HYBSOL_IN(uint64_t, cols, n_entries),
                                             const float *data);

/**
 * Get writable storage for a block, creating it on first use.
 *
 * The shape is implied by the system, so the caller gets a view of exactly the
 * right size and fills it directly — no temporary buffer. An absent block is
 * inserted into the pattern and zero-filled, so a partial scatter never sees
 * stale bytes; a present one is returned as it stands, never cleared and never
 * accumulated, so repeated calls hand back the same buffer.
 *
 * The pointer stays valid while other blocks are added, since each block owns
 * its allocation, but not across :c:func:`hybsol_system_reorder_blocks` or
 * :c:func:`hybsol_system_decompose`, which rebuild the rows. Writing the
 * diagonal block invalidates that row's cached factorization.
 *
 * :param sys: The system.
 * :param row: Block row index; asserted in range.
 * :param col: Block column index; asserted in range.
 * :param out: Receives the view. Must not be ``NULL``.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`,
 *     :c:enumerator:`HYBSOL_ERROR_ALREADY_DECOMPOSED` or
 *     :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY`.
 */
hybsol_result_t hybsol_system_block_storage(hybsol_system_t *sys, uint64_t row, uint64_t col, hybsol_matrix_t *out);

/**
 * The ``_f32`` spelling of :c:func:`hybsol_system_block_storage`; everything
 * said there holds, with the buffer type and the required precision differing.
 *
 * :param sys: The system; asserted to store ``float``.
 * :param row: Block row index; asserted in range.
 * :param col: Block column index; asserted in range.
 * :param out: Receives the view. Must not be ``NULL``.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`,
 *     :c:enumerator:`HYBSOL_ERROR_ALREADY_DECOMPOSED` or
 *     :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY`.
 */
hybsol_result_t hybsol_system_block_storage_f32(hybsol_system_t *sys, uint64_t row, uint64_t col,
                                                hybsol_fmatrix_t *out);

/**
 * Multiply a whole block row from the left by a square matrix.
 *
 * Every stored block in the row at or after ``start_col`` is replaced by
 * ``mat @ block``. Blocks before ``start_col`` are left untouched, which
 * lets a caller skip columns that have already been eliminated.
 *
 * :param sys: The system.
 * :param row: Block row index; asserted in range.
 * :param start_col: First block column (inclusive) to transform.
 * :param mat: Square matrix asserted to be ``size(row)`` by ``size(row)``.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`,
 *     :c:enumerator:`HYBSOL_ERROR_ALREADY_DECOMPOSED` or
 *     :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY`.
 */
hybsol_result_t hybsol_system_multiply_row(hybsol_system_t *sys, uint64_t row, uint64_t start_col,
                                           const hybsol_matrix_t *mat);

/**
 * The single-precision spelling of :c:func:`hybsol_system_multiply_row`.
 *
 * :param sys: The system; asserted to store ``float``.
 * :param row: Block row index; asserted in range.
 * :param start_col: First block column to touch.
 * :param mat: Square multiplier asserted to be ``block_size(row)`` square.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`,
 *     :c:enumerator:`HYBSOL_ERROR_ALREADY_DECOMPOSED` or
 *     :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY`.
 */
hybsol_result_t hybsol_system_multiply_row_f32(hybsol_system_t *sys, uint64_t row, uint64_t start_col,
                                               const hybsol_fmatrix_t *mat);

/**
 * Eliminate a block row using another one, with an explicit multiplier.
 *
 * Computes ``row_tgt := row_tgt - mat @ row_src``. Entries of either row
 * with a column index at or below ``row_src`` are assumed already eliminated
 * and are left alone.
 *
 * The target row must contain its block in column ``row_src``. If a later
 * allocation fails the target row is emptied rather than left half-updated.
 *
 * :param sys: The system.
 * :param row_tgt: Row to update; asserted to hold column ``row_src``.
 * :param row_src: Row to eliminate with.
 * :param mat: Multiplier asserted to be ``size(row_tgt) x size(row_src)``.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`,
 *     :c:enumerator:`HYBSOL_ERROR_EMPTY_ROW`,
 *     :c:enumerator:`HYBSOL_ERROR_ALREADY_DECOMPOSED` or
 *     :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY`.
 */
hybsol_result_t hybsol_system_eliminate_row_with(hybsol_system_t *sys, uint64_t row_tgt, uint64_t row_src,
                                                 const hybsol_matrix_t *mat);

/**
 * The single-precision spelling of :c:func:`hybsol_system_eliminate_row_with`.
 *
 * :param sys: The system; asserted to store ``float``.
 * :param row_tgt: Row to update; asserted to hold column ``row_src``.
 * :param row_src: Row to eliminate with.
 * :param mat: Multiplier asserted to be ``size(row_tgt) x size(row_src)``.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`,
 *     :c:enumerator:`HYBSOL_ERROR_EMPTY_ROW`,
 *     :c:enumerator:`HYBSOL_ERROR_ALREADY_DECOMPOSED` or
 *     :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY`.
 */
hybsol_result_t hybsol_system_eliminate_row_with_f32(hybsol_system_t *sys, uint64_t row_tgt, uint64_t row_src,
                                                     const hybsol_fmatrix_t *mat);

/**
 * Eliminate a block row using the block stored at ``(row_tgt, row_src)``.
 *
 * This is :c:func:`hybsol_system_eliminate_row_with` with the multiplier
 * taken from the system itself, which is what the decomposition needs.
 *
 * :param sys: The system.
 * :param row_tgt: Row to update; asserted to hold column ``row_src``.
 * :param row_src: Row to eliminate with.
 * :returns: The result codes of :c:func:`hybsol_system_eliminate_row_with`.
 */
hybsol_result_t hybsol_system_eliminate_row(hybsol_system_t *sys, uint64_t row_tgt, uint64_t row_src);

/**
 * Write the system as a dense matrix.
 *
 * Missing blocks are written as zeros.
 *
 * :param sys: The system.
 * :param out: Destination for ``total_size * total_size`` row-major
 *     doubles. Must not be ``NULL``.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`.
 */
hybsol_result_t hybsol_system_to_dense(const hybsol_system_t *sys, double *out);

/**
 * The single-precision spelling of :c:func:`hybsol_system_to_dense`, writing
 * ``total * total`` floats.
 *
 * :param sys: The system; asserted to store ``float``.
 * :param out: Row-major destination of at least ``total * total`` floats.
 *     Must not be ``NULL``.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`.
 */
hybsol_result_t hybsol_system_to_dense_f32(const hybsol_system_t *sys, float *out);

#endif /* HYBSOL_BLOCK_SYSTEM_H */
