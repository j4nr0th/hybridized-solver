/**
 * @file hybsol/block_system.h
 * The block-structured linear system the solver operates on.
 *
 * A system is a square ``n x n`` grid of dense blocks laid out on an
 * underlying ``size x size`` matrix (``size`` being the sum of the block
 * sizes). Only the blocks of the sparsity pattern are stored, sorted by
 * column within each row.
 *
 * The pattern must be symmetric below the diagonal and every row must
 * contain its diagonal block; :c:func:`hybsol_system_is_valid` checks this
 * and :c:func:`hybsol_system_decompose` refuses to run on an invalid system.
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
 * :param n_blocks: Number of blocks per dimension; must be at least 1.
 * :param block_sizes: Array of ``n_blocks`` strictly positive sizes.
 * :param out: Receives the new system on success. Set to ``NULL`` on failure.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`,
 *     :c:enumerator:`HYBSOL_ERROR_INVALID_ARGUMENT` or
 *     :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY`.
 */
hybsol_result_t hybsol_system_create(uint64_t n_blocks, HYBSOL_IN(uint64_t, block_sizes, n_blocks),
                                     hybsol_system_t **out);

/**
 * Create an empty system that stores its blocks in a chosen type.
 *
 * Everything about the system follows from this: the blocks it is filled with,
 * the factors :c:func:`hybsol_system_decompose` produces from them and every
 * operation that touches them use ``precision``. Choosing
 * :c:enumerator:`HYBSOL_PRECISION_SINGLE` halves the memory and (on hardware
 * where FP32 is faster) the time of the factorization, at the cost of a
 * forward error of roughly ``cond * 1e-7`` instead of ``cond * eps``.
 *
 * Vectors are unaffected: :c:func:`hybsol_system_solve`,
 * :c:func:`hybsol_system_reorder_vector` and the ordering helpers keep taking
 * doubles for both precisions, converting at the boundary.
 *
 * Precision cannot be changed afterwards, and every function that carries
 * values has a matching spelling — the unsuffixed one for a
 * :c:enumerator:`HYBSOL_PRECISION_DOUBLE` system and ``_f32`` for a
 * :c:enumerator:`HYBSOL_PRECISION_SINGLE` one. Using the wrong spelling is
 * rejected with :c:enumerator:`HYBSOL_ERROR_INVALID_ARGUMENT` rather than
 * converted behind the caller's back.
 *
 * :param n_blocks: Number of blocks per dimension; must be at least 1.
 * :param block_sizes: Array of ``n_blocks`` strictly positive sizes.
 * :param precision: Storage type, one of :c:type:`hybsol_precision_t`.
 * :param out: Receives the new system on success. Set to ``NULL`` on failure.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`,
 *     :c:enumerator:`HYBSOL_ERROR_INVALID_ARGUMENT` or
 *     :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY`.
 */
hybsol_result_t hybsol_system_create_with_precision(uint64_t n_blocks, HYBSOL_IN(uint64_t, block_sizes, n_blocks),
                                                    hybsol_precision_t precision, hybsol_system_t **out);

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
 * Every block ``(row, col)`` has this many rows and
 * :c:func:`hybsol_system_block_size` ``(sys, col)`` columns, so the matrix
 * of the whole system is square with side
 * :c:func:`hybsol_system_total_size`.
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
 * :param row: Block row index.
 * :param out: Destination for ``capacity`` indices; nothing is written if
 *     ``capacity`` is too small.
 * :param capacity: Number of indices ``out`` can hold.
 * :param n_written: Receives the number of indices the row holds, which may
 *     be larger than ``capacity``.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`,
 *     :c:enumerator:`HYBSOL_ERROR_INDEX_OUT_OF_RANGE` or
 *     :c:enumerator:`HYBSOL_ERROR_INVALID_ARGUMENT` if ``out`` is ``NULL``
 *     or ``capacity`` was too small for the row.
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
 * :param row: Block row index.
 * :param col: Block column index.
 * :param out: Receives the view.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`, or
 *     :c:enumerator:`HYBSOL_ERROR_BLOCK_NOT_IN_SYSTEM` if the block is absent.
 */
hybsol_result_t hybsol_system_get_block(hybsol_system_t *sys, uint64_t row, uint64_t col, hybsol_matrix_t *out);

/**
 * The single-precision spelling of :c:func:`hybsol_system_get_block`.
 *
 * Hand it a system that stores doubles and it returns
 * :c:enumerator:`HYBSOL_ERROR_INVALID_ARGUMENT` rather than widening the
 * blocks silently; :c:func:`hybsol_system_get_block` is its double twin.
 *
 * :param sys: The system.
 * :param row: Block row index.
 * :param col: Block column index.
 * :param out: Receives the copy on success.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`,
 *     :c:enumerator:`HYBSOL_ERROR_INVALID_ARGUMENT`,
 *     :c:enumerator:`HYBSOL_ERROR_INDEX_OUT_OF_RANGE` or
 *     :c:enumerator:`HYBSOL_ERROR_BLOCK_NOT_IN_SYSTEM`.
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
 * :param row: Block row index.
 * :param capacity: Number of blocks the row should be able to hold.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`,
 *     :c:enumerator:`HYBSOL_ERROR_INDEX_OUT_OF_RANGE`,
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
 * :param row: Block row index.
 * :param col: Block column index.
 * :param n_rows: Number of rows of ``vals``; must equal the size of ``row``.
 * :param n_cols: Number of columns of ``vals``; must equal the size of ``col``.
 * :param vals: Row-major values, read but not modified.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`,
 *     :c:enumerator:`HYBSOL_ERROR_INDEX_OUT_OF_RANGE`,
 *     :c:enumerator:`HYBSOL_ERROR_INVALID_ARGUMENT` (wrong shape),
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
 * :param sys: The system.
 * :param row: Block row index.
 * :param col: Block column index.
 * :param n_rows: Number of rows of ``vals``; must equal the size of ``row``.
 * :param n_cols: Number of columns of ``vals``; must equal the size of ``col``.
 * :param vals: Row-major values, read but not modified.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`,
 *     :c:enumerator:`HYBSOL_ERROR_INDEX_OUT_OF_RANGE`,
 *     :c:enumerator:`HYBSOL_ERROR_INVALID_ARGUMENT` (wrong shape or wrong
 *     precision), :c:enumerator:`HYBSOL_ERROR_ALREADY_DECOMPOSED` or
 *     :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY`.
 */
hybsol_result_t hybsol_system_add_block_f32(hybsol_system_t *sys, uint64_t row, uint64_t col, uint64_t n_rows,
                                            uint64_t n_cols, const float *vals);

/**
 * Add many blocks in a single pass.
 *
 * Block ``k`` occupies ``block_size(rows[k]) * block_size(cols[k])``
 * consecutive doubles of ``data``, in that order. Duplicate
 * ``(row, col)`` pairs accumulate, exactly as repeated calls to
 * :c:func:`hybsol_system_add_block` would.
 *
 * This is the fast assembly path: the index arrays are grouped and sorted
 * once, and each row allocates its storage a single time.
 *
 * :param sys: The system.
 * :param n_entries: Number of blocks being added.
 * :param rows: Block row indices, ``n_entries`` entries.
 * :param cols: Block column indices, ``n_entries`` entries.
 * :param data: Row-major block values concatenated in the order implied by
 *     ``rows`` and ``cols``.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`,
 *     :c:enumerator:`HYBSOL_ERROR_INDEX_OUT_OF_RANGE`,
 *     :c:enumerator:`HYBSOL_ERROR_INVALID_ARGUMENT`,
 *     :c:enumerator:`HYBSOL_ERROR_ALREADY_DECOMPOSED` or
 *     :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY`.
 */
hybsol_result_t hybsol_system_add_blocks(hybsol_system_t *sys, uint64_t n_entries, HYBSOL_IN(uint64_t, rows, n_entries),
                                         HYBSOL_IN(uint64_t, cols, n_entries), const double *data);

/**
 * The single-precision spelling of :c:func:`hybsol_system_add_blocks`.
 *
 * :param sys: The system.
 * :param n_entries: Number of blocks to add.
 * :param rows: Block row indices.
 * :param cols: Block column indices.
 * :param data: Concatenated, row-major block values.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`,
 *     :c:enumerator:`HYBSOL_ERROR_INDEX_OUT_OF_RANGE`,
 *     :c:enumerator:`HYBSOL_ERROR_INVALID_ARGUMENT` (wrong precision),
 *     :c:enumerator:`HYBSOL_ERROR_ALREADY_DECOMPOSED` or
 *     :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY`.
 */
hybsol_result_t hybsol_system_add_blocks_f32(hybsol_system_t *sys, uint64_t n_entries,
                                             HYBSOL_IN(uint64_t, rows, n_entries), HYBSOL_IN(uint64_t, cols, n_entries),
                                             const float *data);

/**
 * Get writable storage for a block, creating it on first use.
 *
 * The shape of the block is implied by the system, so unlike
 * :c:func:`hybsol_system_add_block` there is nothing to specify: the caller
 * is handed a view of exactly the right size and fills it directly. This is
 * meant for assemblers that would otherwise have to build every element
 * matrix or constraint block in a temporary buffer only to have it copied in.
 *
 * If the block is absent it is inserted into the row's sparsity pattern and
 * zero-filled, so a partial scatter never observes stale heap bytes. If the
 * block is already present it is returned exactly as it stands; the function
 * never clears or accumulates, and repeated calls hand back the same buffer.
 *
 * Writing through the view is allowed but must not change its shape. The
 * pointer stays valid while other blocks are added to the system, since each
 * block owns its own allocation, but it does not survive
 * :c:func:`hybsol_system_reorder_blocks` or :c:func:`hybsol_system_decompose`,
 * both of which rebuild the rows.
 *
 * As with :c:func:`hybsol_system_add_block`, writing the diagonal block
 * invalidates that row's cached factorization.
 *
 * :param sys: The system.
 * :param row: Block row index.
 * :param col: Block column index.
 * :param out: Receives the view on success. Set to ``NULL`` on failure.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`,
 *     :c:enumerator:`HYBSOL_ERROR_INVALID_ARGUMENT` (``out`` is ``NULL``),
 *     :c:enumerator:`HYBSOL_ERROR_INDEX_OUT_OF_RANGE`,
 *     :c:enumerator:`HYBSOL_ERROR_ALREADY_DECOMPOSED` or
 *     :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY`.
 */
hybsol_result_t hybsol_system_block_storage(hybsol_system_t *sys, uint64_t row, uint64_t col, hybsol_matrix_t *out);

/**
 * The single-precision spelling of :c:func:`hybsol_system_block_storage`.
 *
 * Everything said there holds: the block is created zero-filled on first use
 * and handed back untouched afterwards, and the view stays valid until the
 * rows are rebuilt. The difference is the type of the buffer handed back and
 * the precision the system has to store.
 *
 * :param sys: The system.
 * :param row: Block row index.
 * :param col: Block column index.
 * :param out: Receives the view on success. Set to ``NULL`` on failure.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`,
 *     :c:enumerator:`HYBSOL_ERROR_INVALID_ARGUMENT` (``out`` is ``NULL`` or
 *     the system stores doubles),
 *     :c:enumerator:`HYBSOL_ERROR_INDEX_OUT_OF_RANGE`,
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
 * :param row: Block row index.
 * :param start_col: First block column (inclusive) to transform.
 * :param mat: Square matrix of the same size as ``row``.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`,
 *     :c:enumerator:`HYBSOL_ERROR_INDEX_OUT_OF_RANGE`,
 *     :c:enumerator:`HYBSOL_ERROR_INVALID_ARGUMENT` (``mat`` not square and
 *     of the right size), :c:enumerator:`HYBSOL_ERROR_ALREADY_DECOMPOSED` or
 *     :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY`.
 */
hybsol_result_t hybsol_system_multiply_row(hybsol_system_t *sys, uint64_t row, uint64_t start_col,
                                           const hybsol_matrix_t *mat);

/**
 * The single-precision spelling of :c:func:`hybsol_system_multiply_row`.
 *
 * :param sys: The system.
 * :param row: Block row index.
 * :param start_col: First block column to touch.
 * :param mat: Square multiplier, ``block_size(row)`` by ``block_size(row)``.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`,
 *     :c:enumerator:`HYBSOL_ERROR_INDEX_OUT_OF_RANGE`,
 *     :c:enumerator:`HYBSOL_ERROR_INVALID_ARGUMENT` (wrong shape or wrong
 *     precision), :c:enumerator:`HYBSOL_ERROR_ALREADY_DECOMPOSED` or
 *     :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY`.
 */
hybsol_result_t hybsol_system_multiply_row_f32(hybsol_system_t *sys, uint64_t row, uint64_t start_col,
                                               const hybsol_fmatrix_t *mat);

/**
 * Eliminate a block row using another one, with an explicit multiplier.
 *
 * Computes ``row_tgt := row_tgt - mat @ row_src``. Entries of either row
 * with a column index at or below ``row_src`` are assumed to have been
 * eliminated already and are left alone.
 *
 * Both rows must contain their block in column ``row_src``, and the
 * resulting row must be able to hold the union of the two sparsity
 * patterns; if a later allocation fails the target row is emptied rather
 * than left half-updated.
 *
 * :param sys: The system.
 * :param row_tgt: Row to update.
 * :param row_src: Row to eliminate with.
 * :param mat: Multiplier, ``size(row_tgt) x size(row_src)``.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`,
 *     :c:enumerator:`HYBSOL_ERROR_INDEX_OUT_OF_RANGE`,
 *     :c:enumerator:`HYBSOL_ERROR_BLOCK_NOT_IN_SYSTEM`,
 *     :c:enumerator:`HYBSOL_ERROR_INVALID_ARGUMENT`,
 *     :c:enumerator:`HYBSOL_ERROR_ALREADY_DECOMPOSED` or
 *     :c:enumerator:`HYBSOL_ERROR_OUT_OF_MEMORY`.
 */
hybsol_result_t hybsol_system_eliminate_row_with(hybsol_system_t *sys, uint64_t row_tgt, uint64_t row_src,
                                                 const hybsol_matrix_t *mat);

/**
 * The single-precision spelling of :c:func:`hybsol_system_eliminate_row_with`.
 *
 * :param sys: The system.
 * :param row_tgt: Row to update.
 * :param row_src: Row to eliminate with.
 * :param mat: Square multiplier, ``block_size(row_tgt)`` by ``block_size(row_src)``.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`,
 *     :c:enumerator:`HYBSOL_ERROR_INDEX_OUT_OF_RANGE`,
 *     :c:enumerator:`HYBSOL_ERROR_INVALID_ARGUMENT` (wrong shape or wrong
 *     precision), :c:enumerator:`HYBSOL_ERROR_ALREADY_DECOMPOSED`,
 *     :c:enumerator:`HYBSOL_ERROR_EMPTY_ROW`,
 *     :c:enumerator:`HYBSOL_ERROR_BLOCK_NOT_IN_SYSTEM` or
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
 * :param row_tgt: Row to update.
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
 * :param out: Destination for ``total_size * total_size`` row-major doubles.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS`.
 */
hybsol_result_t hybsol_system_to_dense(const hybsol_system_t *sys, double *out);

/**
 * The single-precision spelling of :c:func:`hybsol_system_to_dense`, writing
 * ``total * total`` floats.
 *
 * :param sys: The system.
 * :param out: Row-major destination of at least ``total * total`` floats.
 * :returns: :c:enumerator:`HYBSOL_SUCCESS` or
 *     :c:enumerator:`HYBSOL_ERROR_INVALID_ARGUMENT`.
 */
hybsol_result_t hybsol_system_to_dense_f32(const hybsol_system_t *sys, float *out);

#endif /* HYBSOL_BLOCK_SYSTEM_H */
