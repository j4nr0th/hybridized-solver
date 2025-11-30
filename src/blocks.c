#include "blocks.h"

int block_system_add_block(const block_system_t *this, const u64 idx_row, const u64 idx_col, const u64 nv,
                           const f64 vals[static nv])
{
    CPYUTL_ASSERT(idx_row < this->n && idx_col < this->n,
                  "Row and/or column index was out of bounds (indices: (%zu, %zu) but the limits are [0, %u)).",
                  idx_row, idx_col, this->n);
    CPYUTL_ASSERT(nv == block_system_get_block_size(this, idx_row) * block_system_get_block_size(this, idx_col),
                  "Number of values does not match block size (expected %zu x %zu, got %zu).",
                  block_system_get_block_size(this, idx_row), block_system_get_block_size(this, idx_col), nv);

    sys_row_t *const row = this->rows + idx_row;

    // Find where to insert it
    if (row->count == row->capacity)
    {
        const u64 new_capacity = row->capacity ? row->capacity * 2 : 8;
        row_entry_t **const ptr = (row_entry_t **)PyMem_RawRealloc(row->entries, new_capacity * sizeof(*row->entries));
        if (!ptr)
            return -1;
        // Zero init
        for (u64 i = row->capacity; i < new_capacity; ++i)
        {
            ptr[i] = NULL;
        }
        row->entries = ptr;
        row->capacity = new_capacity;
    }

    const u64 idx_insert =
        row->count ? row_array_find_first_geq(row->count, (const row_entry_t *const *)row->entries, idx_col) : 0;
    if (idx_insert != row->count)
    {
        row_entry_t *const current_entry = row->entries[idx_insert];
        if (current_entry->col == idx_col)
        {
            // Entry matches the column index, just merge them
            for (u64 i = 0; i < nv; ++i)
            {
                current_entry->vals[i] += vals[i];
            }
            // We are done!
            return 0;
        }
    }

    row_entry_t *const new_entry = (row_entry_t *)PyMem_RawMalloc(sizeof(*new_entry) + nv * sizeof(*new_entry->vals));
    if (!new_entry)
        return -1;
    new_entry->col = idx_col;
    for (u64 i = 0; i < nv; ++i)
        new_entry->vals[i] = vals[i];

    // Entry does not match the column index, move it
    if (row->count != idx_insert)
    {
        CPYUTL_ASSERT(row->entries[idx_insert]->col > idx_col, "Binary search was WRONG: %zu is not GEQ than %zu!.",
                      idx_col, row->entries[idx_insert]->col);
        memmove(row->entries + idx_insert + 1, row->entries + idx_insert,
                (row->count - idx_insert) * sizeof(*row->entries));
    }
    else
    {
        CPYUTL_ASSERT(
            row->count == 0 || row->entries[row->count - 1]->col < idx_col,
            "The last entry is not bigger despite the search function claming so (last one was %zu, but we had %zu).",
            row->entries[row->count - 1]->col, idx_col);
    }

    row->entries[idx_insert] = new_entry;
    row->count += 1;

    return 0;
}

int block_system_is_valid(const block_system_t *this)
{
    for (u64 i = 0; i < this->n; ++i)
    {
        const sys_row_t *const row = this->rows + i;
        for (u64 j = 0; j < row->count; ++j)
        {
            const row_entry_t *const entry = row->entries[j];
            if (entry->col == i)
            {
                // The diagonal entry is there, so the row is done
                break;
            }

            if (entry->col > i)
            {
                // There was no diagonal entry!
                return 0;
            }

            // Check if the matrix has a symmetric block
            const sys_row_t *const other_row = this->rows + entry->col;
            const u64 other_idx =
                row_array_find_first_geq(other_row->count, (const row_entry_t *const *)other_row->entries, i);
            // Is the entry valid and does it match?
            if (other_idx == other_row->count || other_row->entries[other_idx]->col != i)
            {
                return 0;
            }
        }
    }

    return 1;
}

u64 row_array_find_first_geq(const u64 size, const row_entry_t *const array[static size], const u64 val)
{
    if (size == 0)
        return 0;

    // Quickly check the extremes
    if (array[0]->col >= val)
        return 0;

    if (array[size - 1]->col < val)
        return size;

    // Use binary search until we are down to 8 values
    u64 lo = 0, len = size;

    while (len > 8)
    {
        const u64 pivot = lo + len / 2;
        const u64 col = array[pivot]->col;
        if (col == val)
            return pivot;

        if (col > val)
        {
            len = pivot - lo + 1;
        }
        else
        {
            len = len - (pivot - lo);
            lo = pivot;
        }
    }

    CPYUTL_ASSERT(lo + len <= size, "Internal error: lo + len > size (%zu + %zu <= %zu).", lo, len, size);
    for (unsigned idx = lo; idx < lo + len; ++idx)
    {
        if (array[idx]->col >= val)
        {
            return idx;
        }
    }

    return size;
}

result_t block_system_get_block(const block_system_t *this, const u64 idx_row, const u64 idx_col, matrix_t *out)
{
    CPYUTL_ASSERT(idx_row < this->n && idx_col < this->n,
                  "Row and/or column index was out of bounds (indices: (%zu, %zu) but the limits are [0, %u)).",
                  idx_row, idx_col, this->n);
    const sys_row_t *const row = this->rows + idx_row;
    const u64 idx = row_array_find_first_geq(row->count, (const row_entry_t *const *)row->entries, idx_col);
    if (idx == row->count || row->entries[idx]->col != idx_col)
    {
        // No block
        return RESULT_BLOCK_NOT_IN_SYSTEM;
    }
    *out = (matrix_t){
        .rows = block_system_get_block_size(this, idx_row),
        .cols = block_system_get_block_size(this, idx_col),
        .data = row->entries[idx]->vals,
    };

    return RESULT_SUCCESS;
}
PyArrayObject *matrix_to_array(const matrix_t *mat)
{
    const npy_intp dims[2] = {(npy_intp)mat->rows, (npy_intp)mat->cols};
    PyArrayObject *const arr = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!arr)
        return NULL;
    memcpy(PyArray_DATA(arr), mat->data, mat->rows * mat->cols * sizeof(*mat->data));
    return arr;
}
matrix_t matrix_from_array(const PyArrayObject *arr)
{
    const int ndim = PyArray_NDIM(arr);
    CPYUTL_ASSERT(ndim <= 2, "Expected a 1D or 2D array, got a %dD array.", ndim);
    CPYUTL_ASSERT(check_input_array(arr, 0, (const npy_intp[]){}, NPY_DOUBLE,
                                    NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, "matrix") == 0,
                  "Array did not have correct dtype and/or flags!");
    return (matrix_t){
        .rows = (u64)PyArray_DIM(arr, 0),
        .cols = ndim == 2 ? (u64)PyArray_DIM(arr, 1) : 1,
        .data = PyArray_DATA(arr),
    };
}

void matrix_multiply(const matrix_t *a, const matrix_t *b, const matrix_t *out)
{
    CPYUTL_ASSERT(a->cols == b->rows,
                  "Matrix multiplication: number of columns of A (%zu) does not match number of rows of B (%zu).",
                  a->cols, b->rows);
    CPYUTL_ASSERT(out->rows == a->rows && out->cols == b->cols, "Output matrix has incorrect dimensions.");
    const f64 *const restrict ptr_a = a->data;
    const f64 *const restrict ptr_b = b->data;
    f64 *const restrict ptr_out = out->data;
    for (u64 i = 0; i < a->rows; ++i)
    {
#pragma omp simd
        for (u64 j = 0; j < b->cols; ++j)
        {
            f64 v = 0;

            for (u64 k = 0; k < a->cols; ++k)
                v += ptr_a[i * a->cols + k] * ptr_b[k * b->cols + j];

            ptr_out[i * out->cols + j] = v;
        }
    }
}

void matrix_multiply_sub_inplace(const matrix_t *a, const matrix_t *b, const matrix_t *out)
{
    CPYUTL_ASSERT(a->cols == b->rows, "Columns of A must match the rows of B (%zu vs %zu).", a->cols, b->rows);
    CPYUTL_ASSERT(out->rows == a->rows && out->cols == b->cols, "Output matrix has incorrect dimensions.");
    // Multiply compute A @ B and subtract the result from whatever is in OUT
    const f64 *const restrict ptr_a = a->data;
    const f64 *const restrict ptr_b = b->data;
    f64 *const restrict ptr_out = out->data;
    for (u64 i = 0; i < a->rows; ++i)
    {
#pragma omp simd
        for (u64 j = 0; j < b->cols; ++j)
        {
            f64 v = 0;
            for (u64 k = 0; k < a->cols; ++k)
                v += ptr_a[i * a->cols + k] * ptr_b[k * b->cols + j];

            ptr_out[i * out->cols + j] -= v;
        }
    }
}

void matrix_subtract_inplace(const matrix_t *a, const matrix_t *b)
{
    CPYUTL_ASSERT(a->rows == b->rows && a->cols == b->cols, "Matrices have different dimensions.");
#pragma omp simd
    for (u64 i = 0; i < a->rows * a->cols; ++i)
        a->data[i] -= b->data[i];
}

void matrix_lu_decompose(const matrix_t *m)
{
    CPYUTL_ASSERT(m->rows == m->cols, "Matrix must be square.");
    for (u64 i = 0; i < m->rows; ++i)
    {
        // Compute a row of U
        for (u64 j = i; j < m->rows; ++j)
        {
            f64 u_ij = m->data[i * m->cols + j];
#pragma omp simd
            for (u64 k = 0; k < i; ++k)
            {
                u_ij -= m->data[i * m->cols + k] * m->data[k * m->cols + j];
            }
            m->data[i * m->cols + j] = u_ij;
        }

        // Compute a column of L (the first element is always implicitly 1, so skip it
        for (u64 j = i + 1; j < m->cols; ++j)
        {
            f64 l_ij = m->data[j * m->cols + i];
#pragma omp simd
            for (u64 k = 0; k < i; ++k)
            {
                l_ij -= m->data[k * m->cols + i] * m->data[j * m->cols + k];
            }
            m->data[j * m->cols + i] = l_ij / m->data[i * m->cols + i];
        }
    }
}

void matrix_lu_solve(const matrix_t *m, const matrix_t *b, const matrix_t *out)
{
    CPYUTL_ASSERT(m->rows == m->cols && m->rows == b->rows, "Matrix must be square and have the same dimensions as b.");
    CPYUTL_ASSERT(b->rows == out->rows && b->cols == out->cols, "Output matrix has incorrect dimensions.");
    // Have to deal with every column in the same way
    for (u64 i_col = 0; i_col < b->cols; ++i_col)
    {
        // Forward substitution to solve the L part (diagonal is 1)
        for (u64 i = 0; i < m->rows; ++i)
        {
            f64 v = b->data[i * b->cols + i_col];
#pragma omp simd
            for (u64 j = 0; j < i; ++j)
            {
                v -= m->data[i * m->cols + j] * out->data[j * out->cols + i_col];
            }
            out->data[i * out->cols + i_col] = v;
        }

        // Backward substitution to solve the U part
        for (u64 i = m->rows; i > 0; --i)
        {
            f64 v = out->data[(i - 1) * out->cols + i_col];
#pragma omp simd
            for (u64 j = i; j < m->rows; ++j)
            {
                v -= m->data[(i - 1) * m->cols + j] * out->data[j * out->cols + i_col];
            }
            out->data[(i - 1) * out->cols + i_col] = v / m->data[(i - 1) * m->cols + (i - 1)];
        }
    }
}

static result_t operations_list_append(u64 *const p_count, u64 *const p_capacity, operation_t **const pp_ops,
                                       const operation_t op)
{
    result_t res = RESULT_SUCCESS;
#pragma omp critical(operations_list_append)
    {
        if (*p_count == *p_capacity)
        {
            const u64 new_capacity = *p_capacity ? *p_capacity * 2 : 8;
            operation_t *const ptr = (operation_t *)PyMem_RawRealloc(*pp_ops, new_capacity * sizeof(**pp_ops));
            if (!ptr)
            {
                res = RESULT_FAILED_ALLOC;
            }
            else
            {
                *pp_ops = ptr;
                *p_capacity = new_capacity;
            }
        }
        if (res == RESULT_SUCCESS)
        {
            (*pp_ops)[*p_count] = op;
            *p_count += 1;
        }
    }
    return res;
}

typedef enum
{
    TARGET_FREE,
    TARGET_IN_USE,
    TARGET_DONE,
} target_status_t;

typedef struct
{
    target_status_t status;
    u64 idx_src_needed;
} target_row_t;

result_t block_system_decompose(const block_system_t *this, u64 *pn_ops, operation_t **pp_ops, const uint n_threads)
{
    u64 count = 0;
    u64 capacity = 0;
    operation_t *ops = NULL;
    result_t shared_res = RESULT_SUCCESS;

    target_row_t *const target_status = (target_row_t *)PyMem_RawMalloc(this->n * sizeof(*target_status));
    if (!target_status)
    {
        return RESULT_FAILED_ALLOC;
    }

    // Check which rows start with the diagonal block
    u64 n_tgt = 0;
#pragma omp parallel for schedule(dynamic) reduction(+ : n_tgt) default(none)                                          \
    shared(this, target_status, shared_res, count, capacity, ops)
    for (u64 i_row = 0; i_row < this->n; ++i_row)
    {
        if (shared_res != RESULT_SUCCESS)
            continue;

        const sys_row_t *const row = this->rows + i_row;
        const u8 is_available = (row->entries[0]->col == i_row);
        if (is_available)
        {
            // Add the operation
            const result_t res = operations_list_append(
                &count, &capacity, &ops, (operation_t){.type = OPERATION_INVDIA, .invdia = {.idx = i_row}});
            if (res != RESULT_SUCCESS)
            {
                shared_res = res;
                continue;
            }
            // Perform the actual operation
            block_system_decompose_diagonal(this, i_row);
            block_system_apply_diagonal_inverse(this, i_row);
            target_status[i_row].status = TARGET_DONE;
        }
        else
        {
            target_status[i_row].status = TARGET_FREE;
            target_status[i_row].idx_src_needed = row->entries[0]->col;
            n_tgt += 1;
        }
    }

    if (shared_res != RESULT_SUCCESS)
    {
        PyMem_RawFree(target_status);
        return shared_res;
    }

#pragma omp parallel default(none) shared(n_tgt, this, target_status, count, capacity, ops, pn_ops, pp_ops,            \
                                              shared_res) if (n_threads > 1) num_threads(n_threads)
    while (n_tgt && shared_res == RESULT_SUCCESS)
    {
        u64 i_tgt = this->n;
        while (i_tgt == this->n && n_tgt > 0 && shared_res == RESULT_SUCCESS)
        {
            for (i_tgt = 0; i_tgt < this->n; ++i_tgt)
            {
                target_row_t *const status = target_status + i_tgt;
                if (status->status == TARGET_FREE && target_status[status->idx_src_needed].status == TARGET_DONE)
                {
                    target_status_t old_status;
#pragma omp atomic capture
                    {
                        old_status = status->status;
                        status->status = (status->status == TARGET_FREE) ? TARGET_IN_USE : status->status;
                    }

                    // Make this an atomic exchange, and we solve race conditions
                    if (old_status != TARGET_FREE)
                    {
                        continue;
                    }

                    break;
                }
            }
        }
        if (n_tgt == 0 || shared_res != RESULT_SUCCESS)
        {
            break;
        }

        target_row_t *const status = target_status + i_tgt;
        result_t res = block_system_eliminate_row(this, i_tgt, status->idx_src_needed);
        CPYUTL_ASSERT(res == RESULT_SUCCESS, "Failed to eliminate row %zu with row %zu.", i_tgt,
                      status->idx_src_needed);
        if (res != RESULT_SUCCESS)
        {
            shared_res = res;
            break;
        }

        res = operations_list_append(
            &count, &capacity, &ops,
            (operation_t){.type = OPERATION_ELIMIN, .elimin = {.idx_row = i_tgt, .idx_col = status->idx_src_needed}});

        if (res != RESULT_SUCCESS)
        {
            shared_res = res;
            break;
        }

        const sys_row_t *const row = this->rows + i_tgt;
        const u64 new_idx_src_needed =
            row_array_find_first_geq(row->count, (const row_entry_t *const *)row->entries, status->idx_src_needed + 1);
        const row_entry_t *const entry = row->entries[new_idx_src_needed];
        if (entry->col == i_tgt)
        {
            // This row is done, just decompose the diagonal and apply that, and we're done!
            block_system_decompose_diagonal(this, i_tgt);
            block_system_apply_diagonal_inverse(this, i_tgt);
            status->status = TARGET_DONE;
            res = operations_list_append(&count, &capacity, &ops,
                                         (operation_t){.type = OPERATION_INVDIA, .invdia = {.idx = i_tgt}});
            if (res != RESULT_SUCCESS)
            {
                shared_res = res;
                break;
            }

#pragma omp atomic update
            n_tgt -= 1;
        }
        else
        {
            // This one still needs work
            status->idx_src_needed = entry->col;
            status->status = TARGET_FREE;
        }
    }

    PyMem_RawFree(target_status);
    *pn_ops = count;
    *pp_ops = ops;
    return RESULT_SUCCESS;
}

void block_system_apply_diagonal_inverse(const block_system_t *this, const u64 idx_row)
{
    const sys_row_t *const row = this->rows + idx_row;
    const u64 i_diag = row_array_find_first_geq(row->count, (const row_entry_t *const *)row->entries, idx_row);
    CPYUTL_ASSERT(i_diag != row->count, "Row %zu does not have a diagonal block.", idx_row);
    row_entry_t *const diag = row->entries[i_diag];
    CPYUTL_ASSERT(diag->col == idx_row, "Row %zu does not have a diagonal block at the correct position.", idx_row);
    const u64 block_rows = block_system_get_block_size(this, idx_row);
    const matrix_t mat_diag = (matrix_t){
        .rows = block_rows,
        .cols = block_rows,
        .data = diag->vals,
    };
    for (u64 i = i_diag + 1; i < row->count; ++i)
    {
        row_entry_t *const entry = row->entries[i];
        const matrix_t mat_entry = (matrix_t){
            .rows = block_rows,
            .cols = block_system_get_block_size(this, entry->col),
            .data = entry->vals,
        };
        matrix_lu_solve(&mat_diag, &mat_entry, &mat_entry);
    }
}

void block_system_decompose_diagonal(const block_system_t *this, const u64 idx_row)
{
    const sys_row_t *const row = this->rows + idx_row;
    const u64 i_diag = row_array_find_first_geq(row->count, (const row_entry_t *const *)row->entries, idx_row);
    CPYUTL_ASSERT(i_diag != row->count, "Row %zu does not have a diagonal block.", idx_row);
    row_entry_t *const diag = row->entries[i_diag];
    CPYUTL_ASSERT(diag->col == idx_row, "Row %zu does not have a diagonal block at the correct position.", idx_row);
    const u64 block_rows = block_system_get_block_size(this, idx_row);
    const matrix_t mat_diag = (matrix_t){
        .rows = block_rows,
        .cols = block_rows,
        .data = diag->vals,
    };
    matrix_lu_decompose(&mat_diag);
}

result_t block_system_eliminate_row(const block_system_t *this, const u64 idx_tgt, const u64 idx_src)
{
    // Check the array has correct dimensions
    const u64 size_tgt = block_system_get_block_size(this, idx_tgt);
    const u64 size_src = block_system_get_block_size(this, idx_src);

    sys_row_t *const row_tgt = this->rows + idx_tgt;
    const sys_row_t *const row_src = this->rows + idx_src;

    matrix_t mat;
    const result_t res = block_system_get_block(this, idx_tgt, idx_src, &mat);
    if (res != RESULT_SUCCESS)
        return res;

    // Find what size the buffer should be to hold all intermediate results and
    // how many entries we will have after this is done.
    u64 unique_cols = 0, pos_tgt, pos_src;
    u64 max_cols = 0;
    for (pos_tgt = row_tgt->count, pos_src = row_src->count;; ++unique_cols)
    {
        const u64 entry_tgt = row_tgt->entries[pos_tgt - 1]->col;
        const u64 entry_src = row_src->entries[pos_src - 1]->col;

        if (entry_tgt <= (u64)idx_src && entry_src <= (u64)idx_src)
        {
            break;
        }

        u64 block_size;
        if (entry_tgt == entry_src)
        {
            pos_tgt -= 1;
            pos_src -= 1;
            block_size = block_system_get_block_size(this, entry_tgt);
        }
        else if (entry_src > entry_tgt)
        {
            pos_src -= 1;
            block_size = block_system_get_block_size(this, entry_src);
        }
        else // if (entry_src < entry_tgt)
        {
            pos_tgt -= 1;
            block_size = block_system_get_block_size(this, entry_tgt);
        }
        max_cols = max_cols < block_size ? block_size : max_cols;
    }
    CPYUTL_ASSERT(row_tgt->entries[pos_tgt - 1]->col == (u64)idx_src ||
                      row_src->entries[pos_src - 1]->col == (u64)idx_src,
                  "Source and target blocks must both contain the diagonal block for the row %zu.", idx_src);
    const u64 needed_size = unique_cols + pos_tgt;

    // Increase the capacity for the receiving row
    if (row_tgt->capacity < needed_size)
    {
        // TODO: change this to use the RAW memory api
        row_entry_t **const new_entries =
            (row_entry_t **)PyMem_RawRealloc((void *)row_tgt->entries, sizeof(*new_entries) * needed_size);
        if (!new_entries)
        {
            return RESULT_FAILED_ALLOC;
        }

        row_tgt->entries = new_entries;
        row_tgt->capacity = needed_size;
        // Zero initialize it
#pragma omp simd
        for (u64 i = row_tgt->count; i < needed_size; ++i)
        {
            row_tgt->entries[i] = NULL;
        }
    }

    f64 *const buffer = PyMem_RawMalloc(sizeof(*buffer) * max_cols * size_tgt);
    if (!buffer)
    {
        return RESULT_FAILED_ALLOC;
    }

    // Subtract the results of multiplying the source entries from the target entries.
    // If there's no target entry matching the "incoming" source entry, add it to
    // the row. Uneffected target entries are just copied, so we move back to the front
    // to perform no additional copies.
    // Since we are modifying these "inplace", there's no recovery if we fail memory
    // allocation here. All we can do is ruin the whole target row, bail, and at least
    // leave it in a valid state, were we should not cause a segfault when we clean up.
    u64 i, k;
    for (i = unique_cols, k = needed_size, pos_tgt = row_tgt->count, pos_src = row_src->count; i > 0; --i, --k)
    {
        row_entry_t *const entry_tgt = row_tgt->entries[pos_tgt - 1];
        const row_entry_t *const entry_src = row_src->entries[pos_src - 1];
        const u64 col_src = entry_src->col;
        const u64 col_tgt = entry_tgt->col;

        const matrix_t out =
            (matrix_t){.rows = size_tgt, .cols = block_system_get_block_size(this, col_src), .data = buffer};
        const matrix_t mat_src = {
            .rows = size_src, .cols = block_system_get_block_size(this, col_src), .data = (f64 *)entry_src->vals};
        const matrix_t mat_tgt = {
            .rows = size_tgt, .cols = block_system_get_block_size(this, col_tgt), .data = (f64 *)entry_tgt->vals};

        CPYUTL_ASSERT(pos_tgt == k || row_tgt->entries[k - 1] == NULL,
                      "Destination at index %zu was non-null and contained entry for column %zu!", k - 1, col_tgt);

        if (col_tgt == col_src)
        {
            // Scale src and subtract output
            matrix_multiply(&mat, &mat_src, &out);
            matrix_subtract_inplace(&mat_tgt, &out);

            row_tgt->entries[pos_tgt - 1] = NULL;
            row_tgt->entries[k - 1] = entry_tgt;
            pos_tgt -= 1;
            pos_src -= 1;
        }
        else if (col_src > col_tgt)
        {
            // Add the new entry
            matrix_multiply(&mat, &mat_src, &out);
            row_entry_t *const new_entry =
                (row_entry_t *)PyMem_RawMalloc(sizeof(*new_entry) + sizeof(*new_entry->vals) * out.rows * out.cols);

            if (!new_entry)
            {
                // Just try and salvage what is left, but the system will be ruined.
                // Free the whole row, release memory, then return.
                for (u64 j = 0; j < row_tgt->capacity; ++j)
                {
                    PyMem_Free(row_tgt->entries[j]);
                    row_tgt->entries[j] = NULL;
                }
                row_tgt->count = 0;

                PyMem_RawFree(buffer);
                return RESULT_FAILED_ALLOC;
            }
            new_entry->col = col_src;
            for (u64 j = 0; j < out.rows * out.cols; ++j)
                new_entry->vals[j] = -out.data[j];

            row_tgt->entries[k - 1] = new_entry;

            pos_src -= 1;
        }
        else // if (col_src < col_tgt)
        {
            // we don't do anything here
            row_tgt->entries[pos_tgt - 1] = NULL;
            row_tgt->entries[k - 1] = entry_tgt;
            pos_tgt -= 1;
        }
    }

    row_tgt->count = needed_size;

#ifdef CPYUTL_ENABLE_ASSERTS
    for (i = 0; i < row_tgt->count; ++i)
    {
        const row_entry_t *const entry = row_tgt->entries[i];
        CPYUTL_ASSERT(entry != NULL, "Entry %zu was NULL!", i);
        if (i > 0)
        {
            CPYUTL_ASSERT(entry->col > row_tgt->entries[i - 1]->col, "Entries %zu and %zu were not sorted!", i - 1, i);
        }
    }
#endif

    PyMem_RawFree(buffer);

    return RESULT_SUCCESS;
}

void block_system_apply_operations(const block_system_t *const this, const u64 nops,
                                   const operation_t ops[const static nops], f64 *const vec)
{
    for (const operation_t *op = ops; op < ops + nops; ++op)
    {
        switch (op->type)
        {
        case OPERATION_ELIMIN: {
            matrix_t block_mat;
            const u64 idx_tgt = op->elimin.idx_row;
            const u64 idx_src = op->elimin.idx_col;
            (void)block_system_get_block(this, idx_tgt, idx_src, &block_mat);
            const matrix_t in_vec = (matrix_t){
                .rows = block_system_get_block_size(this, idx_src),
                .cols = 1,
                .data = (f64 *)vec + this->block_offsets[idx_src],
            };
            const matrix_t out_vec = (matrix_t){
                .rows = block_system_get_block_size(this, idx_tgt),
                .cols = 1,
                .data = (f64 *)vec + this->block_offsets[idx_tgt],
            };
            matrix_multiply_sub_inplace(&block_mat, &in_vec, &out_vec);
        }
        break;

        case OPERATION_INVDIA: {
            matrix_t block_mat;
            const u64 idx_row = op->invdia.idx;
            (void)block_system_get_block(this, idx_row, idx_row, &block_mat);
            const matrix_t inout_vec = (matrix_t){
                .rows = block_system_get_block_size(this, idx_row),
                .cols = 1,
                .data = (f64 *)vec + this->block_offsets[idx_row],
            };
            matrix_lu_solve(&block_mat, &inout_vec, &inout_vec);
        }
        break;
        }
    }
}

void block_system_solve_u(const block_system_t *const this, f64 *const y)
{
    // Solve the U x = y system, where the diagonal blocks are identity
    for (u64 i = this->n; i > 0; --i)
    {
        const u64 idx_row = i - 1;
        const sys_row_t *const row = this->rows + idx_row;
        const u64 block_size = block_system_get_block_size(this, idx_row);
        const matrix_t vec = (matrix_t){
            .rows = block_size,
            .cols = 1,
            .data = y + this->block_offsets[idx_row],
        };
        for (u64 j = row->count; idx_row < row->entries[j - 1]->col; --j)
        {
            row_entry_t *const entry = row->entries[j - 1];
            const matrix_t mat_entry = (matrix_t){
                .rows = block_size,
                .cols = block_system_get_block_size(this, entry->col),
                .data = entry->vals,
            };
            const matrix_t vec_entry = (matrix_t){
                .rows = block_system_get_block_size(this, entry->col),
                .cols = 1,
                .data = y + this->block_offsets[entry->col],
            };
            matrix_multiply_sub_inplace(&mat_entry, &vec_entry, &vec);
        }
    }
}
