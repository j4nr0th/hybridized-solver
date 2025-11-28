#include "blocks.h"

int block_system_add_block(const block_system_t *this, const u64 idx_row, const u64 idx_col, const u64 nv,
                           const f64 vals[static nv])
{
    CPYUTL_ASSERT(idx_row < this->n && idx_col < this->n,
                  "Row and/or column index was out of bounds (indices: (%zu, %zu) but the limits are [0, %u)).",
                  idx_row, idx_col, this->n);
    CPYUTL_ASSERT(nv == this->block_sizes[idx_row] * this->block_sizes[idx_col],
                  "Number of values does not match block size (expected %zu x %zu, got %zu).",
                  this->block_sizes[idx_row], this->block_sizes[idx_col], nv);

    sys_row_t *const row = this->rows + idx_row;

    // Find where to insert it
    if (row->count == row->capacity)
    {
        const u64 new_capacity = row->capacity ? row->capacity * 2 : 8;
        row_entry_t **const ptr = (row_entry_t **)PyMem_Realloc(row->entries, new_capacity * sizeof(*row->entries));
        if (!ptr)
            return -1;
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

    row_entry_t *const new_entry = (row_entry_t *)PyMem_Malloc(sizeof(*new_entry) + nv * sizeof(*new_entry->vals));
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
        .rows = this->block_sizes[idx_row], .cols = this->block_sizes[idx_col], .data = row->entries[idx]->vals};

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
    CPYUTL_ASSERT(PyArray_NDIM(arr) == 2, "Expected a 2D array, got a %dD array.", PyArray_NDIM(arr));
    CPYUTL_ASSERT(check_input_array(arr, 0, (const npy_intp[]){}, NPY_DOUBLE,
                                    NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, "matrix") == 0,
                  "Array did not have correct dtype and/or flags!");
    return (matrix_t){.rows = (u64)PyArray_DIM(arr, 0), .cols = (u64)PyArray_DIM(arr, 1), .data = PyArray_DATA(arr)};
}

void matrix_multiply(const matrix_t *a, const matrix_t *b, const matrix_t *out)
{
    CPYUTL_ASSERT(a->cols == b->rows,
                  "Matrix multiplication: number of columns of A (%zu) does not match number of rows of B (%zu).",
                  a->cols, b->rows);
    CPYUTL_ASSERT(out->rows == a->rows && out->cols == b->cols, "Output matrix has incorrect dimensions.");

    for (u64 i = 0; i < a->rows; ++i)
    {
        for (u64 j = 0; j < b->cols; ++j)
        {
            f64 v = 0;

            for (u64 k = 0; k < a->cols; ++k)
                v += a->data[i * a->cols + k] * b->data[k * b->cols + j];

            out->data[i * out->cols + j] = v;
        }
    }
}
void matrix_subtract_inplace(const matrix_t *a, const matrix_t *b)
{
    CPYUTL_ASSERT(a->rows == b->rows && a->cols == b->cols, "Matrices have different dimensions.");
    for (u64 i = 0; i < a->rows * a->cols; ++i)
        a->data[i] -= b->data[i];
}
