#include "block_system.h"

static int block_system_ensure_decomposed(const block_system_object *const this)
{
    if (this->ops != NULL)
    {
        return 0;
    }

    PyErr_SetString(PyExc_RuntimeError, "The system has not been decomposed yet.");

    return -1;
}

static int block_system_ensure_not_decomposed(const block_system_object *const this)
{
    if (this->ops == NULL)
    {
        return 0;
    }

    PyErr_SetString(PyExc_RuntimeError, "The system has already been decomposed.");

    return -1;
}

static void block_system_dealloc(block_system_object *self)
{
    PyObject_GC_UnTrack(self);
    PyTypeObject *const type = Py_TYPE(self);
    PyMem_Free(self->system.block_offsets);
    self->system.block_offsets = NULL;
    if (self->system.rows)
    {
        for (u64 i = 0; i < self->system.n; ++i)
        {
            sys_row_t *const row = self->system.rows + i;
            row->capacity = 0;
            for (u64 j = 0; j < row->count; ++j)
            {
                PyMem_RawFree(row->entries[j]);
                row->entries[j] = NULL;
            }
            PyMem_RawFree(row->entries);
            row->count = 0;
            row->entries = NULL;
        }

        PyMem_Free(self->system.rows);
        self->system.rows = NULL;
    }
    PyMem_RawFree(self->ops);
    self->ops = NULL;
    self->nops = 0;
    type->tp_free((PyObject *)self);
    Py_DECREF(type);
}

static PyObject *block_system_new(PyTypeObject *subtype, PyObject *args, const PyObject *kwds)
{
    if (kwds != NULL && PyDict_GET_SIZE(kwds) != 0)
    {
        PyErr_SetString(PyExc_TypeError, "BlockSystem takes no keyword arguments");
        return NULL;
    }

    const module_state_t *const state = module_state_from_type(subtype);
    if (!state)
        return NULL;

    const u64 n_blocks = PyTuple_GET_SIZE(args);
    if (n_blocks == 0)
    {
        PyErr_SetString(PyExc_ValueError, "BlockSystem requires at least one block");
        return NULL;
    }

    block_system_object *const self = (block_system_object *)subtype->tp_alloc(subtype, 0);
    if (!self)
    {
        return NULL;
    }
    self->system.n = n_blocks;
    // Zero-initialize
    self->ops = NULL;
    self->nops = 0;
    self->system.rows = NULL;
    self->system.block_offsets = PyMem_Malloc((n_blocks + 1) * sizeof(*self->system.block_offsets));
    if (!self->system.block_offsets)
    {
        Py_DECREF(self);
        return NULL;
    }

    // Get these sizes
    u64 total_size = 0;
    self->system.block_offsets[0] = 0;
    for (u64 i = 0; i < n_blocks; ++i)
    {
        PyObject *const block_size = PyTuple_GET_ITEM(args, i);
        const Py_ssize_t size = PyNumber_AsSsize_t(block_size, PyExc_ValueError);
        if (PyErr_Occurred())
        {
            Py_DECREF(self);
            return NULL;
        }
        if (size <= 0)
        {
            PyErr_Format(PyExc_ValueError, "Block size of block %zu is %zd.", i, size);
            Py_DECREF(self);
            return NULL;
        }
        self->system.block_offsets[i + 1] = (total_size += (u64)size);
    }

    // Allocate rows
    self->system.rows = PyMem_Malloc(n_blocks * sizeof(*self->system.rows));
    if (!self->system.rows)
    {
        Py_DECREF(self);
        return NULL;
    }

    // Zero initialize all rows
    for (u64 i = 0; i < n_blocks; ++i)
    {
        sys_row_t *const row = self->system.rows + i;
        row->capacity = 0;
        row->count = 0;
        row->entries = NULL;
    }

    return (PyObject *)self;
}

PyDoc_STRVAR(block_system_docstring, "BlockSystem(*block_sizes: int)\nBlock system for hybridized solver.");

static int ensure_block_system_and_state(PyObject *self, PyTypeObject *defining_class, block_system_object **p_this,
                                         const module_state_t **p_state)
{
    const module_state_t *const state =
        defining_class ? module_state_from_type(defining_class) : module_state_from_type(Py_TYPE(self));
    if (!state)
        return -1;

    if (!PyObject_TypeCheck(self, state->type_block_system))
    {
        PyErr_Format(PyExc_TypeError, "Expected a %s, got a %s.", MODULE_TYPE_NAME(BlockSystem),
                     Py_TYPE(self)->tp_name);
        return -1;
    }
    *p_this = (block_system_object *)self;
    *p_state = state;
    return 0;
}

static int check_block_indices(const u64 n, const Py_ssize_t row_idx, const char *name)
{
    if (row_idx < 0 || (u64)row_idx >= n)
    {
        PyErr_Format(PyExc_ValueError, "%s must be in the range [0, %zu), but it was %zd.", name, n, row_idx);
        return -1;
    }
    return 0;
}

static PyObject *block_system_object_add_block(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                               const Py_ssize_t nargs, const PyObject *kwnames)
{
    const module_state_t *state;
    block_system_object *this;
    if (ensure_block_system_and_state(self, defining_class, &this, &state) < 0)
        return NULL;

    if (block_system_ensure_not_decomposed(this) < 0)
        return NULL;

    Py_ssize_t row_idx, col_idx;
    PyObject *value;

    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &row_idx, .kwname = "row"},
                {.type = CPYARG_TYPE_SSIZE, .p_val = &col_idx, .kwname = "col"},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &value, .kwname = "val"},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    // Check block indices are alright
    if (check_block_indices(this->system.n, row_idx, "Row index") < 0 ||
        check_block_indices(this->system.n, col_idx, "Column index") < 0)
    {
        return NULL;
    }

    // Convert the value to a NumPy array
    PyArrayObject *const arr =
        (PyArrayObject *)PyArray_FROMANY(value, NPY_DOUBLE, 2, 2, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!arr)
        return NULL;

    // Check the dimensions of the array
    if (check_input_array(arr, 2,
                          (const npy_intp[2]){(npy_intp)block_system_get_block_size(&this->system, row_idx),
                                              (npy_intp)block_system_get_block_size(&this->system, col_idx)},
                          NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, "val") < 0)
    {
        Py_DECREF(arr);
        return NULL;
    }

    const int ret = block_system_add_block(&this->system, row_idx, col_idx, PyArray_SIZE(arr), PyArray_DATA(arr));
    Py_DECREF(arr);
    if (ret < 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "Failed to add block to block system.");
        return NULL;
    }

    return Py_None;
}

PyDoc_STRVAR(block_system_add_block_docstring, "add_block(row: int, col: int, val: numpy.typing.ArrayLike)\n"
                                               "Add a block to the system.\n"
                                               "\n"
                                               "Parameters\n"
                                               "----------\n"
                                               "row : int\n"
                                               "    Row index of the block.\n"
                                               "col : int\n"
                                               "    Column index of the block.\n"
                                               "val : array_like\n"
                                               "    Value of the block.\n");

static PyObject *block_system_object_is_valid(PyObject *self, PyTypeObject *defining_class,
                                              PyObject *const *Py_UNUSED(args), const Py_ssize_t nargs,
                                              const PyObject *kwnames)
{
    if (nargs != 0 || kwnames != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "BlockSystem.is_valid() takes no arguments.");
        return NULL;
    }
    const module_state_t *state;
    block_system_object *this;
    if (ensure_block_system_and_state(self, defining_class, &this, &state) < 0)
        return NULL;

    return PyBool_FromLong(block_system_is_valid(&this->system));
}

PyDoc_STRVAR(block_system_object_is_valid_docstring,
             "is_valid() -> bool\n"
             "Check if the system has symmetric sparsity and has diagonal blocks.\n");

static PyObject *block_system_object_as_array(PyObject *self, PyTypeObject *defining_class,
                                              PyObject *const *Py_UNUSED(args), const Py_ssize_t nargs,
                                              const PyObject *kwnames)
{
    if (nargs != 0 || kwnames != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "BlockSystem.as_array() takes no arguments.");
        return NULL;
    }
    const module_state_t *state;
    block_system_object *this;
    if (ensure_block_system_and_state(self, defining_class, &this, &state) < 0)
        return NULL;

    const u64 total_size = this->system.block_offsets[this->system.n];
    const npy_intp dims[2] = {(npy_intp)total_size, (npy_intp)total_size};
    PyArrayObject *const arr = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!arr)
        return NULL;

    npy_double *const data = PyArray_DATA(arr);
    memset(data, 0, PyArray_SIZE(arr) * sizeof(*data));
    u64 offset_row = 0;
    for (u64 i = 0; i < this->system.n; ++i)
    {
        const sys_row_t *const row = this->system.rows + i;
        const u64 n1 = block_system_get_block_size(&this->system, i);
        u64 offset_col = 0, i_col = 0;
        for (u64 j = 0; j < row->count; ++j)
        {
            const row_entry_t *const entry = row->entries[j];
            while (i_col < entry->col)
            {
                offset_col += block_system_get_block_size(&this->system, i_col);
                i_col += 1;
            }
            const u64 n2 = block_system_get_block_size(&this->system, entry->col);
            for (u64 k1 = 0; k1 < n1; ++k1)
            {
                for (u64 k2 = 0; k2 < n2; ++k2)
                {
                    data[(offset_row + k1) * dims[1] + (offset_col + k2)] = entry->vals[k1 * n2 + k2];
                }
            }
        }
        offset_row += n1;
    }

    return (PyObject *)arr;
}

PyDoc_STRVAR(block_system_object_as_array_docstring, "as_array() -> numpy.typing.NDArray[numpy.double]\n"
                                                     "Return the system representation as a full matrix.\n");

static PyObject *block_system_object_get_row_block_indices(PyObject *self, PyTypeObject *defining_class,
                                                           PyObject *const *args, const Py_ssize_t nargs,
                                                           const PyObject *kwnames)
{
    const module_state_t *state;
    block_system_object *this;
    if (ensure_block_system_and_state(self, defining_class, &this, &state) < 0)
        return NULL;

    Py_ssize_t row_idx;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &row_idx, .kwname = "row"},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;
    if (check_block_indices(this->system.n, row_idx, "Row index") < 0)
    {
        return NULL;
    }

    const sys_row_t *const row = this->system.rows + row_idx;
    PyTupleObject *const ret = (PyTupleObject *)PyTuple_New((Py_ssize_t)row->count);
    if (!ret)
        return NULL;
    for (u64 i = 0; i < row->count; ++i)
    {
        PyObject *const idx = PyLong_FromUnsignedLongLong(row->entries[i]->col);
        if (!idx)
        {
            Py_DECREF(ret);
            return NULL;
        }
        PyTuple_SET_ITEM(ret, i, idx);
    }
    return (PyObject *)ret;
}

PyDoc_STRVAR(block_system_object_get_row_block_indices_docstring, "get_row_block_indices(row: int) -> tuple[int, ...]\n"
                                                                  "Get the column block indices of the specified row.\n"
                                                                  "\n"
                                                                  "Parameters\n"
                                                                  "----------\n"
                                                                  "row : int\n"
                                                                  "    Row index of the block to get the indices for.\n"
                                                                  "\n"
                                                                  "Returns\n"
                                                                  "-------\n"
                                                                  "tuple of int\n"
                                                                  "    Indices of columns that appear in the row.\n");

static PyObject *block_system_object_get_block(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                               const Py_ssize_t nargs, const PyObject *kwnames)
{
    const module_state_t *state;
    block_system_object *this;
    if (ensure_block_system_and_state(self, defining_class, &this, &state) < 0)
        return NULL;

    Py_ssize_t row_idx, col_idx;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &row_idx, .kwname = "row"},
                {.type = CPYARG_TYPE_SSIZE, .p_val = &col_idx, .kwname = "col"},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    if (check_block_indices(this->system.n, row_idx, "Row index") < 0 ||
        check_block_indices(this->system.n, col_idx, "Col index") < 0)
    {
        return NULL;
    }

    matrix_t mat;
    if (block_system_get_block(&this->system, row_idx, col_idx, &mat) != RESULT_SUCCESS)
    {
        PyErr_Format(PyExc_ValueError, "The system does not contain the block (%zd, %zd).", row_idx, col_idx);
        return NULL;
    }

    return (PyObject *)matrix_to_array(&mat);
}

PyDoc_STRVAR(block_system_object_get_block_docstring,
             "get_block(row: int, col: int) -> numpy.typing.NDArray[numpy.double]\n"
             "Get the block of the system.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "row : int\n"
             "    Row index of the block to get.\n"
             "col : int\n"
             "    Column index of the block to get.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "array\n"
             "    New array with the same value as the block.\n");

static PyObject *block_system_object_get_block_size(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                                    const Py_ssize_t nargs, const PyObject *kwnames)
{
    const module_state_t *state;
    block_system_object *this;
    if (ensure_block_system_and_state(self, defining_class, &this, &state) < 0)
        return NULL;
    Py_ssize_t row_idx, col_idx;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &row_idx, .kwname = "row"},
                {.type = CPYARG_TYPE_SSIZE, .p_val = &col_idx, .kwname = "col"},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    if (check_block_indices(this->system.n, row_idx, "Row index") < 0 ||
        check_block_indices(this->system.n, col_idx, "Col index") < 0)
    {
        return NULL;
    }

    return cpyutl_output_create_check(
        CPYOUT_TYPE_TUPLE, (const cpyutl_output_t[]){
                               {
                                   .type = CPYOUT_TYPE_PYINT,
                                   .value_int = (Py_ssize_t)block_system_get_block_size(&this->system, row_idx),
                               },
                               {
                                   .type = CPYOUT_TYPE_PYINT,
                                   .value_int = (Py_ssize_t)block_system_get_block_size(&this->system, col_idx),
                               },
                               {},
                           });
}

PyDoc_STRVAR(block_system_object_get_block_size_docstring, "get_block_size(row: int, col: int) -> tuple[int, int]\n"
                                                           "Get the size of a system block.\n");

static PyObject *block_system_object_multiply_row(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                                  const Py_ssize_t nargs, const PyObject *kwnames)
{
    const module_state_t *state;
    block_system_object *this;
    if (ensure_block_system_and_state(self, defining_class, &this, &state) < 0)
        return NULL;

    if (block_system_ensure_not_decomposed(this) < 0)
        return NULL;

    Py_ssize_t row_idx, start_idx = 0;
    PyObject *value;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &row_idx, .kwname = "row"},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &value, .kwname = "val"},
                {.type = CPYARG_TYPE_SSIZE, .p_val = &start_idx, .kwname = "start", .optional = 1},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    if (check_block_indices(this->system.n, row_idx, "Row index") < 0 ||
        check_block_indices(this->system.n, start_idx, "Start index"))
    {
        return NULL;
    }

    PyArrayObject *const arr =
        (PyArrayObject *)PyArray_FROMANY(value, NPY_DOUBLE, 2, 2, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!arr)
        return NULL;

    const matrix_t mat = matrix_from_array(arr);
    const u64 rows = block_system_get_block_size(&this->system, row_idx);
    if (mat.rows != mat.cols || mat.rows != rows)
    {
        PyErr_Format(PyExc_ValueError, "Value must be a square matrix of size %zd.", rows);
        Py_DECREF(arr);
        return NULL;
    }

    const sys_row_t *row = this->system.rows + row_idx;
    u64 max_cols = 0;
    for (u64 i = 0; i < row->count; ++i)
    {
        const u64 block_size = block_system_get_block_size(&this->system, row->entries[i]->col);
        if (block_size > max_cols)
        {
            max_cols = block_size;
        }
    }

    // This buffer can hold the result of any matmuls for this row
    f64 *const buffer = PyMem_RawMalloc(sizeof(f64) * max_cols * rows);
    if (!buffer)
    {
        Py_DECREF(arr);
        return NULL;
    }

    for (u64 i = 0; i < row->count; ++i)
    {
        // Prepare in and out arrays
        row_entry_t *const entry = row->entries[i];
        if (entry->col < (u64)start_idx)
            continue;

        const matrix_t block = {
            .rows = rows, .cols = block_system_get_block_size(&this->system, entry->col), .data = entry->vals};
        const matrix_t out = {.rows = block.rows, .cols = block.cols, .data = buffer};

        // Multiply
        matrix_multiply(&mat, &block, &out);

        // Copy the result back to the original
        memcpy(entry->vals, buffer, sizeof(*entry->vals) * block.rows * block.cols);
    }

    // Release buffer and array
    PyMem_RawFree(buffer);
    Py_DECREF(arr);
    Py_RETURN_NONE;
}

PyDoc_STRVAR(block_system_object_multiply_row_docstring,
             "multiply_row(row: int, val: numpy.tying.ArrayLike) -> None\n"
             "Multiply the row by the matrix\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "row : int\n"
             "    Row index of blocks to multiply.\n"
             "\n"
             "val : array_like\n"
             "    Square matrix with which the row should be multiplied.\n");

static PyObject *block_system_object_eliminate_row(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                                   const Py_ssize_t nargs, const PyObject *kwnames)
{
    const module_state_t *state;
    block_system_object *this;
    if (ensure_block_system_and_state(self, defining_class, &this, &state) < 0)
        return NULL;

    if (block_system_ensure_not_decomposed(this) < 0)
        return NULL;

    Py_ssize_t i_row_src, i_row_tgt;
    PyObject *py_val;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &i_row_src, .kwname = "row_src"},
                {.type = CPYARG_TYPE_SSIZE, .p_val = &i_row_tgt, .kwname = "row_tgt"},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &py_val, .kwname = "val"},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    if (check_block_indices(this->system.n, i_row_src, "Row index source") < 0 ||
        check_block_indices(this->system.n, i_row_tgt, "Row index target") < 0)
    {
        return NULL;
    }

    // Check the array has correct dimensions
    const u64 size_tgt = block_system_get_block_size(&this->system, i_row_tgt);
    const u64 size_src = block_system_get_block_size(&this->system, i_row_src);

    sys_row_t *const row_tgt = this->system.rows + i_row_tgt;
    const sys_row_t *const row_src = this->system.rows + i_row_src;

    // Find what size the buffer should be to hold all intermediate results and
    // how many entries we will have after this is done.
    u64 unique_cols = 0, pos_tgt, pos_src;
    u64 max_cols = 0;
    for (pos_tgt = row_tgt->count, pos_src = row_src->count;; ++unique_cols)
    {
        const u64 entry_tgt = row_tgt->entries[pos_tgt - 1]->col;
        const u64 entry_src = row_src->entries[pos_src - 1]->col;

        if (entry_tgt <= (u64)i_row_src && entry_src <= (u64)i_row_src)
        {
            break;
        }

        u64 block_size;
        if (entry_tgt == entry_src)
        {
            pos_tgt -= 1;
            pos_src -= 1;
            block_size = block_system_get_block_size(&this->system, entry_tgt);
        }
        else if (entry_src > entry_tgt)
        {
            pos_src -= 1;
            block_size = block_system_get_block_size(&this->system, entry_src);
        }
        else // if (entry_src < entry_tgt)
        {
            pos_tgt -= 1;
            block_size = block_system_get_block_size(&this->system, entry_tgt);
        }
        max_cols = max_cols < block_size ? block_size : max_cols;
    }
    if (row_tgt->entries[pos_tgt - 1]->col != (u64)i_row_src && row_src->entries[pos_src - 1]->col != (u64)i_row_src)
    {
        PyErr_Format(PyExc_ValueError, "Source and target blocks must both contain the diagonal block for the row %zu.",
                     i_row_src);
        return NULL;
    }
    const u64 needed_size = unique_cols + pos_tgt;

    // Make it an array
    PyArrayObject *const arr =
        (PyArrayObject *)PyArray_FROMANY(py_val, NPY_DOUBLE, 2, 2, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!arr)
        return NULL;
    if (check_input_array(arr, 2, (const npy_intp[2]){(npy_intp)size_tgt, (npy_intp)size_src}, NPY_DOUBLE,
                          NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, "val") < 0)
    {
        Py_DECREF(arr);
        return NULL;
    }
    const matrix_t mat = matrix_from_array(arr);

    // Increase the capacity for the receiving row
    if (row_tgt->capacity < needed_size)
    {
        row_entry_t **const new_entries =
            (row_entry_t **)PyMem_RawRealloc((void *)row_tgt->entries, sizeof(*new_entries) * needed_size);
        if (!new_entries)
        {
            Py_DECREF(arr);
            return NULL;
        }

        row_tgt->entries = new_entries;
        row_tgt->capacity = needed_size;
        // Pre-emptively zero it
        for (u64 i = row_tgt->count; i < needed_size; ++i)
        {
            row_tgt->entries[i] = NULL;
        }
    }

    f64 *const buffer = PyMem_RawMalloc(sizeof(*buffer) * max_cols * size_tgt);
    if (!buffer)
    {
        Py_DECREF(arr);
        return NULL;
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
            (matrix_t){.rows = size_tgt, .cols = block_system_get_block_size(&this->system, col_src), .data = buffer};
        const matrix_t mat_src = {.rows = size_src,
                                  .cols = block_system_get_block_size(&this->system, col_src),
                                  .data = (f64 *)entry_src->vals};
        const matrix_t mat_tgt = {.rows = size_tgt,
                                  .cols = block_system_get_block_size(&this->system, col_tgt),
                                  .data = (f64 *)entry_tgt->vals};

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

                PyErr_Format(PyExc_MemoryError, "Failed to allocate memory for eliminated row. This error can not be "
                                                "recovered from for the system, as the entire row was now erased.");

                PyMem_RawFree(buffer);
                Py_DECREF(arr);
                return NULL;
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
    Py_DECREF(arr);
    Py_RETURN_NONE;
}

PyDoc_STRVAR(block_system_object_eliminate_row_docstring,
             "eliminate_row(row_src: int, row_tgt: int, val: npt.ArrayLike) -> None\n"
             "Eliminate a target row using a source row, multiplied by matrix.\n"
             "\n"
             "This performs the row elimination operation from the source row on the target\n"
             "row. If the source row is represented by :math:`\\mathbf{M}_s`, the target row\n"
             "by :math:`\\mathbf{M}_t`, and the scaling matrix as :math:`\\mathbf{S}`, the\n"
             "new vale of the target row will be:\n"
             "\n"
             "..math ::\n"
             "    \\mathbf{M}_t^\\prime = \\mathbf{M}_t - \\mathbf{S} \\mathbf{M}_s\n"
             "\n"
             "This operation will ignore all entres in both rows with column index lower or\n"
             "equal to ``row_src``, since those are assumed to have already been eliminated.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "row_src : int\n"
             "    Index of the source row.\n"
             "\n"
             "row_tgt : int\n"
             "    Index of the row to eliminate.\n"
             "\n"
             "val : array_like\n"
             "    Matrix used to scale the target row.\n");

static PyObject *block_system_object_has_block(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                               const Py_ssize_t nargs, const PyObject *kwnames)
{
    const module_state_t *state;
    block_system_object *this;
    if (ensure_block_system_and_state(self, defining_class, &this, &state) < 0)
        return NULL;

    Py_ssize_t i_row, i_col;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &i_row, .kwname = "row"},
                {.type = CPYARG_TYPE_SSIZE, .p_val = &i_col, .kwname = "col"},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    if (check_block_indices(this->system.n, i_row, "Row index") < 0 ||
        check_block_indices(this->system.n, i_col, "Column index") < 0)
    {
        return NULL;
    }

    const sys_row_t *const row = this->system.rows + i_row;
    const u64 idx = row_array_find_first_geq(row->count, (const row_entry_t *const *)row->entries, i_col);
    if (idx == row->count || row->entries[idx]->col != (u64)i_col)
    {
        Py_RETURN_FALSE;
    }
    Py_RETURN_TRUE;
}

PyDoc_STRVAR(block_system_object_has_block_docstring,
             "has_block(row: int, col: int) -> bool\n"
             "Check if the block at row ``row`` and column ``col`` is present.\n");

static PyObject *block_system_object_no_lower_connections(PyObject *self, PyTypeObject *defining_class,
                                                          PyObject *const *Py_UNUSED(args), const Py_ssize_t nargs,
                                                          const PyObject *kwnames)
{
    const module_state_t *state;
    block_system_object *this;
    if (ensure_block_system_and_state(self, defining_class, &this, &state) < 0)
        return NULL;

    if (nargs != 0 || kwnames != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "No arguments accepted.");
        return NULL;
    }

    const npy_intp dim = (npy_intp)this->system.n;
    PyArrayObject *const arr = (PyArrayObject *)PyArray_SimpleNew(1, &dim, NPY_BOOL);
    if (!arr)
        return NULL;
    npy_bool *const out = (npy_bool *)PyArray_DATA(arr);
    for (u64 i = 0; i < this->system.n; ++i)
    {
        out[i] = this->system.rows[i].count == 0 ? 1 : this->system.rows[i].entries[0]->col >= i;
    }
    return (PyObject *)arr;
}

PyDoc_STRVAR(block_system_object_no_lower_connections_docstring,
             "no_lower_connections() -> numpy.typing.NDArray[numpy.bool]\n"
             "Return an array indicating entries to the left of the diagonal.\n");

static PyObject *block_system_object_first_column(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                                  const Py_ssize_t nargs, const PyObject *kwnames)
{
    const module_state_t *state;
    block_system_object *this;
    if (ensure_block_system_and_state(self, defining_class, &this, &state) < 0)
        return NULL;
    Py_ssize_t i_row;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &i_row, .kwname = "row"},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    if (check_block_indices(this->system.n, i_row, "Row index") < 0)
        return NULL;

    const sys_row_t *const row = this->system.rows + i_row;
    if (row->count == 0)
    {
        PyErr_SetString(PyExc_ValueError, "Row has no entries.");
        return NULL;
    }
    return PyLong_FromUnsignedLongLong(row->entries[0]->col);
}

PyDoc_STRVAR(block_system_object_first_column_docstring, "first_column(row: int) -> int\n"
                                                         "Return the index of the first index in a non-empy row.\n");

static PyObject *block_system_object_get_next_column_index(PyObject *self, PyTypeObject *defining_class,
                                                           PyObject *const *args, const Py_ssize_t nargs,
                                                           const PyObject *kwnames)
{
    const module_state_t *state;
    block_system_object *this;
    if (ensure_block_system_and_state(self, defining_class, &this, &state) < 0)
        return NULL;

    Py_ssize_t i_row, i_col;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &i_row, .kwname = "row"},
                {.type = CPYARG_TYPE_SSIZE, .p_val = &i_col, .kwname = "col"},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    if (check_block_indices(this->system.n, i_row, "Row index") < 0 ||
        check_block_indices(this->system.n, i_col, "Column index") < 0)
    {
        return NULL;
    }

    const sys_row_t *const row = this->system.rows + i_row;
    if (row->count == 0)
    {
        PyErr_SetString(PyExc_ValueError, "Row has no entries.");
        return NULL;
    }

    const u64 idx = row_array_find_first_geq(row->count, (const row_entry_t *const *)row->entries, i_col + 1);
    if (idx == row->count)
    {
        PyErr_SetString(PyExc_ValueError, "Row has no entries after column index.");
        return NULL;
    }

    return PyLong_FromUnsignedLongLong(row->entries[idx]->col);
}

PyDoc_STRVAR(block_system_object_get_next_column_index_docstring, "get_next_column_index(row: int, col: int) -> int\n"
                                                                  "Get the column index after the current one.\n");

static PyObject *block_system_object_decompose_diagonal(PyObject *self, PyTypeObject *defining_class,
                                                        PyObject *const *args, const Py_ssize_t nargs,
                                                        const PyObject *kwnames)
{
    const module_state_t *state;
    block_system_object *this;
    if (ensure_block_system_and_state(self, defining_class, &this, &state) < 0)
        return NULL;

    if (block_system_ensure_not_decomposed(this) < 0)
        return NULL;

    Py_ssize_t i_row;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &i_row, .kwname = "idx"},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    if (check_block_indices(this->system.n, i_row, "Block index") < 0)
        return NULL;

    matrix_t mat;
    if (block_system_get_block(&this->system, i_row, i_row, &mat) != RESULT_SUCCESS)
    {
        PyErr_Format(PyExc_ValueError, "Row %zu has no diagonal entry.", i_row);
        return NULL;
    }

    matrix_lu_decompose(&mat);

    Py_RETURN_NONE;
}

PyDoc_STRVAR(block_system_object_decompose_diagonal_docstring,
             "decompose_diagonal(idx: int) -> None\n"
             "Decomposes the diagonal block using LU decomposition.\n"
             "\n"
             "Performs unpivoted LU decomposition on the block ``(idx, idx)``. This\n"
             "is done in preparation to a call to ``solve_diagonal``.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "idx : int\n"
             "    Index of the block to decompose.\n");

static PyObject *block_system_object_solve_diagonal(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                                    const Py_ssize_t nargs, const PyObject *kwnames)
{
    const module_state_t *state;
    block_system_object *this;
    if (ensure_block_system_and_state(self, defining_class, &this, &state) < 0)
        return NULL;
    Py_ssize_t i_block;
    PyObject *val;
    PyArrayObject *out = NULL;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &i_block, .kwname = "idx"},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &val, .kwname = "val"},
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = &out,
                    .kwname = "out",
                    .type_check = &PyArray_Type,
                    .optional = 1,
                },
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    if (check_block_indices(this->system.n, i_block, "Block index") < 0)
        return NULL;

    matrix_t mat_block;
    if (block_system_get_block(&this->system, i_block, i_block, &mat_block) != RESULT_SUCCESS)
    {
        PyErr_Format(PyExc_ValueError, "Row %zu has no diagonal entry.", i_block);
        return NULL;
    }

    // Convert the input into an array
    PyArrayObject *const arr =
        (PyArrayObject *)PyArray_FROMANY(val, NPY_DOUBLE, 1, 2, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    if (!arr)
        return NULL;
    const matrix_t mat_in = matrix_from_array(arr);

    // Deal with output
    if (out)
    {
        // It was given, so check it!
        if (check_input_array(out, PyArray_NDIM(arr), PyArray_DIMS(arr), NPY_DOUBLE,
                              NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE, "out") < 0)
        {
            Py_DECREF(arr);
            return NULL;
        }
        Py_INCREF(out);
    }
    else
    {
        out = (PyArrayObject *)PyArray_SimpleNew(2, PyArray_DIMS(arr), NPY_DOUBLE);
        if (!out)
        {
            Py_DECREF(arr);
            return NULL;
        }
    }
    const matrix_t mat_out = matrix_from_array(out);

    // Solve and store the output.
    matrix_lu_solve(&mat_block, &mat_in, &mat_out);

    Py_RETURN_NONE;
}

PyDoc_STRVAR(block_system_object_solve_diagonal_docstring,
             "solve_diagonal(idx: int, val: numpy.tying.ArrayLike, out: numpy.typing.NDArray[numpy.double] | None = "
             "None) -> numpy.tying.NDArray[numpy.double]\n"
             "Use the previously decomposed diagonal to solve the linear system.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "idx : int\n"
             "    Index of the diagonal to use. For this method to make any sense, a call to\n"
             "    :meth:`BlockSystem.decompose_diagonal` should have been made for the same\n"
             "    block ``idx``.\n"
             "\n"
             "val : array_like\n"
             "    Value to use as the right side of the matrix.\n"
             "\n"
             "out : array, optional\n"
             "    Array to write the result to. If not given (or none), a new array\n"
             "    is created.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "array\n"
             "    Result, which if ``out`` was ``None`` will be in a new array, otherwise\n"
             "    another reference to ``out`` is returned.\n");

static PyObject *block_system_object_row_apply_decomposition(PyObject *self, PyTypeObject *defining_class,
                                                             PyObject *const *args, const Py_ssize_t nargs,
                                                             const PyObject *kwnames)
{
    const module_state_t *state;
    block_system_object *this;
    if (ensure_block_system_and_state(self, defining_class, &this, &state) < 0)
        return NULL;

    Py_ssize_t i_row;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &i_row, .kwname = "row"},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    if (check_block_indices(this->system.n, i_row, "Row index") < 0)
        return NULL;

    const sys_row_t *const row = this->system.rows + i_row;
    if (row->count == 0)
    {
        PyErr_Format(PyExc_ValueError, "Row %zu has no entries.", i_row);
        return NULL;
    }
    const u64 diagonal_idx = row_array_find_first_geq(row->count, (const row_entry_t *const *)row->entries, i_row);

    if (diagonal_idx == row->count || row->entries[diagonal_idx]->col != (u64)i_row)
    {
        PyErr_Format(PyExc_ValueError, "Row %zu has no diagonal entry.", i_row);
        return NULL;
    }

    const u64 block_rows = block_system_get_block_size(&this->system, i_row);
    const matrix_t mat_block =
        (matrix_t){.rows = block_rows, .cols = block_rows, .data = row->entries[diagonal_idx]->vals};
    for (u64 i = diagonal_idx + 1; i < row->count; ++i)
    {
        row_entry_t *const entry = row->entries[i];
        const matrix_t mat_entry = (matrix_t){
            .rows = block_rows, .cols = block_system_get_block_size(&this->system, entry->col), .data = entry->vals};
        matrix_lu_solve(&mat_block, &mat_entry, &mat_entry);
    }

    Py_RETURN_NONE;
}

PyDoc_STRVAR(block_system_object_row_apply_decomposition_docstring,
             "row_apply_decomposition(row: int) -> None\n"
             "Apply the decomposition of the diagonal block to the rest of the same row.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "row : int\n"
             "    Index of the row to perform this on. This row must have had its diagonal\n"
             "    block decomposed by a call to :meth:`BlockSystem.decompose_diagonal` with\n"
             "    ``row`` passed to it before.\n");

static PyObject *block_system_object_decompose(PyObject *self, PyTypeObject *defining_class,
                                               PyObject *const *Py_UNUSED(args), const Py_ssize_t nargs,
                                               const PyObject *kwnames)
{
    const module_state_t *state;
    block_system_object *this;
    if (ensure_block_system_and_state(self, defining_class, &this, &state) < 0)
        return NULL;

    if (block_system_ensure_not_decomposed(this) < 0)
        return NULL;

    if (block_system_is_valid(&this->system) < 0)
    {
        PyErr_SetString(PyExc_ValueError, "Block system is not valid.");
        return NULL;
    }

    if (nargs != 0 || kwnames != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "No arguments accepted.");
        return NULL;
    }

    result_t res;
    Py_BEGIN_ALLOW_THREADS;
    res = block_system_decompose(&this->system, &this->nops, &this->ops);
    Py_END_ALLOW_THREADS;

    if (res != RESULT_SUCCESS)
    {
        PyErr_Format(PyExc_RuntimeError, "Block system decomposition failed with error %d.", res);
        return NULL;
    }

    Py_RETURN_NONE;
}

PyDoc_STRVAR(block_system_object_decompose_docstring, "decompose() -> None\n"
                                                      "Decompose the block system.\n");

static PyObject *block_system_object_operations(PyObject *self, PyTypeObject *defining_class,
                                                PyObject *const *Py_UNUSED(args), const Py_ssize_t nargs,
                                                const PyObject *kwnames)
{
    const module_state_t *state;
    block_system_object *this;
    if (ensure_block_system_and_state(self, defining_class, &this, &state) < 0)
        return NULL;

    if (block_system_ensure_decomposed(this) < 0)
        return NULL;

    if (nargs != 0 || kwnames != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "No arguments accepted.");
        return NULL;
    }

    PyTupleObject *const out = (PyTupleObject *)PyTuple_New((Py_ssize_t)this->nops);

    for (u64 i = 0; i < this->nops; ++i)
    {
        const operation_t op = this->ops[i];
        PyObject *val = NULL;
        switch (op.type)
        {
        case OPERATION_INVDIA:
            val = cpyutl_output_create_check(CPYOUT_TYPE_TUPLE,
                                             (const cpyutl_output_t[]){
                                                 {.type = CPYOUT_TYPE_PYINT, .value_int = (Py_ssize_t)op.invdia.idx},
                                                 {},
                                             });
            break;

        case OPERATION_ELIMIN:
            val = cpyutl_output_create_check(
                CPYOUT_TYPE_TUPLE, (const cpyutl_output_t[]){
                                       {.type = CPYOUT_TYPE_PYINT, .value_int = (Py_ssize_t)op.elimin.idx_row},
                                       {.type = CPYOUT_TYPE_PYINT, .value_int = (Py_ssize_t)op.elimin.idx_col},
                                       {},
                                   });
            break;
        default:
            PyErr_Format(PyExc_RuntimeError, "Unknown operation type (%u).", op.type);
            Py_DECREF(out);
            return NULL;
        }
        if (val == NULL)
        {
            Py_DECREF(out);
            return NULL;
        }
        PyTuple_SET_ITEM(out, i, val);
    }

    return (PyObject *)out;
}

PyDoc_STRVAR(block_system_object_operations_docstring, "operations() -> tuple[tuple[int, int] | tuple[int]]\n"
                                                       "Get operations as tuples of one or two ints.");

static PyObject *block_system_object_solve(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                           const Py_ssize_t nargs, const PyObject *kwnames)
{
    const module_state_t *state;
    block_system_object *this;
    if (ensure_block_system_and_state(self, defining_class, &this, &state) < 0)
        return NULL;

    if (block_system_ensure_decomposed(this) < 0)
        return NULL;

    PyObject *val;
    PyArrayObject *out = NULL;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = &val, .kwname = "val"},
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = &out,
                    .kwname = "out",
                    .type_check = &PyArray_Type,
                    .optional = 1,
                },
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    // Check the input array has the correct shape
    PyArrayObject *const arr =
        (PyArrayObject *)PyArray_FROMANY(val, NPY_DOUBLE, 1, 1, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    if (!arr)
        return NULL;
    if (check_input_array(arr, 1, (const npy_intp[1]){(npy_intp)this->system.block_offsets[this->system.n]}, NPY_DOUBLE,
                          NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, "val") < 0)
    {
        Py_DECREF(arr);
        return NULL;
    }

    // Deal with output
    if (out)
    {
        // It was given, so check it!
        if (check_input_array(out, PyArray_NDIM(arr), PyArray_DIMS(arr), NPY_DOUBLE,
                              NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_WRITEABLE, "out") < 0)
        {
            Py_DECREF(arr);
            return NULL;
        }
        Py_INCREF(out);
    }
    else
    {
        out = (PyArrayObject *)PyArray_SimpleNew(1, PyArray_DIMS(arr), NPY_DOUBLE);
        if (!out)
        {
            Py_DECREF(arr);
            return NULL;
        }
    }

    // Do the actual solving
    Py_BEGIN_ALLOW_THREADS;
    npy_double *const ptr = PyArray_DATA(out);
    if (arr != out)
    {
        // If input and output params are the same array, do not bother copying
        memcpy(ptr, PyArray_DATA(arr), PyArray_SIZE(arr) * sizeof(double));
    }
    block_system_apply_operations(&this->system, this->nops, this->ops, ptr);
    block_system_solve_u(&this->system, ptr);
    Py_END_ALLOW_THREADS;

    Py_DECREF(arr);
    return (PyObject *)out;
}

PyDoc_STRVAR(block_system_object_solve_docstring,
             "solve(val: numpy.typing.ArrayLike, out: numpy.typing.NDArray[numpy.double] | None = None) -> "
             "numpy.typing.NDArray[numpy.double]\n"

);

static PyObject *block_system_object_get_n_blocks(PyObject *self, void *Py_UNUSED(closure))
{
    const block_system_object *this = (block_system_object *)self;
    return PyLong_FromUnsignedLongLong(this->system.n);
}

static PyObject *block_system_object_get_block_sizes(PyObject *self, void *Py_UNUSED(closure))
{
    const block_system_object *this = (block_system_object *)self;
    const npy_intp dim = (npy_intp)this->system.n;
    PyArrayObject *const arr = (PyArrayObject *)PyArray_SimpleNew(1, &dim, NPY_UINT64);
    if (!arr)
        return NULL;
    npy_uint64 *const out = (npy_uint64 *)PyArray_DATA(arr);
    for (u64 i = 0; i < this->system.n; ++i)
    {
        out[i] = block_system_get_block_size(&this->system, i);
    }
    return (PyObject *)arr;
}

PyType_Spec block_system_type_spec = {
    .name = MODULE_TYPE_NAME(BlockSystem),
    .basicsize = sizeof(block_system_object),
    .itemsize = 0,
    .flags = Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_DEFAULT,
    .slots =
        (PyType_Slot[]){
            {Py_tp_new, block_system_new},
            {Py_tp_traverse, heap_type_traverse_type},
            {Py_tp_dealloc, block_system_dealloc},
            {Py_tp_doc, (void *)block_system_docstring},
            {Py_tp_methods,
             (PyMethodDef[]){
                 {
                     .ml_name = "add_block",
                     .ml_meth = (void *)block_system_object_add_block,
                     .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
                     .ml_doc = block_system_add_block_docstring,
                 },
                 {
                     .ml_name = "is_valid",
                     .ml_meth = (void *)block_system_object_is_valid,
                     .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
                     .ml_doc = block_system_object_is_valid_docstring,
                 },
                 {
                     .ml_name = "as_array",
                     .ml_meth = (void *)block_system_object_as_array,
                     .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
                     .ml_doc = block_system_object_as_array_docstring,
                 },
                 {
                     .ml_name = "get_row_block_indices",
                     .ml_meth = (void *)block_system_object_get_row_block_indices,
                     .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
                     .ml_doc = block_system_object_get_row_block_indices_docstring,
                 },
                 {
                     .ml_name = "get_block",
                     .ml_meth = (void *)block_system_object_get_block,
                     .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
                     .ml_doc = block_system_object_get_block_docstring,
                 },
                 {
                     .ml_name = "get_block_size",
                     .ml_meth = (void *)block_system_object_get_block_size,
                     .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
                     .ml_doc = block_system_object_get_block_size_docstring,
                 },
                 {
                     .ml_name = "multiply_row",
                     .ml_meth = (void *)block_system_object_multiply_row,
                     .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
                     .ml_doc = block_system_object_multiply_row_docstring,
                 },
                 {
                     .ml_name = "eliminate_row",
                     .ml_meth = (void *)block_system_object_eliminate_row,
                     .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
                     .ml_doc = block_system_object_eliminate_row_docstring,
                 },
                 {
                     .ml_name = "has_block",
                     .ml_meth = (void *)block_system_object_has_block,
                     .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
                     .ml_doc = block_system_object_has_block_docstring,
                 },
                 {
                     .ml_name = "no_lower_connections",
                     .ml_meth = (void *)block_system_object_no_lower_connections,
                     .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
                     .ml_doc = block_system_object_no_lower_connections_docstring,
                 },
                 {
                     .ml_name = "first_column",
                     .ml_meth = (void *)block_system_object_first_column,
                     .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
                     .ml_doc = block_system_object_first_column_docstring,
                 },
                 {
                     .ml_name = "get_next_column_index",
                     .ml_meth = (void *)block_system_object_get_next_column_index,
                     .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
                     .ml_doc = block_system_object_get_next_column_index_docstring,
                 },
                 {
                     .ml_name = "decompose_diagonal",
                     .ml_meth = (void *)block_system_object_decompose_diagonal,
                     .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
                     .ml_doc = block_system_object_decompose_diagonal_docstring,
                 },
                 {
                     .ml_name = "solve_diagonal",
                     .ml_meth = (void *)block_system_object_solve_diagonal,
                     .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
                     .ml_doc = block_system_object_solve_diagonal_docstring,
                 },
                 {
                     .ml_name = "row_apply_decomposition",
                     .ml_meth = (void *)block_system_object_row_apply_decomposition,
                     .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
                     .ml_doc = block_system_object_row_apply_decomposition_docstring,
                 },
                 {
                     .ml_name = "decompose",
                     .ml_meth = (void *)block_system_object_decompose,
                     .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
                     .ml_doc = block_system_object_decompose_docstring,
                 },
                 {
                     .ml_name = "operations",
                     .ml_meth = (void *)block_system_object_operations,
                     .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
                     .ml_doc = block_system_object_operations_docstring,
                 },
                 {
                     .ml_name = "solve",
                     .ml_meth = (void *)block_system_object_solve,
                     .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
                     .ml_doc = block_system_object_solve_docstring,
                 },
                 {},
             }},
            {Py_tp_getset,
             (PyGetSetDef[]){
                 {
                     .name = "n_blocks",
                     .get = block_system_object_get_n_blocks,
                     .doc = "int : Number of blocks.",
                 },
                 {
                     .name = "block_sizes",
                     .get = block_system_object_get_block_sizes,
                     .doc = "numpy.typing.NDArray[numpy.uint64] : Array of sizes of blocks.",
                 },
                 {},
             }},
            {},
        },

};
