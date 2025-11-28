#include "block_system.h"

static void block_system_dealloc(block_system_object *self)
{
    PyObject_GC_UnTrack(self);
    PyTypeObject *const type = Py_TYPE(self);
    PyMem_Free(self->system.block_sizes);
    self->system.block_sizes = NULL;
    if (self->system.rows)
    {
        for (u64 i = 0; i < self->system.n; ++i)
        {
            sys_row_t *const row = self->system.rows + i;
            row->capacity = 0;
            for (u64 j = 0; j < row->count; ++j)
            {
                PyMem_Free(row->entries[j]);
                row->entries[j] = NULL;
            }
            PyMem_Free(row->entries);
            row->count = 0;
            row->entries = NULL;
        }
        PyMem_Free(self->system.rows);
        self->system.rows = NULL;
    }
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
    self->system.rows = NULL;
    self->system.block_sizes = PyMem_Malloc(n_blocks * sizeof(*self->system.block_sizes));
    if (!self->system.block_sizes)
    {
        Py_DECREF(self);
        return NULL;
    }

    // Get these sizes
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
        self->system.block_sizes[i] = (u64)size;
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

static PyObject *block_system_object_add_block(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                               const Py_ssize_t nargs, const PyObject *kwnames)
{
    const module_state_t *state;
    block_system_object *this;
    if (ensure_block_system_and_state(self, defining_class, &this, &state) < 0)
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
    if (row_idx < 0 || row_idx >= this->system.n || col_idx < 0 || col_idx >= this->system.n)
    {
        PyErr_Format(PyExc_ValueError, "Block indices must be in the range [0, %zu), but they were instead %zd and %zd",
                     this->system.n, row_idx, col_idx);
        return NULL;
    }

    // Convert the value to a NumPy array
    PyArrayObject *const arr =
        (PyArrayObject *)PyArray_FROMANY(value, NPY_DOUBLE, 2, 2, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!arr)
        return NULL;

    // Check the dimensions of the array
    if (check_input_array(arr, 2,
                          (const npy_intp[2]){(npy_intp)this->system.block_sizes[row_idx],
                                              (npy_intp)this->system.block_sizes[col_idx]},
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

    u64 total_size = 0;
    for (u64 i = 0; i < this->system.n; ++i)
    {
        total_size += this->system.block_sizes[i];
    }
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
        const u64 n1 = this->system.block_sizes[i];
        u64 offset_col = 0, i_col = 0;
        for (u64 j = 0; j < row->count; ++j)
        {
            const row_entry_t *const entry = row->entries[j];
            while (i_col < entry->col)
            {
                offset_col += this->system.block_sizes[i_col++];
            }
            const u64 n2 = this->system.block_sizes[entry->col];
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
    if (row_idx < 0 || row_idx >= this->system.n)
    {
        PyErr_Format(PyExc_ValueError, "Row index must be in the range [0, %zu), but it was %zd.", this->system.n,
                     row_idx);
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
                 {},
             }},
            {},
        },

};
