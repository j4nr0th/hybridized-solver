/**
 * @file src/python/block_system_type.c
 * Implementation of the :class:`hybsol.BlockSystem` extension type.
 *
 * Every method here is a thin argument-validation shell around one core call;
 * the actual algorithms live in ``src/core``.
 */

#include "block_system_type.h"

#include "decomposition_type.h"
#include "elimination_type.h"

#include "numpy_convert.h"

/* ------------------------------------------------------------------------- */
/* Small helpers                                                              */
/* ------------------------------------------------------------------------- */

static int ensure_block_system_and_state(PyObject *const self, PyTypeObject *const defining_class,
                                         block_system_object **const p_this, const module_state_t **const p_state)
{
    const module_state_t *const state =
        defining_class ? module_state_from_type(defining_class) : module_state_from_type(Py_TYPE(self));
    if (!state)
    {
        return -1;
    }

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

/** Reject unexpected positional/keyword arguments for a method taking none. */
static int no_arguments(const char *const what, const Py_ssize_t nargs, const PyObject *kwnames)
{
    if (nargs != 0 || kwnames != NULL)
    {
        PyErr_Format(PyExc_TypeError, "%s() takes no arguments.", what);
        return -1;
    }
    return 0;
}

static int check_block_indices(const hybsol_system_t *const sys, const Py_ssize_t idx, const char *const name)
{
    const Py_ssize_t n = (Py_ssize_t)hybsol_system_n_blocks(sys);
    if (idx < 0 || idx >= n)
    {
        PyErr_Format(PyExc_ValueError, "%s must be in the range [0, %zd), but it was %zd.", name, n, idx);
        return -1;
    }
    return 0;
}

/**
 * Reject a block the system does not store. The core asserts on this rather
 * than returning a code, so lookups must confirm the block is present first.
 */
static int require_block(const hybsol_system_t *const sys, const Py_ssize_t row, const Py_ssize_t col)
{
    if (hybsol_system_has_block(sys, (uint64_t)row, (uint64_t)col))
    {
        return 0;
    }
    PyErr_Format(PyExc_ValueError, "The system does not contain the block (%zd, %zd).", row, col);
    return -1;
}

/**
 * Reject a system that does not satisfy the solver's structural assumptions.
 * The core asserts on this rather than returning a code, so anything walking
 * the pattern has to check it first.
 */
static int require_valid_system(const hybsol_system_t *const sys, const char *const what)
{
    if (hybsol_system_is_valid(sys))
    {
        return 0;
    }
    PyErr_Format(PyExc_ValueError,
                 "%s: the block system is not valid. Every row needs its diagonal block, and every block "
                 "below the diagonal needs its mirror; is_valid() says which.",
                 what);
    return -1;
}

/** Reject a method that would relocate blocks a caller may still be writing to. */
static int ensure_no_live_views(const block_system_object *const self, const char *const what)
{
    if (self->n_live_views == 0)
    {
        return 0;
    }
    PyErr_Format(PyExc_RuntimeError,
                 "%s() cannot run while %zd array(s) returned by block_storage() are still alive; "
                 "delete them first.",
                 what, self->n_live_views);
    return -1;
}

/* ------------------------------------------------------------------------- */
/* Live views into block storage                                             */
/* ------------------------------------------------------------------------- */

#define BLOCK_VIEW_CAPSULE_NAME "hybsol.BlockSystem.block_storage_view"

/**
 * Release the view count a handed-out array was holding.
 *
 * Registered as a PyCapsule destructor, so it runs when the array the capsule
 * is the base object of is collected. The capsule also keeps the owning
 * system alive, which is what stops the storage from being freed underneath
 * the array.
 */
static void block_view_destructor(PyObject *const capsule)
{
    block_system_object *const owner = PyCapsule_GetPointer(capsule, BLOCK_VIEW_CAPSULE_NAME);
    if (owner == NULL)
    {
        PyErr_Clear();
        return;
    }
    owner->n_live_views -= 1;
    Py_DECREF(owner);
}

/** Create the base object for one array returned by :meth:`block_storage`. */
static PyObject *block_view_capsule(block_system_object *const owner)
{
    PyObject *const capsule = PyCapsule_New(owner, BLOCK_VIEW_CAPSULE_NAME, block_view_destructor);
    if (capsule == NULL)
    {
        return NULL;
    }
    Py_INCREF(owner);
    owner->n_live_views += 1;
    return capsule;
}

/**
 * Validate an ``out=`` keyword that may be ``None``.
 *
 * Done by hand because cpyutl's ``type_check`` rejects ``None`` outright,
 * while these methods document "if not given (or ``None``), a new array is
 * created".
 */
#define optional_array(obj, out) hybsol_optional_array((obj), (out))

/**
 * Turn ``new_order`` into a validated ``uint64`` array of length ``n``.
 *
 * The entries must be a permutation of ``[0, n)``, a precondition of every
 * core function taking one, so it is checked here rather than left to abort
 * there.
 */
static int get_order_array(PyObject *const obj, const Py_ssize_t n, PyArrayObject **const arr_out)
{
    const npy_intp dim = n;
    if (hybsol_index_array(obj, 1, &dim, "new_order", arr_out) < 0)
    {
        return -1;
    }

    const uint64_t *const order = (const uint64_t *)PyArray_DATA(*arr_out);
    char *const seen = PyMem_Calloc(n > 0 ? (size_t)n : 1, sizeof(*seen));
    if (seen == NULL)
    {
        Py_DECREF(*arr_out);
        return -1;
    }

    for (Py_ssize_t i = 0; i < n; ++i)
    {
        if (order[i] >= (uint64_t)n || seen[order[i]])
        {
            PyErr_Format(PyExc_ValueError,
                         "Argument \"new_order\" must be a permutation of [0, %zd), but it repeats or exceeds "
                         "%llu at entry %zd.",
                         n, (unsigned long long)order[i], i);
            PyMem_Free(seen);
            Py_DECREF(*arr_out);
            return -1;
        }
        seen[order[i]] = 1;
    }

    PyMem_Free(seen);
    return 0;
}

/**
 * Read ``*block_sizes`` out of any sequence of positive integers.
 *
 * ``*p_sizes`` is a ``PyMem_Malloc``-ed array the caller must free.
 */
static int parse_block_sizes(PyObject *const obj, uint64_t **const p_sizes, Py_ssize_t *const p_n)
{
    PyObject *const seq = PySequence_Fast(obj, "block_sizes must be a sequence of positive integers.");
    if (!seq)
    {
        return -1;
    }

    const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    if (n == 0)
    {
        PyErr_SetString(PyExc_ValueError, "BlockSystem requires at least one block");
        Py_DECREF(seq);
        return -1;
    }

    uint64_t *const sizes = PyMem_Malloc(sizeof(uint64_t) * (size_t)n);
    if (!sizes)
    {
        Py_DECREF(seq);
        PyErr_NoMemory();
        return -1;
    }

    for (Py_ssize_t i = 0; i < n; ++i)
    {
        const Py_ssize_t size = PyNumber_AsSsize_t(PySequence_Fast_GET_ITEM(seq, i), PyExc_ValueError);
        if (PyErr_Occurred())
        {
            PyMem_Free(sizes);
            Py_DECREF(seq);
            return -1;
        }
        if (size <= 0)
        {
            PyErr_Format(PyExc_ValueError, "Block size of block %zd is %zd.", i, size);
            PyMem_Free(sizes);
            Py_DECREF(seq);
            return -1;
        }
        sizes[i] = (uint64_t)size;
    }

    Py_DECREF(seq);
    *p_sizes = sizes;
    *p_n = n;
    return 0;
}

/**
 * Read a ``precision=`` argument.
 *
 * :class:`hybsol.Precision` is a ``StrEnum``, so its members *are* strings and
 * a plain ``"single"`` works too; anything else is rejected rather than
 * quietly defaulting to double.
 */
static int parse_precision(PyObject *const obj, hybsol_precision_t *const out)
{
    if (!PyUnicode_Check(obj))
    {
        PyErr_Format(PyExc_TypeError, "precision must be a hybsol.Precision member, got %R.", Py_TYPE(obj));
        return -1;
    }
    const char *const name = PyUnicode_AsUTF8(obj);
    if (!name)
    {
        return -1;
    }

    if (strcmp(name, "double") == 0)
    {
        *out = HYBSOL_PRECISION_DOUBLE;
    }
    else if (strcmp(name, "single") == 0)
    {
        *out = HYBSOL_PRECISION_SINGLE;
    }
    else
    {
        PyErr_Format(PyExc_ValueError, "precision must be 'double' or 'single', got '%s'.", name);
        return -1;
    }

    return 0;
}

/** Allocate a Python ``BlockSystem`` of ``n`` blocks with the given sizes. */
static PyObject *block_system_alloc(PyTypeObject *const type, const Py_ssize_t n, const uint64_t *const sizes,
                                    const hybsol_precision_t precision)
{
    block_system_object *const self = (block_system_object *)type->tp_alloc(type, 0);
    if (!self)
    {
        return NULL;
    }
    self->system = NULL;
    self->n_live_views = 0;

    const hybsol_result_t res =
        hybsol_system_create_with_precision((uint64_t)n, sizes, precision, &self->system, &CUTL_STD_ALLOCATOR);
    if (res != HYBSOL_SUCCESS)
    {
        Py_DECREF(self);
        return hybsol_raise(Py_TYPE(self), "BlockSystem", res);
    }
    return (PyObject *)self;
}

/* ------------------------------------------------------------------------- */
/* Construction and destruction                                               */
/* ------------------------------------------------------------------------- */

static PyObject *block_system_new(PyTypeObject *const subtype, PyObject *const args, PyObject *const kwds)
{
    PyObject *precision_obj = NULL;
    if (kwds != NULL && PyDict_GET_SIZE(kwds) != 0)
    {
        precision_obj = PyDict_GetItemString(kwds, "precision");
        if (precision_obj == NULL)
        {
            PyErr_SetString(PyExc_TypeError, "BlockSystem takes no keyword argument other than 'precision'");
            return NULL;
        }
        if (PyDict_GET_SIZE(kwds) != 1)
        {
            PyErr_SetString(PyExc_TypeError, "BlockSystem takes only the 'precision' keyword argument");
            return NULL;
        }
    }

    hybsol_precision_t precision = HYBSOL_PRECISION_DOUBLE;
    if (precision_obj != NULL && parse_precision(precision_obj, &precision) < 0)
    {
        return NULL;
    }

    if (module_state_from_type(subtype) == NULL)
    {
        return NULL;
    }

    uint64_t *sizes = NULL;
    Py_ssize_t n = 0;
    if (parse_block_sizes(args, &sizes, &n) < 0)
    {
        return NULL;
    }

    PyObject *const res = block_system_alloc(subtype, n, sizes, precision);
    PyMem_Free(sizes);
    return res;
}

static void block_system_dealloc(block_system_object *const self)
{
    PyObject_GC_UnTrack(self);
    PyTypeObject *const type = Py_TYPE(self);
    hybsol_system_destroy(self->system);
    self->system = NULL;
    type->tp_free((PyObject *)self);
    Py_DECREF(type);
}

static PyObject *block_system_repr(block_system_object *const self)
{
    return PyUnicode_FromFormat("<hybsol.BlockSystem n_blocks=%zd, size=%zd>",
                                (Py_ssize_t)hybsol_system_n_blocks(self->system),
                                (Py_ssize_t)hybsol_system_total_size(self->system));
}

/* ------------------------------------------------------------------------- */
/* Queries                                                                    */
/* ------------------------------------------------------------------------- */

PyDoc_STRVAR(block_system_docstring, "BlockSystem(*block_sizes: int, precision: hybsol.Precision = Precision.DOUBLE)\n"
                                     "Block system for hybridized solver.\n"
                                     "\n"
                                     "An ``n x n`` arrangement of blocks, where block ``(i, j)`` has shape\n"
                                     "``(block_sizes[i], block_sizes[j])``. Not every block needs to be\n"
                                     "stored: the fill-in supplies the rest. Every diagonal block and every\n"
                                     "mirror of a block below the diagonal *must* be stored;\n"
                                     ":meth:`is_valid` reports whether the structure is decomposable.\n"
                                     "\n"
                                     "``precision`` fixes the type blocks are stored in and computed in for\n"
                                     "the life of the system.\n");

static PyObject *block_system_object_is_valid(PyObject *const self, PyTypeObject *const defining_class,
                                              PyObject *const *const Py_UNUSED(args), const Py_ssize_t nargs,
                                              const PyObject *kwnames)
{
    block_system_object *this;
    const module_state_t *state;
    if (ensure_block_system_and_state(self, defining_class, &this, &state) < 0)
    {
        return NULL;
    }
    if (no_arguments("is_valid", nargs, kwnames) < 0)
    {
        return NULL;
    }

    return PyBool_FromLong(hybsol_system_is_valid(this->system));
}

PyDoc_STRVAR(block_system_object_is_valid_docstring,
             "is_valid() -> bool\n"
             "Check if the system has symmetric sparsity and has diagonal blocks.\n");

static PyObject *block_system_object_as_array(PyObject *const self, PyTypeObject *const defining_class,
                                              PyObject *const *const Py_UNUSED(args), const Py_ssize_t nargs,
                                              const PyObject *kwnames)
{
    block_system_object *this;
    const module_state_t *state;
    if (ensure_block_system_and_state(self, defining_class, &this, &state) < 0)
    {
        return NULL;
    }
    if (no_arguments("as_array", nargs, kwnames) < 0)
    {
        return NULL;
    }

    const int single = hybsol_system_precision(this->system) == HYBSOL_PRECISION_SINGLE;
    const npy_intp total = (npy_intp)hybsol_system_total_size(this->system);
    const npy_intp dims[2] = {total, total};
    PyArrayObject *const arr = (PyArrayObject *)PyArray_SimpleNew(2, dims, single ? NPY_FLOAT : NPY_DOUBLE);
    if (!arr)
    {
        return NULL;
    }

    hybsol_result_t res;
    Py_BEGIN_ALLOW_THREADS;
    if (single)
    {
        res = hybsol_system_to_dense_f32(this->system, PyArray_DATA(arr));
    }
    else
    {
        res = hybsol_system_to_dense(this->system, PyArray_DATA(arr));
    }
    Py_END_ALLOW_THREADS;

    if (res != HYBSOL_SUCCESS)
    {
        Py_DECREF(arr);
        return hybsol_raise(Py_TYPE(self), "as_array", res);
    }
    return (PyObject *)arr;
}

PyDoc_STRVAR(block_system_object_as_array_docstring,
             "as_array() -> numpy.typing.NDArray[numpy.double] | numpy.typing.NDArray[numpy.float]\n"
             "Return the system representation as a full matrix.\n"
             "\n"
             "Blocks the system does not store come out zero, and the result has the\n"
             "system's own :attr:`precision` -- ``float32`` for a single-precision\n"
             "system, ``float64`` otherwise.\n");

static PyObject *block_system_object_get_row_block_indices(PyObject *const self, PyTypeObject *const defining_class,
                                                           PyObject *const *const args, const Py_ssize_t nargs,
                                                           const PyObject *kwnames)
{
    block_system_object *this;
    const module_state_t *state;
    if (ensure_block_system_and_state(self, defining_class, &this, &state) < 0)
    {
        return NULL;
    }

    Py_ssize_t row_idx;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &row_idx, .kwname = "row"},
                {},
            },
            args, nargs, kwnames) < 0)
    {
        return NULL;
    }
    if (check_block_indices(this->system, row_idx, "Row index") < 0)
    {
        return NULL;
    }

    const uint64_t row = (uint64_t)row_idx;
    const uint64_t count = hybsol_system_row_count(this->system, row);
    PyObject *const ret = PyTuple_New((Py_ssize_t)count);
    if (!ret)
    {
        return NULL;
    }
    if (count == 0)
    {
        return ret;
    }

    uint64_t *const indices = PyMem_Malloc(sizeof(uint64_t) * (size_t)count);
    if (!indices)
    {
        Py_DECREF(ret);
        return PyErr_NoMemory();
    }

    uint64_t n_written = 0;
    const hybsol_result_t res = hybsol_system_row_indices(this->system, row, indices, count, &n_written);
    if (res != HYBSOL_SUCCESS)
    {
        PyMem_Free(indices);
        Py_DECREF(ret);
        return hybsol_raise(Py_TYPE(self), "get_row_block_indices", res);
    }

    for (uint64_t i = 0; i < n_written; ++i)
    {
        PyObject *const idx = PyLong_FromUnsignedLongLong(indices[i]);
        if (!idx)
        {
            PyMem_Free(indices);
            Py_DECREF(ret);
            return NULL;
        }
        PyTuple_SET_ITEM(ret, (Py_ssize_t)i, idx);
    }
    PyMem_Free(indices);
    return ret;
}

PyDoc_STRVAR(block_system_object_get_row_block_indices_docstring,
             "get_row_block_indices(row: int) -> tuple[int, ...]\n"
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
             "    Indices of columns that appear in the row, in increasing order.\n");

static PyObject *block_system_object_get_block(PyObject *const self, PyTypeObject *const defining_class,
                                               PyObject *const *const args, const Py_ssize_t nargs,
                                               const PyObject *kwnames)
{
    block_system_object *this;
    const module_state_t *state;
    if (ensure_block_system_and_state(self, defining_class, &this, &state) < 0)
    {
        return NULL;
    }

    Py_ssize_t row_idx, col_idx;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &row_idx, .kwname = "row"},
                {.type = CPYARG_TYPE_SSIZE, .p_val = &col_idx, .kwname = "col"},
                {},
            },
            args, nargs, kwnames) < 0)
    {
        return NULL;
    }
    if (check_block_indices(this->system, row_idx, "Row index") < 0 ||
        check_block_indices(this->system, col_idx, "Col index") < 0 ||
        require_block(this->system, row_idx, col_idx) < 0)
    {
        return NULL;
    }

    if (hybsol_system_precision(this->system) == HYBSOL_PRECISION_SINGLE)
    {
        hybsol_fmatrix_t mat;
        const hybsol_result_t res =
            hybsol_system_get_block_f32(this->system, (uint64_t)row_idx, (uint64_t)col_idx, &mat);
        if (res != HYBSOL_SUCCESS)
        {
            return hybsol_raise(Py_TYPE(self), "get_block", res);
        }
        return (PyObject *)hybsol_fmatrix_to_array(&mat);
    }

    hybsol_matrix_t mat;
    const hybsol_result_t res = hybsol_system_get_block(this->system, (uint64_t)row_idx, (uint64_t)col_idx, &mat);

    if (res != HYBSOL_SUCCESS)
    {
        return hybsol_raise(Py_TYPE(self), "get_block", res);
    }

    return (PyObject *)hybsol_matrix_to_array(&mat);
}

PyDoc_STRVAR(block_system_object_get_block_docstring,
             "get_block(row: int, col: int) -> numpy.typing.NDArray[numpy.double] | "
             "numpy.typing.NDArray[numpy.float]\n"
             "Get a copy of the value of the specified block.\n"
             "\n"
             "The copy has the system's own :attr:`precision`, so it is ``float32``\n"
             "for a single-precision system and ``float64`` otherwise.\n"
             "\n"
             "Raises\n"
             "------\n"
             "ValueError\n"
             "    An index is outside ``[0, n_blocks)``, or the system does not\n"
             "    store that block.\n");

static PyObject *block_system_object_block_storage(PyObject *const self, PyTypeObject *const defining_class,
                                                   PyObject *const *const args, const Py_ssize_t nargs,
                                                   const PyObject *kwnames)
{
    block_system_object *this;
    const module_state_t *state;
    if (ensure_block_system_and_state(self, defining_class, &this, &state) < 0)
    {
        return NULL;
    }

    Py_ssize_t row_idx, col_idx;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &row_idx, .kwname = "row"},
                {.type = CPYARG_TYPE_SSIZE, .p_val = &col_idx, .kwname = "col"},
                {},
            },
            args, nargs, kwnames) < 0)
    {
        return NULL;
    }
    if (check_block_indices(this->system, row_idx, "Row index") < 0 ||
        check_block_indices(this->system, col_idx, "Col index") < 0)
    {
        return NULL;
    }

    PyObject *const capsule = block_view_capsule(this);
    if (capsule == NULL)
    {
        return NULL;
    }

    // The capsule is handed over either way: on failure the wrap releases it,
    // which runs the destructor and drops the live-view count.
    if (hybsol_system_precision(this->system) == HYBSOL_PRECISION_SINGLE)
    {
        hybsol_fmatrix_t mat;
        const hybsol_result_t res =
            hybsol_system_block_storage_f32(this->system, (uint64_t)row_idx, (uint64_t)col_idx, &mat);
        if (res != HYBSOL_SUCCESS)
        {
            Py_DECREF(capsule);
            return hybsol_raise(Py_TYPE(self), "block_storage", res);
        }
        return (PyObject *)hybsol_fmatrix_wrap(&mat, capsule);
    }

    hybsol_matrix_t mat;
    const hybsol_result_t res = hybsol_system_block_storage(this->system, (uint64_t)row_idx, (uint64_t)col_idx, &mat);
    if (res != HYBSOL_SUCCESS)
    {
        Py_DECREF(capsule);
        return hybsol_raise(Py_TYPE(self), "block_storage", res);
    }
    return (PyObject *)hybsol_matrix_wrap(&mat, capsule);
}

PyDoc_STRVAR(block_system_object_block_storage_docstring,
             "block_storage(row: int, col: int) -> numpy.typing.NDArray[numpy.double] | "
             "numpy.typing.NDArray[numpy.float]\n"
             "Get writable storage for a block, creating it on first use.\n"
             "\n"
             "The shape is implied by the system, so this hands back the right buffer to\n"
             "fill in place. The first call adds the block to the sparsity pattern and\n"
             "zeroes it; later calls return the same buffer.\n"
             "\n"
             "The buffer has the system's own :attr:`precision`, so it is ``float32``\n"
             "for a single-precision system and ``float64`` otherwise.\n"
             "\n"
             "The array is a view backed by the system, which it keeps alive. Value\n"
             "rewrites through the system are visible through it, but\n"
             ":meth:`eliminate_row` and :meth:`reorder_blocks` refuse to run while\n"
             "such an array is alive. :meth:`decompose` does not: the decomposition\n"
             "copies what it needs and leaves the system, and its storage, alone.\n"
             "\n"
             "Raises\n"
             "------\n"
             "ValueError\n"
             "    An index is outside ``[0, n_blocks)``.\n"
             "RuntimeError\n"
             "    A previously returned view is still alive in front of\n"
             "    :meth:`eliminate_row` or :meth:`reorder_blocks`.\n");

static PyObject *block_system_object_get_block_size(PyObject *const self, PyTypeObject *const defining_class,
                                                    PyObject *const *const args, const Py_ssize_t nargs,
                                                    const PyObject *kwnames)
{
    block_system_object *this;
    const module_state_t *state;
    if (ensure_block_system_and_state(self, defining_class, &this, &state) < 0)
    {
        return NULL;
    }

    Py_ssize_t row_idx, col_idx;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &row_idx, .kwname = "row"},
                {.type = CPYARG_TYPE_SSIZE, .p_val = &col_idx, .kwname = "col"},
                {},
            },
            args, nargs, kwnames) < 0)
    {
        return NULL;
    }
    if (check_block_indices(this->system, row_idx, "Row index") < 0 ||
        check_block_indices(this->system, col_idx, "Col index") < 0)
    {
        return NULL;
    }

    return cpyutl_output_create_check(
        CPYOUT_TYPE_TUPLE, (const cpyutl_output_t[]){
                               {.type = CPYOUT_TYPE_PYINT,
                                .value_int = (Py_ssize_t)hybsol_system_block_size(this->system, (uint64_t)row_idx)},
                               {.type = CPYOUT_TYPE_PYINT,
                                .value_int = (Py_ssize_t)hybsol_system_block_size(this->system, (uint64_t)col_idx)},
                               {},
                           });
}

PyDoc_STRVAR(block_system_object_get_block_size_docstring, "get_block_size(row: int, col: int) -> tuple[int, int]\n"
                                                           "Get the shape ``(rows, cols)`` of a system block.\n");

static PyObject *block_system_object_has_block(PyObject *const self, PyTypeObject *const defining_class,
                                               PyObject *const *const args, const Py_ssize_t nargs,
                                               const PyObject *kwnames)
{
    block_system_object *this;
    const module_state_t *state;
    if (ensure_block_system_and_state(self, defining_class, &this, &state) < 0)
    {
        return NULL;
    }

    Py_ssize_t row_idx, col_idx;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &row_idx, .kwname = "row"},
                {.type = CPYARG_TYPE_SSIZE, .p_val = &col_idx, .kwname = "col"},
                {},
            },
            args, nargs, kwnames) < 0)
    {
        return NULL;
    }
    if (check_block_indices(this->system, row_idx, "Row index") < 0 ||
        check_block_indices(this->system, col_idx, "Column index") < 0)
    {
        return NULL;
    }

    return PyBool_FromLong(hybsol_system_has_block(this->system, (uint64_t)row_idx, (uint64_t)col_idx));
}

PyDoc_STRVAR(block_system_object_has_block_docstring,
             "has_block(row: int, col: int) -> bool\n"
             "Check if the block at row ``row`` and column ``col`` is present.\n");

static PyObject *block_system_object_no_lower_connections(PyObject *const self, PyTypeObject *const defining_class,
                                                          PyObject *const *const Py_UNUSED(args),
                                                          const Py_ssize_t nargs, const PyObject *kwnames)
{
    block_system_object *this;
    const module_state_t *state;
    if (ensure_block_system_and_state(self, defining_class, &this, &state) < 0)
    {
        return NULL;
    }
    if (no_arguments("no_lower_connections", nargs, kwnames) < 0)
    {
        return NULL;
    }

    const npy_intp dim = (npy_intp)hybsol_system_n_blocks(this->system);
    PyArrayObject *const arr = (PyArrayObject *)PyArray_SimpleNew(1, &dim, NPY_BOOL);
    if (!arr)
    {
        return NULL;
    }
    hybsol_system_no_lower_connections(this->system, (uint8_t *)PyArray_DATA(arr));
    return (PyObject *)arr;
}

PyDoc_STRVAR(block_system_object_no_lower_connections_docstring,
             "no_lower_connections() -> numpy.typing.NDArray[numpy.bool]\n"
             "Return an array flagging rows with nothing to the left of their diagonal.\n");

static PyObject *block_system_object_first_column(PyObject *const self, PyTypeObject *const defining_class,
                                                  PyObject *const *const args, const Py_ssize_t nargs,
                                                  const PyObject *kwnames)
{
    block_system_object *this;
    const module_state_t *state;
    if (ensure_block_system_and_state(self, defining_class, &this, &state) < 0)
    {
        return NULL;
    }

    Py_ssize_t row_idx;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &row_idx, .kwname = "row"},
                {},
            },
            args, nargs, kwnames) < 0)
    {
        return NULL;
    }
    if (check_block_indices(this->system, row_idx, "Row index") < 0)
    {
        return NULL;
    }

    uint64_t out = 0;
    const hybsol_result_t res = hybsol_system_first_column(this->system, (uint64_t)row_idx, &out);
    if (res == HYBSOL_ERROR_EMPTY_ROW)
    {
        PyErr_Format(PyExc_ValueError, "Row %zd has no entries.", row_idx);
        return NULL;
    }
    if (res != HYBSOL_SUCCESS)
    {
        return hybsol_raise(Py_TYPE(self), "first_column", res);
    }
    return PyLong_FromUnsignedLongLong(out);
}

PyDoc_STRVAR(block_system_object_first_column_docstring, "first_column(row: int) -> int\n"
                                                         "Return the index of the first index in a non-empty row.\n");

static PyObject *block_system_object_get_next_column_index(PyObject *const self, PyTypeObject *const defining_class,
                                                           PyObject *const *const args, const Py_ssize_t nargs,
                                                           const PyObject *kwnames)
{
    block_system_object *this;
    const module_state_t *state;
    if (ensure_block_system_and_state(self, defining_class, &this, &state) < 0)
    {
        return NULL;
    }

    Py_ssize_t row_idx, col_idx;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &row_idx, .kwname = "row"},
                {.type = CPYARG_TYPE_SSIZE, .p_val = &col_idx, .kwname = "col"},
                {},
            },
            args, nargs, kwnames) < 0)
    {
        return NULL;
    }
    if (check_block_indices(this->system, row_idx, "Row index") < 0 ||
        check_block_indices(this->system, col_idx, "Column index") < 0)
    {
        return NULL;
    }

    uint64_t out = 0;
    const hybsol_result_t res = hybsol_system_next_column(this->system, (uint64_t)row_idx, (uint64_t)col_idx, &out);
    if (res == HYBSOL_ERROR_EMPTY_ROW)
    {
        PyErr_Format(PyExc_ValueError, "Row %zd has no entries.", row_idx);
        return NULL;
    }
    if (res == HYBSOL_ERROR_NO_MORE_COLUMNS)
    {
        PyErr_SetString(PyExc_ValueError, "Row has no entries after column index.");
        return NULL;
    }
    if (res != HYBSOL_SUCCESS)
    {
        return hybsol_raise(Py_TYPE(self), "get_next_column_index", res);
    }
    return PyLong_FromUnsignedLongLong(out);
}

PyDoc_STRVAR(block_system_object_get_next_column_index_docstring, "get_next_column_index(row: int, col: int) -> int\n"
                                                                  "Get the column index after the current one.\n");

/* ------------------------------------------------------------------------- */
/* Assembly                                                                   */
/* ------------------------------------------------------------------------- */

PyDoc_STRVAR(block_system_add_block_docstring,
             "add_block(row: int, col: int, val: numpy.typing.ArrayLike) -> None\n"
             "Add a block to the system.\n"
             "\n"
             "If the block is already present the values are summed into it.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "row : int\n"
             "    Row index of the block.\n"
             "col : int\n"
             "    Column index of the block.\n"
             "val : array_like\n"
             "    Value of the block, with shape ``(block_sizes[row], block_sizes[col])``.\n");

static PyObject *block_system_object_add_block(PyObject *const self, PyTypeObject *const defining_class,
                                               PyObject *const *const args, const Py_ssize_t nargs,
                                               const PyObject *kwnames)
{
    block_system_object *this;
    const module_state_t *state;
    if (ensure_block_system_and_state(self, defining_class, &this, &state) < 0)
    {
        return NULL;
    }

    Py_ssize_t row_idx, col_idx;
    PyObject *py_val;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &row_idx, .kwname = "row"},
                {.type = CPYARG_TYPE_SSIZE, .p_val = &col_idx, .kwname = "col"},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &py_val, .kwname = "val"},
                {},
            },
            args, nargs, kwnames) < 0)
    {
        return NULL;
    }
    if (check_block_indices(this->system, row_idx, "Row index") < 0 ||
        check_block_indices(this->system, col_idx, "Col index") < 0)
    {
        return NULL;
    }

    const uint64_t n_rows = hybsol_system_block_size(this->system, (uint64_t)row_idx);
    const uint64_t n_cols = hybsol_system_block_size(this->system, (uint64_t)col_idx);
    const npy_intp dims[2] = {(npy_intp)n_rows, (npy_intp)n_cols};

    const int single = hybsol_system_precision(this->system) == HYBSOL_PRECISION_SINGLE;

    PyArrayObject *arr = NULL;
    if ((single ? hybsol_float_array : hybsol_double_array)(py_val, 2, dims, "val", &arr) < 0)
    {
        return NULL;
    }

    const hybsol_result_t res = single ? hybsol_system_add_block_f32(this->system, (uint64_t)row_idx, (uint64_t)col_idx,
                                                                     n_rows, n_cols, (const float *)PyArray_DATA(arr))
                                       : hybsol_system_add_block(this->system, (uint64_t)row_idx, (uint64_t)col_idx,
                                                                 n_rows, n_cols, (const double *)PyArray_DATA(arr));
    Py_DECREF(arr);
    if (res != HYBSOL_SUCCESS)
    {
        return hybsol_raise(Py_TYPE(self), "add_block", res);
    }
    Py_RETURN_NONE;
}

PyDoc_STRVAR(block_system_object_add_blocks_docstring,
             "add_blocks(rows: numpy.typing.ArrayLike, cols: numpy.typing.ArrayLike, "
             "data: numpy.typing.ArrayLike) -> None\n"
             "Add many blocks in a single pass.\n"
             "\n"
             "The fast assembly path: the index arrays are processed once and each row\n"
             "allocates its storage a single time. Duplicate ``(row, col)`` pairs\n"
             "accumulate, as repeated :meth:`add_block` calls would.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "rows : array_like\n"
             "    Row index of every block being added.\n"
             "cols : array_like\n"
             "    Column index of every block being added.\n"
             "data : array_like\n"

             "    Concatenated, row-major block values; block ``k`` occupies\n"
             "    ``block_sizes[rows[k]] * block_sizes[cols[k]]`` entries.\n"
             "\n"
             "    A block and its transpose share a row-major ravel *only* when one of\n"
             "    them has a single column.\n"
             "");

static PyObject *block_system_object_add_blocks(PyObject *const self, PyTypeObject *const defining_class,
                                                PyObject *const *const args, const Py_ssize_t nargs,
                                                const PyObject *kwnames)
{
    block_system_object *this;
    const module_state_t *state;
    if (ensure_block_system_and_state(self, defining_class, &this, &state) < 0)
    {
        return NULL;
    }

    PyObject *py_rows, *py_cols, *py_data;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = &py_rows, .kwname = "rows"},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &py_cols, .kwname = "cols"},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &py_data, .kwname = "data"},
                {},
            },
            args, nargs, kwnames) < 0)
    {
        return NULL;
    }

    const npy_intp any_len = 0;
    PyArrayObject *rows_arr = NULL, *cols_arr = NULL, *data_arr = NULL;
    if (hybsol_index_array(py_rows, 1, &any_len, "rows", &rows_arr) < 0)
    {
        return NULL;
    }

    const npy_intp n_entries = PyArray_DIM(rows_arr, 0);
    const npy_intp dims_cols[1] = {n_entries};
    if (hybsol_index_array(py_cols, 1, dims_cols, "cols", &cols_arr) < 0)
    {
        Py_DECREF(rows_arr);
        return NULL;
    }
    const int single = hybsol_system_precision(this->system) == HYBSOL_PRECISION_SINGLE;
    if ((single ? hybsol_float_array : hybsol_double_array)(py_data, 1, &any_len, "data", &data_arr) < 0)
    {
        Py_DECREF(rows_arr);
        Py_DECREF(cols_arr);
        return NULL;
    }

    const Py_ssize_t n = (Py_ssize_t)hybsol_system_n_blocks(this->system);
    if (n_entries == 0)
    {
        // Nothing to do, and the core's `static n_entries` parameters must
        // not be handed a zero-length array.
        Py_DECREF(rows_arr);
        Py_DECREF(cols_arr);
        Py_DECREF(data_arr);
        Py_RETURN_NONE;
    }

    const uint64_t *const rows = (const uint64_t *)PyArray_DATA(rows_arr);
    const uint64_t *const cols = (const uint64_t *)PyArray_DATA(cols_arr);

    // The core validates the indices too, but the size of `data` has to be
    // known before it is called, so the check is duplicated here.
    size_t needed = 0;
    for (Py_ssize_t k = 0; k < (Py_ssize_t)n_entries; ++k)
    {
        if (rows[k] >= (uint64_t)n || cols[k] >= (uint64_t)n)
        {
            PyErr_Format(PyExc_ValueError, "Block index %llu (entry %zd) is not in the range [0, %zd).",
                         (unsigned long long)(rows[k] >= (uint64_t)n ? rows[k] : cols[k]), k, n);
            Py_DECREF(rows_arr);
            Py_DECREF(cols_arr);
            Py_DECREF(data_arr);
            return NULL;
        }
        needed += (size_t)hybsol_system_block_size(this->system, rows[k]) *
                  (size_t)hybsol_system_block_size(this->system, cols[k]);
    }

    if ((size_t)PyArray_SIZE(data_arr) != needed)
    {
        PyErr_Format(PyExc_ValueError, "data holds %zd values, but the given rows and cols need %zu.",
                     (Py_ssize_t)PyArray_SIZE(data_arr), needed);
        Py_DECREF(rows_arr);
        Py_DECREF(cols_arr);
        Py_DECREF(data_arr);
        return NULL;
    }

    hybsol_result_t res;
    Py_BEGIN_ALLOW_THREADS;
    if (single)
    {
        res = hybsol_system_add_blocks_f32(this->system, (uint64_t)n_entries, rows, cols,
                                           (const float *)PyArray_DATA(data_arr));
    }
    else
    {
        res = hybsol_system_add_blocks(this->system, (uint64_t)n_entries, rows, cols,
                                       (const double *)PyArray_DATA(data_arr));
    }
    Py_END_ALLOW_THREADS;

    Py_DECREF(rows_arr);
    Py_DECREF(cols_arr);
    Py_DECREF(data_arr);

    if (res != HYBSOL_SUCCESS)
    {
        return hybsol_raise(Py_TYPE(self), "add_blocks", res);
    }
    Py_RETURN_NONE;
}

PyDoc_STRVAR(block_system_object_reserve_docstring,
             "reserve(row: int, capacity: int) -> None\n"
             "Make sure the row can hold ``capacity`` blocks without reallocating.\n"
             "\n"
             "Assembly loops that know a row's block count should call this once per row.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "row : int\n"
             "    Row index to reserve space for.\n"
             "capacity : int\n"
             "    Number of blocks the row should be able to hold.\n");

static PyObject *block_system_object_reserve(PyObject *const self, PyTypeObject *const defining_class,
                                             PyObject *const *const args, const Py_ssize_t nargs,
                                             const PyObject *kwnames)
{
    block_system_object *this;
    const module_state_t *state;
    if (ensure_block_system_and_state(self, defining_class, &this, &state) < 0)
    {
        return NULL;
    }

    Py_ssize_t row_idx, capacity;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &row_idx, .kwname = "row"},
                {.type = CPYARG_TYPE_SSIZE, .p_val = &capacity, .kwname = "capacity"},
                {},
            },
            args, nargs, kwnames) < 0)
    {
        return NULL;
    }
    if (check_block_indices(this->system, row_idx, "Row index") < 0)
    {
        return NULL;
    }
    if (capacity < 0)
    {
        PyErr_SetString(PyExc_ValueError, "Capacity must be non-negative.");
        return NULL;
    }

    const hybsol_result_t res = hybsol_system_reserve(this->system, (uint64_t)row_idx, (uint64_t)capacity);
    if (res != HYBSOL_SUCCESS)
    {
        return hybsol_raise(Py_TYPE(self), "reserve", res);
    }
    Py_RETURN_NONE;
}

/* ------------------------------------------------------------------------- */
/* Row operations                                                             */
/* ------------------------------------------------------------------------- */

PyDoc_STRVAR(block_system_object_multiply_row_docstring,
             "multiply_row(row: int, val: numpy.typing.ArrayLike, start: int = 0) -> None\n"
             "Multiply the row by the matrix.\n"
             "\n"
             "Every stored block in the row at or after column ``start`` is replaced by\n"
             "``val @ block``; blocks before ``start`` are left alone.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "row : int\n"
             "    Row index of blocks to multiply.\n"
             "val : array_like\n"
             "    Square matrix with which the row should be multiplied.\n"
             "start : int, default: 0\n"
             "    First block column (inclusive) to transform.\n");

static PyObject *block_system_object_multiply_row(PyObject *const self, PyTypeObject *const defining_class,
                                                  PyObject *const *const args, const Py_ssize_t nargs,
                                                  const PyObject *kwnames)
{
    block_system_object *this;
    const module_state_t *state;
    if (ensure_block_system_and_state(self, defining_class, &this, &state) < 0)
    {
        return NULL;
    }

    Py_ssize_t row_idx, start_idx = 0;
    PyObject *py_val;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &row_idx, .kwname = "row"},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &py_val, .kwname = "val"},
                {.type = CPYARG_TYPE_SSIZE, .p_val = &start_idx, .kwname = "start", .optional = 1},
                {},
            },
            args, nargs, kwnames) < 0)
    {
        return NULL;
    }
    if (check_block_indices(this->system, row_idx, "Row index") < 0 ||
        check_block_indices(this->system, start_idx, "Start index") < 0)
    {
        return NULL;
    }

    const uint64_t n_rows = hybsol_system_block_size(this->system, (uint64_t)row_idx);
    const npy_intp dims[2] = {(npy_intp)n_rows, (npy_intp)n_rows};
    const int single = hybsol_system_precision(this->system) == HYBSOL_PRECISION_SINGLE;

    PyArrayObject *arr = NULL;
    if ((single ? hybsol_float_array : hybsol_double_array)(py_val, 2, dims, "val", &arr) < 0)
    {
        return NULL;
    }

    hybsol_result_t res;
    Py_BEGIN_ALLOW_THREADS;
    if (single)
    {
        const hybsol_fmatrix_t mat = hybsol_fmatrix_from_array(arr);
        res = hybsol_system_multiply_row_f32(this->system, (uint64_t)row_idx, (uint64_t)start_idx, &mat);
    }
    else
    {
        const hybsol_matrix_t mat = hybsol_matrix_from_array(arr);
        res = hybsol_system_multiply_row(this->system, (uint64_t)row_idx, (uint64_t)start_idx, &mat);
    }
    Py_END_ALLOW_THREADS;

    Py_DECREF(arr);
    if (res != HYBSOL_SUCCESS)
    {
        return hybsol_raise(Py_TYPE(self), "multiply_row", res);
    }
    Py_RETURN_NONE;
}

PyDoc_STRVAR(block_system_object_eliminate_row_docstring,
             "eliminate_row(row_src: int, row_tgt: int, val: numpy.typing.ArrayLike) -> None\n"
             "Eliminate a target row using a source row, multiplied by matrix.\n"
             "\n"
             "With source row :math:`\\mathbf{M}_s`, target row :math:`\\mathbf{M}_t` and\n"
             "scaling matrix :math:`\\mathbf{S}`, the new target row is:\n"
             "\n"
             ".. math ::\n"
             "    \\mathbf{M}_t^\\prime = \\mathbf{M}_t - \\mathbf{S} \\mathbf{M}_s\n"
             "\n"
             "Entries in both rows with column index lower than or equal to ``row_src``\n"
             "are ignored, as they are assumed to have been eliminated already.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "row_src : int\n"
             "    Index of the source row.\n"
             "row_tgt : int\n"
             "    Index of the row to eliminate.\n"
             "val : array_like\n"
             "    Matrix used to scale the source row.\n");

static PyObject *block_system_object_eliminate_row(PyObject *const self, PyTypeObject *const defining_class,
                                                   PyObject *const *const args, const Py_ssize_t nargs,
                                                   const PyObject *kwnames)
{
    block_system_object *this;
    const module_state_t *state;
    if (ensure_block_system_and_state(self, defining_class, &this, &state) < 0)
    {
        return NULL;
    }
    if (ensure_no_live_views(this, "eliminate_row") < 0)
    {
        return NULL;
    }

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
    {
        return NULL;
    }
    if (check_block_indices(this->system, i_row_src, "Row index source") < 0 ||
        check_block_indices(this->system, i_row_tgt, "Row index target") < 0 ||
        require_block(this->system, i_row_tgt, i_row_src) < 0)
    {
        if (!PyErr_Occurred())
            PyErr_Format(PyExc_ValueError,
                         "The target row %zd must contain the block in column %zd to eliminate with it.", i_row_tgt,
                         i_row_src);
        return NULL;
    }

    const uint64_t size_tgt = hybsol_system_block_size(this->system, (uint64_t)i_row_tgt);
    const uint64_t size_src = hybsol_system_block_size(this->system, (uint64_t)i_row_src);
    const npy_intp dims[2] = {(npy_intp)size_tgt, (npy_intp)size_src};

    const int single = hybsol_system_precision(this->system) == HYBSOL_PRECISION_SINGLE;

    PyArrayObject *arr = NULL;
    if ((single ? hybsol_float_array : hybsol_double_array)(py_val, 2, dims, "val", &arr) < 0)
    {
        return NULL;
    }

    hybsol_result_t res;
    Py_BEGIN_ALLOW_THREADS;
    if (single)
    {
        const hybsol_fmatrix_t mat = hybsol_fmatrix_from_array(arr);
        res = hybsol_system_eliminate_row_with_f32(this->system, (uint64_t)i_row_tgt, (uint64_t)i_row_src, &mat);
    }
    else
    {
        const hybsol_matrix_t mat = hybsol_matrix_from_array(arr);
        res = hybsol_system_eliminate_row_with(this->system, (uint64_t)i_row_tgt, (uint64_t)i_row_src, &mat);
    }
    Py_END_ALLOW_THREADS;

    Py_DECREF(arr);
    if (res != HYBSOL_SUCCESS)
    {
        return hybsol_raise(Py_TYPE(self), "eliminate_row", res);
    }
    Py_RETURN_NONE;
}

/* ------------------------------------------------------------------------- */
/* Decomposition                                                              */
/* ------------------------------------------------------------------------- */

PyDoc_STRVAR(block_system_object_workspace_bytes_docstring,
             "workspace_bytes(n_threads: int = 0) -> int\n"
             "Bytes of scratch :meth:`decompose` needs at this thread count.\n"
             "\n"
             "Sizes the ``workspace`` array that :meth:`decompose` accepts, so one\n"
             "buffer can be reused. It depends only on the block sizes, the\n"
             "precision and the thread count, so it may be asked for before any\n"
             "blocks are added.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "n_threads : int, default: 0\n"
             "    The thread count that will be passed to :meth:`decompose`.\n"
             "precision : hybsol.Precision, optional\n"
             "    The precision :meth:`decompose` will factor in. Defaults to the\n"
             "    system's own.\n");

static PyObject *block_system_object_workspace_bytes(PyObject *const self, PyTypeObject *const defining_class,
                                                     PyObject *const *const args, const Py_ssize_t nargs,
                                                     PyObject *kwnames)
{
    block_system_object *this;
    const module_state_t *state;
    if (ensure_block_system_and_state(self, defining_class, &this, &state) < 0)
    {
        return NULL;
    }

    Py_ssize_t n_threads = 0;
    PyObject *py_precision = Py_None;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &n_threads, .kwname = "n_threads", .optional = 1},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &py_precision, .kwname = "precision", .optional = 1},
                {},
            },
            args, nargs, kwnames) < 0)
    {
        return NULL;
    }
    if (hybsol_check_n_threads(n_threads) < 0)
    {
        return NULL;
    }

    hybsol_precision_t precision;
    if (py_precision == Py_None)
        precision = hybsol_system_precision(this->system);
    else if (parse_precision(py_precision, &precision) < 0)
        return NULL;

    const size_t bytes = hybsol_workspace_bytes(this->system, precision, (uint64_t)n_threads);
    return PyLong_FromSize_t(bytes);
}

PyDoc_STRVAR(block_system_object_decompose_docstring,
             "decompose(n_threads: int = 0, workspace: numpy.typing.ArrayLike | None = None,\n"
             "           precision: hybsol.Precision | None = None) -> Decomposition\n"
             "Factorize the system and return the result.\n"
             "\n"
             "The system is left exactly as it was assembled, so it can be decomposed\n"
             "again -- under a different block order, or with a different thread count\n"
             "-- and the decompositions are independent.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "n_threads : int, default: 0\n"
             "    Number of OpenMP threads; ``0`` selects the OpenMP default\n"
             "    and ``1`` runs the factorization serially.\n"
             "workspace : array_like, optional\n"
             "    Scratch for the factorization, as a writable 1-D\n"
             "    ``uint8`` array of at least :meth:`workspace_bytes`\n"
             "    bytes. Its contents are overwritten. Omit it and the\n"
             "    scratch is allocated internally.\n"
             "precision : hybsol.Precision, optional\n"
             "    The precision of the factors, which need not match the\n"
             "    system's: a double system can produce single factors and\n"
             "    a single system double ones. Defaults to the system's\n"
             "    own precision. Values convert on the way in, so the\n"
             "    factorization is exact for the precision it runs in, but\n"
             "    the values themselves are still the system's.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "Decomposition\n"
             "    The factorized system.\n"
             "\n"
             "Raises\n"
             "------\n"
             "ValueError\n"
             "    The system does not satisfy the solver's structural\n"
             "    assumptions; a diagonal block holds nothing but zeros,\n"
             "    which no block order can rescue; ``n_threads`` is\n"
             "    negative; or ``workspace`` is too small or read-only.\n"
             "hybsol.SingularSystemError\n"
             "    A diagonal block is singular, so the LU factorization hit\n"
             "    a zero pivot. The message names the block.\n");

static PyObject *block_system_object_decompose(PyObject *const self, PyTypeObject *const defining_class,
                                               PyObject *const *const args, const Py_ssize_t nargs, PyObject *kwnames)
{
    block_system_object *this;
    const module_state_t *state;
    if (ensure_block_system_and_state(self, defining_class, &this, &state) < 0)
    {
        return NULL;
    }

    Py_ssize_t n_threads = 0;
    PyObject *py_workspace = Py_None;
    PyObject *py_precision = Py_None;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &n_threads, .kwname = "n_threads", .optional = 1},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &py_workspace, .kwname = "workspace", .optional = 1},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &py_precision, .kwname = "precision", .optional = 1},
                {},
            },
            args, nargs, kwnames) < 0)
    {
        return NULL;
    }
    if (hybsol_check_n_threads(n_threads) < 0)
    {
        return NULL;
    }

    hybsol_precision_t precision;
    if (py_precision == Py_None)
        precision = hybsol_system_precision(this->system);
    else if (parse_precision(py_precision, &precision) < 0)
        return NULL;

    /* A supplied workspace must be big enough for this system, this precision
     * and this thread count; the buffer is opaque, so all that can be checked
     * here is its size. */
    const size_t needed = hybsol_workspace_bytes(this->system, precision, (uint64_t)n_threads);

    // The core asserts validity rather than reporting it, and raising is the only way out.
    if (require_valid_system(this->system, "decompose") < 0)
    {
        return NULL;
    }

    uint64_t failing_block = UINT64_MAX;
    PyArrayObject *workspace = NULL;
    if (py_workspace != Py_None)
    {
        if (hybsol_byte_array(py_workspace, 1, needed, "workspace", &workspace) < 0)
        {
            return NULL;
        }
    }

    hybsol_elimination_t *graph = NULL;
    hybsol_decomposition_t *dec = NULL;
    hybsol_result_t res;

    Py_BEGIN_ALLOW_THREADS;
    res = hybsol_elimination_create(this->system, &graph, &failing_block);
    if (res == HYBSOL_SUCCESS)
    {
        res = hybsol_decomposition_create_with_precision(this->system, graph, precision, &dec);
        // The decomposition copied the schedule it needs, so the graph is done.
        hybsol_elimination_destroy(graph);
        graph = NULL;
    }
    if (res == HYBSOL_SUCCESS)
        res = workspace != NULL ? hybsol_decomposition_factorize_with_workspace(dec, PyArray_DATA(workspace),
                                                                                (size_t)needed, (uint64_t)n_threads)
                                : hybsol_decomposition_factorize(dec, (uint64_t)n_threads);
    Py_END_ALLOW_THREADS;

    Py_XDECREF(workspace);
    if (res != HYBSOL_SUCCESS)
    {
        hybsol_elimination_destroy(graph);
        hybsol_decomposition_destroy(dec);
        // A rejected order is named by the walk; a singular diagonal by the
        // factorization, the only stage that gets far enough to see one.
        return hybsol_raise_block(Py_TYPE(self), "decompose", res,
                                  dec != NULL ? hybsol_decomposition_failing_block(dec) : failing_block);
    }

    PyObject *const out = decomposition_alloc(state->type_decomposition, dec);
    if (!out)
    {
        hybsol_decomposition_destroy(dec);
        return NULL;
    }
    return out;
}

PyDoc_STRVAR(block_system_object_elimination_docstring,
             "elimination() -> Elimination\n"
             "Walk the elimination graph without computing any values.\n"
             "\n"
             "Reports the pattern the fill-in will produce, the passes a\n"
             "factorization runs in and what it will cost. Cheap enough to ask for\n"
             "while deciding on a block structure, and it touches nothing, so the\n"
             "system may still be assembled.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "Elimination\n"
             "    The symbolic result for this system.\n"
             "\n"
             "Raises\n"
             "------\n"
             "ValueError\n"
             "    The system does not satisfy the solver's structural\n"
             "    assumptions, or a diagonal block holds nothing but zeros,\n"
             "    which no block order can rescue. The message names the\n"
             "    offending block.\n");

static PyObject *block_system_object_elimination(PyObject *const self, PyTypeObject *const defining_class,
                                                 PyObject *const *const Py_UNUSED(args), const Py_ssize_t nargs,
                                                 PyObject *kwnames)
{
    block_system_object *this;
    const module_state_t *state;
    if (ensure_block_system_and_state(self, defining_class, &this, &state) < 0)
    {
        return NULL;
    }
    if (no_arguments("elimination", nargs, kwnames) < 0)
    {
        return NULL;
    }

    // The core asserts validity rather than reporting it, and raising is the only way out.
    if (require_valid_system(this->system, "elimination") < 0)
    {
        return NULL;
    }

    hybsol_elimination_t *graph = NULL;
    uint64_t failing_block = UINT64_MAX;
    hybsol_result_t res;
    Py_BEGIN_ALLOW_THREADS;
    res = hybsol_elimination_create(this->system, &graph, &failing_block);
    Py_END_ALLOW_THREADS;

    if (res != HYBSOL_SUCCESS)
        return hybsol_raise_block(Py_TYPE(self), "elimination", res, failing_block);

    PyObject *const out = elimination_alloc(state->type_elimination, graph);
    if (!out)
    {
        hybsol_elimination_destroy(graph);
        return NULL;
    }
    return out;
}

PyDoc_STRVAR(block_system_object_copy_docstring, "copy() -> BlockSystem\n"
                                                 "Create a deep copy of the system.\n");

static PyObject *block_system_object_copy(PyObject *const self, PyTypeObject *const defining_class,
                                          PyObject *const *const Py_UNUSED(args), const Py_ssize_t nargs,
                                          const PyObject *kwnames)
{
    block_system_object *this;
    const module_state_t *state;
    if (ensure_block_system_and_state(self, defining_class, &this, &state) < 0)
    {
        return NULL;
    }
    if (no_arguments("copy", nargs, kwnames) < 0)
    {
        return NULL;
    }

    hybsol_system_t *dup = NULL;
    hybsol_result_t res;
    Py_BEGIN_ALLOW_THREADS;
    res = hybsol_system_copy(this->system, &dup);
    Py_END_ALLOW_THREADS;

    if (res != HYBSOL_SUCCESS)
    {
        return hybsol_raise(Py_TYPE(self), "copy", res);
    }

    block_system_object *const that =
        (block_system_object *)state->type_block_system->tp_alloc(state->type_block_system, 0);
    if (!that)
    {
        hybsol_system_destroy(dup);
        return NULL;
    }
    that->system = dup;
    return (PyObject *)that;
}

/* ------------------------------------------------------------------------- */
/* Ordering                                                                   */
/* ------------------------------------------------------------------------- */

PyDoc_STRVAR(block_system_object_reorder_blocks_docstring,
             "reorder_blocks(new_order: numpy.typing.ArrayLike, n_threads: int = 0) -> None\n"
             "Reorders blocks of the system to follow the newly specified ordering.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "new_order : array_like\n"
             "    Array of indices specifying where the old values should be moved to.\n"
             "n_threads : int, default: 0\n"
             "    Number of OpenMP threads to use for reordering. ``0`` selects the\n"
             "    OpenMP default and ``1`` reorders serially.\n"
             "\n"
             "Raises\n"
             "------\n"
             "ValueError\n"
             "    ``new_order`` is not a permutation of ``[0, n_blocks)``;\n"
             "    ``n_threads`` is negative; the system does not satisfy the\n"
             "    solver's structural assumptions; or a diagonal block holds\n"
             "    nothing but zeros, which no block order can rescue.\n"
             "RuntimeError\n"
             "    An array returned by :meth:`block_storage` is still alive.\n");

static PyObject *block_system_object_reorder_blocks(PyObject *const self, PyTypeObject *const defining_class,
                                                    PyObject *const *const args, const Py_ssize_t nargs,
                                                    const PyObject *kwnames)
{
    block_system_object *this;
    const module_state_t *state;
    if (ensure_block_system_and_state(self, defining_class, &this, &state) < 0)
    {
        return NULL;
    }
    if (ensure_no_live_views(this, "reorder_blocks") < 0)
    {
        return NULL;
    }

    PyObject *py_order;
    Py_ssize_t n_threads = 0;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = &py_order, .kwname = "new_order"},
                {.type = CPYARG_TYPE_SSIZE, .p_val = &n_threads, .kwname = "n_threads", .optional = 1},
                {},
            },
            args, nargs, kwnames) < 0)
    {
        return NULL;
    }
    if (hybsol_check_n_threads(n_threads) < 0)
    {
        return NULL;
    }

    PyArrayObject *arr = NULL;
    if (get_order_array(py_order, (Py_ssize_t)hybsol_system_n_blocks(this->system), &arr) < 0)
    {
        return NULL;
    }

    // The core asserts validity rather than reporting it, and raising is the only way out.
    if (require_valid_system(this->system, "reorder_blocks") < 0)
    {
        Py_DECREF(arr);
        return NULL;
    }

    uint64_t failing_block = UINT64_MAX;
    hybsol_result_t res;
    Py_BEGIN_ALLOW_THREADS;
    res = hybsol_system_reorder_blocks(this->system, (const uint64_t *)PyArray_DATA(arr), (uint64_t)n_threads,
                                       &failing_block);
    Py_END_ALLOW_THREADS;

    Py_DECREF(arr);
    if (res != HYBSOL_SUCCESS)
        return hybsol_raise_block(Py_TYPE(self), "reorder_blocks", res, failing_block);
    Py_RETURN_NONE;
}

static int parse_ordering_strategy(PyObject *const obj, hybsol_ordering_strategy_t *const out)
{
    if (!PyUnicode_Check(obj))
    {
        PyErr_Format(PyExc_TypeError, "Ordering strategy must be a string, got %R.", Py_TYPE(obj));
        return -1;
    }
    const char *const strategy = PyUnicode_AsUTF8(obj);
    if (!strategy)
    {
        return -1;
    }

    if (strcmp(strategy, "first") == 0)
    {
        *out = HYBSOL_ORDERING_FIRST;
    }
    else if (strcmp(strategy, "greedy") == 0)
    {
        *out = HYBSOL_ORDERING_GREEDY;
    }
    else if (strcmp(strategy, "balanced") == 0)
    {
        *out = HYBSOL_ORDERING_BALANCED;
    }
    else
    {
        PyErr_Format(PyExc_ValueError, "Unknown ordering strategy '%s'.", strategy);
        return -1;
    }

    return 0;
}

PyDoc_STRVAR(block_system_object_compute_reordering_docstring,
             "compute_reordering(strategy: typing.Literal[\"first\", \"greedy\", \"balanced\"] = \"first\", "
             "max_colors: int = 0) -> numpy.typing.NDArray[numpy.uint64]\n"
             "Find a *coloring* of the blocks, grouping them so that no two in a group\n"
             "share a non-zero off-diagonal block.\n"
             "\n"
             "This is **not** a factorization ordering. A coloring makes no promise\n"
             "about which blocks come first, so :meth:`decompose` may fill in far\n"
             "more than it would have to. Choose the permutation yourself, with every\n"
             "block ahead of the blocks it couples to, and pass *that* to\n"
             ":meth:`reorder_blocks`.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "strategy : typing.Literal[\"first\", \"greedy\", \"balanced\"], default: \"first\"\n"
             "    How the grouping is constructed. \"first\" tries to group as many degrees of\n"
             "    freedom into the first available group, while \"greedy\" and \"balanced\" will\n"
             "    use the group with the most or the least others in them respectively.\n"
             "max_colors : int, default: 0\n"
             "    Maximum number of colors allowed for the coloring. If ``0`` is specified,\n"
             "    each unknown may have its own color. If the coloring can not be completed\n"
             "    using this number of colors, an exception will be raised.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "array\n"
             "    Array which specifies new indices for old degrees of freedom. If the\n"
             "    old degree of freedom had index ``i``, then its new index will be in\n"
             "    this array at the same index.\n");

static PyObject *block_system_object_compute_reordering(PyObject *const self, PyTypeObject *const defining_class,
                                                        PyObject *const *const args, const Py_ssize_t nargs,
                                                        const PyObject *kwnames)
{
    block_system_object *this;
    const module_state_t *state;
    if (ensure_block_system_and_state(self, defining_class, &this, &state) < 0)
    {
        return NULL;
    }

    PyObject *py_strategy = NULL;
    hybsol_ordering_strategy_t strategy = HYBSOL_ORDERING_FIRST;
    Py_ssize_t max_colors = 0;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = &py_strategy, .kwname = "strategy", .optional = 1},
                {.type = CPYARG_TYPE_SSIZE, .p_val = &max_colors, .kwname = "max_colors", .optional = 1},
                {},
            },
            args, nargs, kwnames) < 0)
    {
        return NULL;
    }
    if (py_strategy != NULL && parse_ordering_strategy(py_strategy, &strategy) < 0)
    {
        return NULL;
    }
    if (max_colors < 0)
    {
        PyErr_SetString(PyExc_ValueError, "Maximum number of colors must be non-negative.");
        return NULL;
    }

    const npy_intp dim = (npy_intp)hybsol_system_n_blocks(this->system);
    PyArrayObject *const arr = (PyArrayObject *)PyArray_SimpleNew(1, &dim, NPY_UINT64);
    if (!arr)
    {
        return NULL;
    }

    hybsol_result_t res;
    Py_BEGIN_ALLOW_THREADS;
    res = hybsol_system_compute_reordering(this->system, strategy, (uint64_t)max_colors, (uint64_t *)PyArray_DATA(arr));
    Py_END_ALLOW_THREADS;

    if (res != HYBSOL_SUCCESS)
    {
        if (res == HYBSOL_ERROR_MAX_COLORS)
        {
            PyErr_Format(PyExc_ValueError, "Maximum number of colors (%zd) exceeded.", max_colors);
        }
        else
        {
            hybsol_raise(Py_TYPE(self), "compute_reordering", res);
        }
        Py_DECREF(arr);
        return NULL;
    }
    return (PyObject *)arr;
}

/** Shared argument handling of :meth:`reorder_vector` and :meth:`unorder_vector`. */
static int prepare_for_reordering(PyObject *const self, PyTypeObject *const defining_class, PyObject *const *const args,
                                  const Py_ssize_t nargs, const PyObject *kwnames,
                                  const block_system_object **const p_this, PyArrayObject **const p_arr_vector,
                                  PyArrayObject **const p_arr_order, PyArrayObject **const p_out_arr)
{
    block_system_object *this;
    const module_state_t *state;
    if (ensure_block_system_and_state(self, defining_class, &this, &state) < 0)
    {
        return -1;
    }

    PyObject *py_new_order, *py_vector, *py_out = NULL;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = &py_new_order, .kwname = "new_order"},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &py_vector, .kwname = "vector"},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &py_out, .kwname = "out", .optional = 1},
                {},
            },
            args, nargs, kwnames) < 0)
    {
        return -1;
    }

    PyArrayObject *user_out = NULL;
    if (optional_array(py_out, &user_out) < 0)
    {
        return -1;
    }

    PyArrayObject *arr_order = NULL;
    if (get_order_array(py_new_order, (Py_ssize_t)hybsol_system_n_blocks(this->system), &arr_order) < 0)
    {
        return -1;
    }

    const npy_intp dim = (npy_intp)hybsol_system_total_size(this->system);
    PyArrayObject *arr_vector = NULL;
    if (hybsol_double_array(py_vector, 1, &dim, "vector", &arr_vector) < 0)
    {
        Py_DECREF(arr_order);
        return -1;
    }

    PyArrayObject *out_arr = NULL;
    if (hybsol_prepare_output(user_out, 1, &dim, NPY_DOUBLE, 0, "out", &out_arr) < 0)
    {
        Py_DECREF(arr_order);
        Py_DECREF(arr_vector);
        return -1;
    }

    if (PyArray_DATA(out_arr) == PyArray_DATA(arr_vector))
    {
        PyErr_SetString(PyExc_ValueError, "out must not be the same array as vector.");
        Py_DECREF(arr_order);
        Py_DECREF(arr_vector);
        Py_DECREF(out_arr);
        return -1;
    }

    *p_this = this;
    *p_arr_vector = arr_vector;
    *p_arr_order = arr_order;
    *p_out_arr = out_arr;
    return 0;
}

PyDoc_STRVAR(block_system_object_reorder_vector_docstring,
             "reorder_vector(new_order: numpy.typing.ArrayLike, vector: numpy.typing.ArrayLike, out: "
             "numpy.typing.NDArray[numpy.double] | None = None) -> numpy.typing.NDArray[numpy.double]\n"
             "Reorder the vector based on new block ordering.\n"
             "\n"
             "The block that was at slot ``i`` is moved to slot ``new_order[i]``, so\n"
             "the result is the vector of the system as :meth:`reorder_blocks` left\n"
             "it. The method is meant to be called on the reordered system.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "new_order : array_like\n"
             "    New order of blocks, as returned by :meth:`BlockSystem.compute_reordering`.\n"
             "vector : array_like\n"
             "    Vector in the *old* ordering.\n"
             "out : array, optional\n"
             "    Output array to receive the reordered vector contents. Must not be the same\n"
             "    array as ``vector``.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "array\n"
             "    Reordered contents of ``vector``, or a reference to ``out``.\n");

static PyObject *block_system_object_reorder_vector(PyObject *const self, PyTypeObject *const defining_class,
                                                    PyObject *const *const args, const Py_ssize_t nargs,
                                                    const PyObject *kwnames)
{
    const block_system_object *this;
    PyArrayObject *arr_vector, *arr_order, *out_arr;
    if (prepare_for_reordering(self, defining_class, args, nargs, kwnames, &this, &arr_vector, &arr_order, &out_arr) <
        0)
    {
        return NULL;
    }

    hybsol_system_reorder_vector(this->system, (const uint64_t *)PyArray_DATA(arr_order),
                                 (const double *)PyArray_DATA(arr_vector), (double *)PyArray_DATA(out_arr));

    Py_DECREF(arr_vector);
    Py_DECREF(arr_order);
    return (PyObject *)out_arr;
}

PyDoc_STRVAR(block_system_object_unorder_vector_docstring,
             "unorder_vector(new_order: numpy.typing.ArrayLike, vector: numpy.typing.ArrayLike, out: "
             "numpy.typing.NDArray[numpy.double] | None = None) -> numpy.typing.NDArray[numpy.double]\n"
             "Undo reordering of the vector based on new block ordering.\n"
             "\n"
             "The block that sits at slot ``new_order[i]`` is moved back to slot ``i``,\n"
             "restoring the vector of the system as it was before\n"
             ":meth:`reorder_blocks`. The method is meant to be called on the\n"
             "reordered system.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "new_order : array_like\n"
             "    New order of blocks, as returned by :meth:`BlockSystem.compute_reordering`.\n"
             "vector : array_like\n"
             "    Vector in the *new* ordering.\n"
             "out : array, optional\n"
             "    Output array to receive the resulting vector contents. Must not be the same\n"
             "    array as ``vector``.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "array\n"
             "    Contents of ``vector`` in the original order, or a reference to ``out``.\n");

static PyObject *block_system_object_unorder_vector(PyObject *const self, PyTypeObject *const defining_class,
                                                    PyObject *const *const args, const Py_ssize_t nargs,
                                                    const PyObject *kwnames)
{
    const block_system_object *this;
    PyArrayObject *arr_vector, *arr_order, *out_arr;
    if (prepare_for_reordering(self, defining_class, args, nargs, kwnames, &this, &arr_vector, &arr_order, &out_arr) <
        0)
    {
        return NULL;
    }

    hybsol_system_unorder_vector(this->system, (const uint64_t *)PyArray_DATA(arr_order),
                                 (const double *)PyArray_DATA(arr_vector), (double *)PyArray_DATA(out_arr));

    Py_DECREF(arr_vector);
    Py_DECREF(arr_order);
    return (PyObject *)out_arr;
}

/* ------------------------------------------------------------------------- */
/* Bulk constructors                                                          */
/* ------------------------------------------------------------------------- */

PyDoc_STRVAR(block_system_from_blocks_docstring,
             "from_blocks(block_sizes, rows, cols, data, precision=Precision.DOUBLE) -> BlockSystem\n"
             "classmethod\n"
             "Build a whole system from a flat COO description in one pass.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "block_sizes : sequence of int\n"
             "    Size of every block on the diagonal.\n"
             "rows : array_like\n"
             "    Row index of every block.\n"
             "cols : array_like\n"
             "    Column index of every block.\n"
             "data : array_like\n"
             "    Concatenated, row-major block values; block ``k`` occupies\n"
             "    ``block_sizes[rows[k]] * block_sizes[cols[k]]`` entries.\n"
             "\n"
             "    A block and its transpose share a row-major ravel *only* when one of\n"
             "    them has a single column.\n"
             "precision : hybsol.Precision, default: Precision.DOUBLE\n"
             "    Type the blocks are stored in. Values are converted on the way in,\n"
             "    so ``data`` may hold any floating-point type.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "BlockSystem\n"
             "    A new system holding exactly the given blocks.\n");

static PyObject *block_system_from_blocks(PyObject *const cls, PyObject *const *const args, const Py_ssize_t nargs,
                                          const PyObject *kwnames)
{
    PyTypeObject *const type = (PyTypeObject *)cls;
    if (module_state_from_type(type) == NULL)
    {
        return NULL;
    }

    PyObject *py_sizes, *py_rows, *py_cols, *py_data, *py_precision = NULL;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = &py_sizes, .kwname = "block_sizes"},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &py_rows, .kwname = "rows"},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &py_cols, .kwname = "cols"},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &py_data, .kwname = "data"},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &py_precision, .kwname = "precision", .optional = 1},
                {},
            },
            args, nargs, kwnames) < 0)
    {
        return NULL;
    }

    uint64_t *sizes = NULL;
    Py_ssize_t n = 0;
    if (parse_block_sizes(py_sizes, &sizes, &n) < 0)
    {
        return NULL;
    }

    hybsol_precision_t precision = HYBSOL_PRECISION_DOUBLE;
    if (py_precision != NULL && parse_precision(py_precision, &precision) < 0)
    {
        PyMem_Free(sizes);
        return NULL;
    }

    PyObject *const result = block_system_alloc(type, n, sizes, precision);
    PyMem_Free(sizes);
    if (!result)
    {
        return NULL;
    }

    // Delegate to the instance method so both entry points share exactly one
    // implementation of the bulk assembly path.
    PyObject *const ret = block_system_object_add_blocks((PyObject *)result, type,
                                                         (PyObject *const[]){py_rows, py_cols, py_data}, 3, NULL);
    if (!ret)
    {
        Py_DECREF(result);
        return NULL;
    }
    Py_DECREF(ret);
    return result;
}

PyDoc_STRVAR(block_system_from_block_list_docstring,
             "from_block_list(blocks, block_sizes=None, precision=Precision.DOUBLE) -> BlockSystem\n"
             "classmethod\n"
             "Build a system from a list of ``(row, col, array)`` triples.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "blocks : sequence of tuple\n"
             "    Triples ``(row, col, value)``, where ``value`` has shape\n"
             "    ``(block_sizes[row], block_sizes[col])``.\n"
             "block_sizes : sequence of int, optional\n"
             "    Size of every block on the diagonal. When omitted, the sizes are\n"
             "    inferred from the shapes of the given blocks; that is only possible\n"
             "    if every block index appears in at least one triple.\n"
             "precision : hybsol.Precision, default: Precision.DOUBLE\n"
             "    Type the blocks are stored in. Unlike :meth:`add_block`, the values\n"
             "    are *not* converted: each ``value`` must already have this type, and a\n"
             "    ``float64`` array is refused for a single-precision system.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "BlockSystem\n"
             "    A new system holding exactly the given blocks.\n");

static PyObject *block_system_from_block_list(PyObject *const cls, PyObject *const *const args, const Py_ssize_t nargs,
                                              const PyObject *kwnames)
{
    PyTypeObject *const type = (PyTypeObject *)cls;
    if (module_state_from_type(type) == NULL)
    {
        return NULL;
    }

    PyObject *py_blocks, *py_sizes = NULL, *py_precision = NULL;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = &py_blocks, .kwname = "blocks"},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &py_sizes, .kwname = "block_sizes", .optional = 1},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &py_precision, .kwname = "precision", .optional = 1},
                {},
            },
            args, nargs, kwnames) < 0)
    {
        return NULL;
    }

    PyObject *const seq = PySequence_Fast(py_blocks, "blocks must be a sequence of (row, col, array) triples");
    if (!seq)
    {
        return NULL;
    }
    const Py_ssize_t n_entries = PySequence_Fast_GET_SIZE(seq);
    if (n_entries == 0)
    {
        PyErr_SetString(PyExc_ValueError, "blocks must contain at least one block");
        Py_DECREF(seq);
        return NULL;
    }

    hybsol_precision_t precision = HYBSOL_PRECISION_DOUBLE;
    if (py_precision != NULL && parse_precision(py_precision, &precision) < 0)
    {
        Py_DECREF(seq);
        return NULL;
    }
    const int precision_single = precision == HYBSOL_PRECISION_SINGLE;

    // First pass: validate the triples, convert every value and figure out
    // how much room the flat buffer needs.
    PyArrayObject **const values = PyMem_Calloc((size_t)n_entries, sizeof(*values));
    uint64_t *const rows = PyMem_Malloc(sizeof(uint64_t) * (size_t)n_entries);
    uint64_t *const cols = PyMem_Malloc(sizeof(uint64_t) * (size_t)n_entries);
    if (!values || !rows || !cols)
    {
        PyMem_Free(values);
        PyMem_Free(rows);
        PyMem_Free(cols);
        Py_DECREF(seq);
        return PyErr_NoMemory();
    }

    Py_ssize_t n_blocks = 0;
    size_t needed = 0;
    PyObject *result = NULL;

    for (Py_ssize_t k = 0; k < n_entries; ++k)
    {
        PyObject *const item = PySequence_Fast_GET_ITEM(seq, k);
        if (!PyTuple_Check(item) || PyTuple_GET_SIZE(item) != 3)
        {
            PyErr_Format(PyExc_TypeError, "Block %zd is not a (row, col, array) triple.", k);
            goto cleanup;
        }

        const Py_ssize_t row = PyNumber_AsSsize_t(PyTuple_GET_ITEM(item, 0), PyExc_ValueError);
        if (PyErr_Occurred())
        {
            goto cleanup;
        }
        const Py_ssize_t col = PyNumber_AsSsize_t(PyTuple_GET_ITEM(item, 1), PyExc_ValueError);
        if (PyErr_Occurred())
        {
            goto cleanup;
        }
        if (row < 0 || col < 0)
        {
            PyErr_Format(PyExc_ValueError, "Block %zd has a negative index (%zd, %zd).", k, row, col);
            goto cleanup;
        }

        PyArrayObject *const val =
            (PyArrayObject *)PyArray_FROMANY(PyTuple_GET_ITEM(item, 2), precision_single ? NPY_FLOAT : NPY_DOUBLE, 2, 2,
                                             NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
        if (!val)
        {
            goto cleanup;
        }

        rows[k] = (uint64_t)row;
        cols[k] = (uint64_t)col;
        values[k] = val;
        n_blocks = row + 1 > n_blocks ? row + 1 : n_blocks;
        n_blocks = col + 1 > n_blocks ? col + 1 : n_blocks;
        needed += (size_t)PyArray_DIM(val, 0) * (size_t)PyArray_DIM(val, 1);
    }

    // Second pass: work out (or check) the block sizes.
    uint64_t *sizes = NULL;
    if (py_sizes != NULL && py_sizes != Py_None)
    {
        if (parse_block_sizes(py_sizes, &sizes, &n_blocks) < 0)
        {
            goto cleanup;
        }
    }
    else
    {
        sizes = PyMem_Calloc((size_t)n_blocks, sizeof(uint64_t));
        if (!sizes)
        {
            PyErr_NoMemory();
            goto cleanup;
        }
        for (Py_ssize_t k = 0; k < n_entries; ++k)
        {
            const Py_ssize_t shape[2] = {PyArray_DIM(values[k], 0), PyArray_DIM(values[k], 1)};
            const Py_ssize_t idx[2] = {(Py_ssize_t)rows[k], (Py_ssize_t)cols[k]};
            for (int j = 0; j < 2; ++j)
            {
                if (sizes[idx[j]] == 0)
                {
                    sizes[idx[j]] = (uint64_t)shape[j];
                }
                else if (sizes[idx[j]] != (uint64_t)shape[j])
                {
                    PyErr_Format(PyExc_ValueError, "Inconsistent size for block %zd: seen both %llu and %lld.", idx[j],
                                 (unsigned long long)sizes[idx[j]], (long long)shape[j]);
                    PyMem_Free(sizes);
                    goto cleanup;
                }
            }
        }
        for (Py_ssize_t i = 0; i < n_blocks; ++i)
        {
            if (sizes[i] == 0)
            {
                PyErr_Format(PyExc_ValueError, "Cannot infer the size of block %zd; pass block_sizes explicitly.", i);
                PyMem_Free(sizes);
                goto cleanup;
            }
        }
    }

    const int single = precision_single;
    const size_t elem_size = single ? sizeof(float) : sizeof(double);

    // Third pass: flatten into a single buffer the bulk path understands.
    void *const data = PyMem_Malloc(elem_size * needed);
    if (!data)
    {
        PyMem_Free(sizes);
        PyErr_NoMemory();
        goto cleanup;
    }
    size_t offset = 0;
    for (Py_ssize_t k = 0; k < n_entries; ++k)
    {
        const size_t len = (size_t)PyArray_DIM(values[k], 0) * (size_t)PyArray_DIM(values[k], 1);
        memcpy((char *)data + offset, PyArray_DATA(values[k]), elem_size * len);
        offset += elem_size * len;
    }

    result = block_system_alloc(type, n_blocks, sizes, precision);
    PyMem_Free(sizes);
    if (!result)
    {
        PyMem_Free(data);
        goto cleanup;
    }

    {
        const hybsol_result_t res =
            single ? hybsol_system_add_blocks_f32(((block_system_object *)result)->system, (uint64_t)n_entries, rows,
                                                  cols, (const float *)data)
                   : hybsol_system_add_blocks(((block_system_object *)result)->system, (uint64_t)n_entries, rows, cols,
                                              (const double *)data);
        PyMem_Free(data);
        if (res != HYBSOL_SUCCESS)
        {
            hybsol_raise(type, "from_block_list", res);
            Py_DECREF(result);
            result = NULL;
            goto cleanup;
        }
    }

cleanup:
    for (Py_ssize_t k = 0; k < n_entries; ++k)
    {
        Py_XDECREF(values[k]);
    }
    PyMem_Free(values);
    PyMem_Free(rows);
    PyMem_Free(cols);
    Py_DECREF(seq);
    return result;
}

/* ------------------------------------------------------------------------- */
/* Getters                                                                    */
/* ------------------------------------------------------------------------- */

/**
 * Build the :class:`hybsol.Precision` member a system's precision spells as.
 *
 * The enum lives in the Python half of the package, so it is looked up on
 * demand rather than cached at import: by the time anyone reads
 * ``BlockSystem.precision`` the package has finished importing, which keeps
 * the extension free of an import-time dependency on it.
 */
#define precision_member(precision) hybsol_precision_member(precision)

static PyObject *block_system_object_get_precision(PyObject *const self, void *const Py_UNUSED(closure))
{
    const block_system_object *const this = (const block_system_object *)self;
    return precision_member(hybsol_system_precision(this->system));
}

static PyObject *block_system_object_get_n_blocks(PyObject *const self, void *const Py_UNUSED(closure))
{
    const block_system_object *const this = (const block_system_object *)self;
    return PyLong_FromUnsignedLongLong(hybsol_system_n_blocks(this->system));
}

static PyObject *block_system_object_get_block_sizes(PyObject *const self, void *const Py_UNUSED(closure))
{
    const block_system_object *const this = (const block_system_object *)self;
    const uint64_t n = hybsol_system_n_blocks(this->system);
    const npy_intp dim = (npy_intp)n;
    PyArrayObject *const arr = (PyArrayObject *)PyArray_SimpleNew(1, &dim, NPY_UINT64);
    if (!arr)
    {
        return NULL;
    }
    npy_uint64 *const out = (npy_uint64 *)PyArray_DATA(arr);
    for (uint64_t i = 0; i < n; ++i)
    {
        out[i] = hybsol_system_block_size(this->system, i);
    }
    return (PyObject *)arr;
}

/* ------------------------------------------------------------------------- */
/* Applying the system                                                        */
/* ------------------------------------------------------------------------- */

PyDoc_STRVAR(block_system_object_matvec_docstring,
             "matvec(x: numpy.typing.ArrayLike, out: numpy.typing.NDArray[numpy.double] | None = None, "
             "n_threads: int = 0) -> numpy.typing.NDArray[numpy.double]\n"
             "Apply the system to a vector, ``y = A x``.\n"
             "\n"
             "This is the operator the elimination solves against, and the one worth\n"
             "having fast: it walks the stored blocks rather than a dense form, so a\n"
             "sparse system costs the sum of its blocks rather than the square of its\n"
             "dimension.\n"
             "\n"
             "``out`` is written, not accumulated into. The work is split across\n"
             "block rows, each of which writes only its own slice, so the system\n"
             "is only read and stays usable throughout.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "x : array_like\n"
             "    Right-hand side of length ``sum(block_sizes)``. Narrower\n"
             "    floating point is widened as it is read.\n"
             "out : array, optional\n"
             "    Array to write the result to. If not given (or ``None``), a new\n"
             "    array is created.\n"
             "n_threads : int, default: 0\n"
             "    Number of OpenMP threads; ``0`` selects the OpenMP default\n"
             "    and ``1`` runs it serially.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "array\n"
             "    The result, always in double precision.\n");

static PyObject *block_system_object_matvec(PyObject *const self, PyTypeObject *const defining_class,
                                            PyObject *const *args, const Py_ssize_t nargs, PyObject *kwnames)
{
    block_system_object *this;
    const module_state_t *state;
    if (ensure_block_system_and_state(self, defining_class, &this, &state) < 0)
    {
        return NULL;
    }

    PyObject *py_x, *py_out = NULL;
    Py_ssize_t n_threads = 0;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = &py_x, .kwname = "x"},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &py_out, .kwname = "out", .optional = 1},
                {.type = CPYARG_TYPE_SSIZE, .p_val = &n_threads, .kwname = "n_threads", .optional = 1},
                {},
            },
            args, nargs, kwnames) < 0)
    {
        return NULL;
    }
    if (hybsol_check_n_threads(n_threads) < 0)
    {
        return NULL;
    }

    const npy_intp dim = (npy_intp)hybsol_system_total_size(this->system);
    PyArrayObject *arr = NULL;
    if (hybsol_double_array(py_x, 1, &dim, "x", &arr) < 0)
    {
        return NULL;
    }

    PyArrayObject *user_out = NULL;
    if (hybsol_optional_array(py_out, &user_out) < 0)
    {
        Py_DECREF(arr);
        return NULL;
    }
    PyArrayObject *out = NULL;
    if (hybsol_prepare_output(user_out, 1, &dim, NPY_DOUBLE, 0, "out", &out) < 0)
    {
        Py_DECREF(arr);
        return NULL;
    }

    const hybsol_matrix_t x = hybsol_matrix_from_array(arr);
    hybsol_matrix_t y = hybsol_matrix_from_array(out);

    hybsol_result_t res;
    Py_BEGIN_ALLOW_THREADS;
    res = hybsol_system_matvec(this->system, &x, &y, (uint64_t)n_threads);
    Py_END_ALLOW_THREADS;

    Py_DECREF(arr);
    if (res != HYBSOL_SUCCESS)
    {
        Py_DECREF(out);
        return hybsol_raise(Py_TYPE(self), "matvec", res);
    }
    return (PyObject *)out;
}

PyDoc_STRVAR(block_system_object_matmat_docstring,
             "matmat(x: numpy.typing.ArrayLike, out: numpy.typing.NDArray[numpy.double] | None = None, "
             "n_threads: int = 0) -> numpy.typing.NDArray[numpy.double]\n"
             "Apply the system to several right-hand sides, ``y = A x``.\n"
             "\n"
             "The same walk as :meth:`matvec`, done for every column of ``x``\n"
             "at once, so the pattern is traversed once rather than once per\n"
             "column. With one column the two are identical.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "x : array_like\n"
             "    Two-dimensional input of ``sum(block_sizes)`` rows. Narrower\n"
             "    floating point is widened as it is read.\n"
             "out : array, optional\n"
             "    Array to write the result to, with the same shape as ``x``. If\n"
             "    not given (or ``None``), a new array is created.\n"
             "n_threads : int, default: 0\n"
             "    Number of OpenMP threads; ``0`` selects the OpenMP default\n"
             "    and ``1`` runs it serially.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "array\n"
             "    The result, always in double precision.\n");

static PyObject *block_system_object_matmat(PyObject *const self, PyTypeObject *const defining_class,
                                            PyObject *const *args, const Py_ssize_t nargs, PyObject *kwnames)
{
    block_system_object *this;
    const module_state_t *state;
    if (ensure_block_system_and_state(self, defining_class, &this, &state) < 0)
    {
        return NULL;
    }

    PyObject *py_x, *py_out = NULL;
    Py_ssize_t n_threads = 0;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = &py_x, .kwname = "x"},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &py_out, .kwname = "out", .optional = 1},
                {.type = CPYARG_TYPE_SSIZE, .p_val = &n_threads, .kwname = "n_threads", .optional = 1},
                {},
            },
            args, nargs, kwnames) < 0)
    {
        return NULL;
    }
    if (hybsol_check_n_threads(n_threads) < 0)
    {
        return NULL;
    }

    const npy_intp dim = (npy_intp)hybsol_system_total_size(this->system);
    const npy_intp x_dims[2] = {dim, 0};
    PyArrayObject *arr = NULL;
    if (hybsol_double_array(py_x, 2, x_dims, "x", &arr) < 0)
    {
        return NULL;
    }

    const npy_intp out_dims[2] = {dim, PyArray_DIM(arr, 1)};
    PyArrayObject *user_out = NULL;
    if (hybsol_optional_array(py_out, &user_out) < 0)
    {
        Py_DECREF(arr);
        return NULL;
    }
    PyArrayObject *out = NULL;
    if (hybsol_prepare_output(user_out, 2, out_dims, NPY_DOUBLE, 0, "out", &out) < 0)
    {
        Py_DECREF(arr);
        return NULL;
    }

    const hybsol_matrix_t x = hybsol_matrix_from_array(arr);
    hybsol_matrix_t y = hybsol_matrix_from_array(out);

    hybsol_result_t res;
    Py_BEGIN_ALLOW_THREADS;
    res = hybsol_system_matmat(this->system, &x, &y, (uint64_t)n_threads);
    Py_END_ALLOW_THREADS;

    Py_DECREF(arr);
    if (res != HYBSOL_SUCCESS)
    {
        Py_DECREF(out);
        return hybsol_raise(Py_TYPE(self), "matmat", res);
    }
    return (PyObject *)out;
}

/* ------------------------------------------------------------------------- */
/* Type definition                                                            */
/* ------------------------------------------------------------------------- */

#define BS_METHOD(name, fn, doc)                                                                                       \
    {                                                                                                                  \
        .ml_name = name,                                                                                               \
        .ml_meth = (void *)(fn),                                                                                       \
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,                                                       \
        .ml_doc = (doc),                                                                                               \
    }

#define BS_CLASS_METHOD(name, fn, doc)                                                                                 \
    {                                                                                                                  \
        .ml_name = name,                                                                                               \
        .ml_meth = (void *)(fn),                                                                                       \
        .ml_flags = METH_CLASS | METH_FASTCALL | METH_KEYWORDS,                                                        \
        .ml_doc = (doc),                                                                                               \
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
            {Py_tp_repr, block_system_repr},
            {Py_tp_doc, (void *)block_system_docstring},
            {Py_tp_methods,
             (PyMethodDef[]){
                 BS_METHOD("add_block", block_system_object_add_block, block_system_add_block_docstring),
                 BS_METHOD("add_blocks", block_system_object_add_blocks, block_system_object_add_blocks_docstring),
                 BS_METHOD("reserve", block_system_object_reserve, block_system_object_reserve_docstring),
                 BS_METHOD("is_valid", block_system_object_is_valid, block_system_object_is_valid_docstring),
                 BS_METHOD("as_array", block_system_object_as_array, block_system_object_as_array_docstring),
                 BS_METHOD("get_row_block_indices", block_system_object_get_row_block_indices,
                           block_system_object_get_row_block_indices_docstring),
                 BS_METHOD("get_block", block_system_object_get_block, block_system_object_get_block_docstring),
                 BS_METHOD("block_storage", block_system_object_block_storage,
                           block_system_object_block_storage_docstring),
                 BS_METHOD("get_block_size", block_system_object_get_block_size,
                           block_system_object_get_block_size_docstring),
                 BS_METHOD("has_block", block_system_object_has_block, block_system_object_has_block_docstring),
                 BS_METHOD("no_lower_connections", block_system_object_no_lower_connections,
                           block_system_object_no_lower_connections_docstring),
                 BS_METHOD("first_column", block_system_object_first_column,
                           block_system_object_first_column_docstring),
                 BS_METHOD("get_next_column_index", block_system_object_get_next_column_index,
                           block_system_object_get_next_column_index_docstring),
                 BS_METHOD("matvec", block_system_object_matvec, block_system_object_matvec_docstring),
                 BS_METHOD("matmat", block_system_object_matmat, block_system_object_matmat_docstring),
                 BS_METHOD("multiply_row", block_system_object_multiply_row,
                           block_system_object_multiply_row_docstring),
                 BS_METHOD("eliminate_row", block_system_object_eliminate_row,
                           block_system_object_eliminate_row_docstring),
                 BS_METHOD("decompose", block_system_object_decompose, block_system_object_decompose_docstring),
                 BS_METHOD("workspace_bytes", block_system_object_workspace_bytes,
                           block_system_object_workspace_bytes_docstring),
                 BS_METHOD("elimination", block_system_object_elimination, block_system_object_elimination_docstring),
                 BS_METHOD("copy", block_system_object_copy, block_system_object_copy_docstring),
                 BS_METHOD("reorder_blocks", block_system_object_reorder_blocks,
                           block_system_object_reorder_blocks_docstring),
                 BS_METHOD("compute_reordering", block_system_object_compute_reordering,
                           block_system_object_compute_reordering_docstring),
                 BS_METHOD("reorder_vector", block_system_object_reorder_vector,
                           block_system_object_reorder_vector_docstring),
                 BS_METHOD("unorder_vector", block_system_object_unorder_vector,
                           block_system_object_unorder_vector_docstring),
                 BS_CLASS_METHOD("from_blocks", block_system_from_blocks, block_system_from_blocks_docstring),
                 BS_CLASS_METHOD("from_block_list", block_system_from_block_list,
                                 block_system_from_block_list_docstring),
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
                     .name = "precision",
                     .get = block_system_object_get_precision,
                     .doc = "hybsol.Precision : The type this system stores its blocks in.",
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
