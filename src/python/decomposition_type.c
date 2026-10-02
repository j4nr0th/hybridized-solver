/**
 * @file src/python/decomposition_type.c
 * Implementation of the :class:`hybsol.Decomposition` extension type.
 *
 * Every method is a thin argument-validation shell around one core call.
 */

#include "decomposition_type.h"

#include "numpy_convert.h"

/* ------------------------------------------------------------------------- */
/* Small helpers                                                              */
/* ------------------------------------------------------------------------- */

static int ensure_decomposition(PyObject *const self, PyTypeObject *const defining_class,
                                decomposition_object **const p_this)
{
    const module_state_t *const state =
        defining_class ? module_state_from_type(defining_class) : module_state_from_type(Py_TYPE(self));
    if (!state)
    {
        return -1;
    }

    if (!PyObject_TypeCheck(self, state->type_decomposition))
    {
        PyErr_Format(PyExc_TypeError, "Expected a %s, got a %s.", MODULE_TYPE_NAME(Decomposition),
                     Py_TYPE(self)->tp_name);
        return -1;
    }

    *p_this = (decomposition_object *)self;
    return 0;
}

/* ------------------------------------------------------------------------- */
/* Construction and destruction                                               */
/* ------------------------------------------------------------------------- */

PyObject *decomposition_alloc(PyTypeObject *const type, hybsol_decomposition_t *const decomposition)
{
    decomposition_object *const self = (decomposition_object *)type->tp_alloc(type, 0);
    if (!self)
    {
        return NULL;
    }
    self->decomposition = decomposition;
    return (PyObject *)self;
}

static void decomposition_dealloc(decomposition_object *const self)
{
    PyObject_GC_UnTrack(self);
    PyTypeObject *const type = Py_TYPE(self);
    hybsol_decomposition_destroy(self->decomposition);
    self->decomposition = NULL;
    type->tp_free((PyObject *)self);
    Py_DECREF(type);
}

static PyObject *decomposition_repr(decomposition_object *const self)
{
    return PyUnicode_FromFormat("<hybsol.Decomposition n_blocks=%llu, size=%llu>",
                                (unsigned long long)hybsol_decomposition_n_blocks(self->decomposition),
                                (unsigned long long)hybsol_decomposition_total_size(self->decomposition));
}

PyDoc_STRVAR(decomposition_docstring, "Decomposition(n_blocks, size)\n"
                                      "A factorized block system.\n"
                                      "\n"
                                      "Produced by :meth:`BlockSystem.decompose`. It owns a\n"
                                      "copy of every block the elimination needs, so the\n"
                                      "system it came from is unchanged and can be\n"
                                      "decomposed again under a different block order.\n");

/* ------------------------------------------------------------------------- */
/* Solving                                                                    */
/* ------------------------------------------------------------------------- */

PyDoc_STRVAR(decomposition_object_solve_docstring,
             "solve(val: numpy.typing.ArrayLike, out: numpy.typing.NDArray[numpy.double] | None = None, "
             "n_threads: int = 0) -> numpy.typing.NDArray[numpy.double]\n"
             "Solve the system for the given right side.\n"
             "\n"
             "The forward substitution is parallel over the block rows of each\n"
             "elimination pass; the back substitution that follows is serial.\n"
             "The answer does not depend on the thread count, and a\n"
             "decomposition may be solved any number of times.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "val : array_like\n"
             "    Right-hand side of length ``sum(block_sizes)``.\n"
             "out : array, optional\n"
             "    Array to write the solution to. If not given (or ``None``), a new array\n"
             "    is created. It may be the same array as ``val``, in which case the\n"
             "    solution overwrites the right-hand side in place.\n"
             "n_threads : int, default: 0\n"
             "    Number of OpenMP threads for the forward substitution; ``0``\n"
             "    selects the OpenMP default and ``1`` runs it serially.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "array\n"
             "    Solution of the linear system.\n"
             "\n"
             "Raises\n"
             "------\n"
             "ValueError\n"
             "    ``val`` does not hold exactly ``total_size`` values;\n"
             "    ``n_threads`` is negative; or ``out`` is not a writable\n"
             "    ``float64`` array of that length.\n");

static PyObject *decomposition_object_solve(PyObject *const self, PyTypeObject *const defining_class,
                                            PyObject *const *args, const Py_ssize_t nargs, PyObject *kwnames)
{
    decomposition_object *this;
    if (ensure_decomposition(self, defining_class, &this) < 0)
    {
        return NULL;
    }

    Py_ssize_t n_threads = 0;
    PyObject *py_val, *py_out = NULL;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = &py_val, .kwname = "val"},
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

    const npy_intp dim = (npy_intp)hybsol_decomposition_total_size(this->decomposition);
    PyArrayObject *user_out = NULL;
    if (hybsol_optional_array(py_out, &user_out) < 0)
    {
        return NULL;
    }

    PyArrayObject *arr = NULL;
    if (hybsol_double_array(py_val, 1, &dim, "val", &arr) < 0)
    {
        return NULL;
    }

    PyArrayObject *out = NULL;
    if (hybsol_prepare_output(user_out, 1, &dim, NPY_DOUBLE, 0, "out", &out) < 0)
    {
        Py_DECREF(arr);
        return NULL;
    }

    double *const ptr = (double *)PyArray_DATA(out);
    if (PyArray_DATA(arr) != ptr)
    {
        memcpy(ptr, PyArray_DATA(arr), sizeof(double) * (size_t)dim);
    }

    hybsol_result_t res;
    Py_BEGIN_ALLOW_THREADS;
    res = hybsol_decomposition_solve(this->decomposition, ptr, (uint64_t)n_threads);
    Py_END_ALLOW_THREADS;

    Py_DECREF(arr);
    if (res != HYBSOL_SUCCESS)
    {
        Py_DECREF(out);
        return hybsol_raise_block(Py_TYPE(self), "solve", res, hybsol_decomposition_failing_block(this->decomposition));
    }

    return (PyObject *)out;
}

/* ------------------------------------------------------------------------- */
/* The recorded operations                                                    */
/* ------------------------------------------------------------------------- */

PyDoc_STRVAR(decomposition_object_operations_docstring,
             "operations() -> tuple[tuple[int, ...], ...]\n"
             "Get the recorded operations as tuples of one or two ints.\n"
             "\n"
             "A one-element tuple ``(idx_row,)`` solves with the LU factors of the\n"
             "diagonal block; a two-element tuple ``(idx_row, idx_col)`` eliminates\n"
             "block ``(idx_row, idx_col)`` using block row ``idx_col``.\n"
             "\n"
             "The list is rebuilt from the decomposition's own copy of the schedule\n"
             "rather than stored, so it costs nothing until it is asked for.\n");

static PyObject *decomposition_object_operations(PyObject *const self, PyTypeObject *const defining_class,
                                                 PyObject *const *const Py_UNUSED(args), const Py_ssize_t nargs,
                                                 PyObject *kwnames)
{
    decomposition_object *this;
    if (ensure_decomposition(self, defining_class, &this) < 0)
    {
        return NULL;
    }
    if (nargs != 0 || kwnames != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "operations() takes no arguments.");
        return NULL;
    }

    const uint64_t n_ops = hybsol_decomposition_n_operations(this->decomposition);
    PyObject *const out = PyTuple_New((Py_ssize_t)n_ops);
    if (!out)
    {
        return NULL;
    }

    hybsol_operation_t *const ops = PyMem_Malloc(sizeof(*ops) * (size_t)n_ops);
    if (!ops)
    {
        Py_DECREF(out);
        return PyErr_NoMemory();
    }

    // The core asserts the capacity and can only report success, so this is
    // unreachable; raise rather than abort if it ever is not.
    uint64_t written = 0;
    if (hybsol_decomposition_operations(this->decomposition, ops, n_ops, &written) != HYBSOL_SUCCESS)
    {
        PyMem_Free(ops);
        Py_DECREF(out);
        PyErr_SetString(PyExc_RuntimeError, "operations: the decomposition did not report its own operation count.");
        return NULL;
    }

    for (uint64_t i = 0; i < written; ++i)
    {
        PyObject *val;
        switch (ops[i].type)
        {
        case HYBSOL_OPERATION_INVERT_DIAGONAL:
            val = cpyutl_output_create_check(CPYOUT_TYPE_TUPLE,
                                             (const cpyutl_output_t[]){
                                                 {.type = CPYOUT_TYPE_PYINT, .value_int = (Py_ssize_t)ops[i].idx_row},
                                                 {},
                                             });
            break;

        case HYBSOL_OPERATION_ELIMINATE:
            val = cpyutl_output_create_check(CPYOUT_TYPE_TUPLE,
                                             (const cpyutl_output_t[]){
                                                 {.type = CPYOUT_TYPE_PYINT, .value_int = (Py_ssize_t)ops[i].idx_row},
                                                 {.type = CPYOUT_TYPE_PYINT, .value_int = (Py_ssize_t)ops[i].idx_col},
                                                 {},
                                             });
            break;

        default:
            PyMem_Free(ops);
            Py_DECREF(out);
            PyErr_Format(PyExc_RuntimeError, "Unknown operation type %d.", (int)ops[i].type);
            return NULL;
        }

        if (!val)
        {
            PyMem_Free(ops);
            Py_DECREF(out);
            return NULL;
        }
        PyTuple_SET_ITEM(out, (Py_ssize_t)i, val);
    }

    PyMem_Free(ops);
    return out;
}

/* ------------------------------------------------------------------------- */
/* Queries                                                                    */
/* ------------------------------------------------------------------------- */

static PyObject *decomposition_object_get_n_blocks(PyObject *const self, void *const Py_UNUSED(closure))
{
    decomposition_object *this;
    if (ensure_decomposition(self, NULL, &this) < 0)
    {
        return NULL;
    }
    return PyLong_FromUnsignedLongLong(hybsol_decomposition_n_blocks(this->decomposition));
}

static PyObject *decomposition_object_get_total_size(PyObject *const self, void *const Py_UNUSED(closure))
{
    decomposition_object *this;
    if (ensure_decomposition(self, NULL, &this) < 0)
    {
        return NULL;
    }
    return PyLong_FromUnsignedLongLong(hybsol_decomposition_total_size(this->decomposition));
}

static PyObject *decomposition_object_get_n_operations(PyObject *const self, void *const Py_UNUSED(closure))
{
    decomposition_object *this;
    if (ensure_decomposition(self, NULL, &this) < 0)
    {
        return NULL;
    }
    return PyLong_FromUnsignedLongLong(hybsol_decomposition_n_operations(this->decomposition));
}

static PyObject *decomposition_object_get_device(PyObject *const self, void *const Py_UNUSED(closure))
{
    decomposition_object *this;
    if (ensure_decomposition(self, NULL, &this) < 0)
    {
        return NULL;
    }

    const uint64_t index = hybsol_decomposition_device(this->decomposition);
    if (index == UINT64_MAX)
    {
        Py_RETURN_NONE;
    }
    return PyLong_FromUnsignedLongLong(index);
}

/* ------------------------------------------------------------------------- */
/* Type definition                                                            */
/* ------------------------------------------------------------------------- */

#define DECOMPOSITION_METHOD(name, fn, doc)                                                                            \
    {                                                                                                                  \
        .ml_name = name,                                                                                               \
        .ml_meth = (void *)(fn),                                                                                       \
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,                                                       \
        .ml_doc = (doc),                                                                                               \
    }

PyType_Spec decomposition_type_spec = {
    .name = MODULE_TYPE_NAME(Decomposition),
    .basicsize = sizeof(decomposition_object),
    .itemsize = 0,
    .flags = Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_DEFAULT,
    .slots =
        (PyType_Slot[]){
            {Py_tp_traverse, heap_type_traverse_type},
            {Py_tp_dealloc, decomposition_dealloc},
            {Py_tp_repr, decomposition_repr},
            {Py_tp_doc, (void *)decomposition_docstring},
            {Py_tp_methods,
             (PyMethodDef[]){
                 DECOMPOSITION_METHOD("solve", decomposition_object_solve, decomposition_object_solve_docstring),
                 DECOMPOSITION_METHOD("operations", decomposition_object_operations,
                                      decomposition_object_operations_docstring),
                 {},
             }},
            {Py_tp_getset,
             (PyGetSetDef[]){
                 {
                     .name = "n_blocks",
                     .get = decomposition_object_get_n_blocks,
                     .doc = "int : Number of blocks.",
                 },
                 {
                     .name = "total_size",
                     .get = decomposition_object_get_total_size,
                     .doc = "int : Rows of the matrix the decomposition solves.",
                 },
                 {
                     .name = "n_operations",
                     .get = decomposition_object_get_n_operations,
                     .doc = "int : Number of operations the factorization records.",
                 },
                 {
                     .name = "device",
                     .get = decomposition_object_get_device,
                     .doc = "int | None : Index of the backend device the factors live on,"
                            " or None when they are in host memory.",
                 },
                 {},
             }},
            {},
        },
};
