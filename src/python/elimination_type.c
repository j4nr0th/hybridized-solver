/**
 * @file src/python/elimination_type.c
 * Implementation of the :class:`hybsol.Elimination` extension type.
 *
 * Every method is a thin argument-validation shell around one call into the
 * Python-free core.
 */

#include "elimination_type.h"

/* ------------------------------------------------------------------------- */
/* Small helpers                                                              */
/* ------------------------------------------------------------------------- */

static int ensure_elimination(PyObject *const self, PyTypeObject *const defining_class,
                              elimination_object **const p_this)
{
    const module_state_t *const state =
        defining_class ? module_state_from_type(defining_class) : module_state_from_type(Py_TYPE(self));
    if (!state)
    {
        return -1;
    }

    if (!PyObject_TypeCheck(self, state->type_elimination))
    {
        PyErr_Format(PyExc_TypeError, "Expected a %s, got a %s.", MODULE_TYPE_NAME(Elimination),
                     Py_TYPE(self)->tp_name);
        return -1;
    }

    *p_this = (elimination_object *)self;
    return 0;
}

/**
 * Reject an index the core would only accept under an assertion.
 *
 * @param what Name of the argument, used in the error message.
 * @param n Number of valid indices.
 * @param idx The index the caller gave.
 * @return 0 on success, -1 with a ValueError set.
 */
static int check_index(const char *const what, const uint64_t n, const Py_ssize_t idx)
{
    if (idx >= 0 && (uint64_t)idx < n)
    {
        return 0;
    }
    PyErr_Format(PyExc_ValueError, "%s must be in the range [0, %llu), but it was %zd.", what, (unsigned long long)n,
                 idx);
    return -1;
}

/** Copy ``length`` indices into a tuple. */
static PyObject *indices_to_tuple(const uint64_t *const values, const uint64_t length)
{
    PyObject *const out = PyTuple_New((Py_ssize_t)length);
    if (!out)
    {
        return NULL;
    }
    for (uint64_t i = 0; i < length; ++i)
    {
        PyObject *const value = PyLong_FromUnsignedLongLong(values[i]);
        if (!value)
        {
            Py_DECREF(out);
            return NULL;
        }
        PyTuple_SET_ITEM(out, (Py_ssize_t)i, value);
    }
    return out;
}

/* ------------------------------------------------------------------------- */
/* Construction and destruction                                               */
/* ------------------------------------------------------------------------- */

PyObject *elimination_alloc(PyTypeObject *const type, hybsol_elimination_t *const graph)
{
    elimination_object *const self = (elimination_object *)type->tp_alloc(type, 0);
    if (!self)
    {
        return NULL;
    }
    self->graph = graph;
    return (PyObject *)self;
}

static void elimination_dealloc(elimination_object *const self)
{
    PyObject_GC_UnTrack(self);
    PyTypeObject *const type = Py_TYPE(self);
    hybsol_elimination_destroy(self->graph);
    self->graph = NULL;
    type->tp_free((PyObject *)self);
    Py_DECREF(type);
}

static PyObject *elimination_repr(elimination_object *const self)
{
    return PyUnicode_FromFormat("<hybsol.Elimination n_blocks=%llu, n_levels=%llu, n_operations=%llu>",
                                (unsigned long long)hybsol_elimination_n_blocks(self->graph),
                                (unsigned long long)hybsol_elimination_n_levels(self->graph),
                                (unsigned long long)hybsol_elimination_n_operations(self->graph));
}

PyDoc_STRVAR(elimination_docstring, "Elimination(n_blocks, n_levels, n_operations)\n"
                                    "The symbolic result of eliminating a system.\n"
                                    "\n"
                                    "Produced by :meth:`BlockSystem.elimination`. It\n"
                                    "computes no values, so it can be asked for\n"
                                    "before a factorization is committed: it reports\n"
                                    "the pattern the fill-in will produce, the\n"
                                    "passes the factorization runs in, what each\n"
                                    "will cost, and whether the block order admits\n"
                                    "a factorization at all.\n");

/* ------------------------------------------------------------------------- */
/* Per-row and per-pass queries                                               */
/* ------------------------------------------------------------------------- */

PyDoc_STRVAR(elimination_object_row_columns_docstring,
             "row_columns(row: int) -> tuple[int, ...]\n"
             "Column indices ``row`` holds once the fill-in is complete.\n"
             "\n"
             "Includes every block the system already stores and every block\n"
             "the elimination adds, in increasing order.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "row : int\n"
             "    Block row index.\n");

static PyObject *elimination_object_row_columns(PyObject *const self, PyTypeObject *const defining_class,
                                                PyObject *const *args, const Py_ssize_t nargs, PyObject *kwnames)
{
    elimination_object *this;
    if (ensure_elimination(self, defining_class, &this) < 0)
    {
        return NULL;
    }

    Py_ssize_t row = 0;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &row, .kwname = "row"},
                {},
            },
            args, nargs, kwnames) < 0)
    {
        return NULL;
    }

    const hybsol_elimination_t *const graph = this->graph;
    if (check_index("row", hybsol_elimination_n_blocks(graph), row) < 0)
    {
        return NULL;
    }

    const uint64_t length = hybsol_elimination_row_length(graph, (uint64_t)row);
    uint64_t *const columns = PyMem_Malloc(sizeof(*columns) * (size_t)length);
    if (!columns)
    {
        return PyErr_NoMemory();
    }

    uint64_t written = 0;
    const hybsol_result_t res = hybsol_elimination_row_columns(graph, (uint64_t)row, columns, length, &written);
    if (res != HYBSOL_SUCCESS)
    {
        PyMem_Free(columns);
        return hybsol_raise("row_columns", res);
    }

    PyObject *const out = indices_to_tuple(columns, written);
    PyMem_Free(columns);
    return out;
}

PyDoc_STRVAR(elimination_object_level_rows_docstring, "level_rows(level: int) -> tuple[int, ...]\n"
                                                      "Block rows the given elimination pass processes.\n"
                                                      "\n"
                                                      "A row appears in every pass from its first to its last, so a\n"
                                                      "row whose source is not ready yet simply sits a pass out.\n"
                                                      "Within a pass the rows are in ascending index order, which is\n"
                                                      "the order the recorded operations come out in.\n"
                                                      "\n"
                                                      "Parameters\n"
                                                      "----------\n"
                                                      "level : int\n"
                                                      "    Pass index, in ``[0, n_levels)``.\n");

static PyObject *elimination_object_level_rows(PyObject *const self, PyTypeObject *const defining_class,
                                               PyObject *const *args, const Py_ssize_t nargs, PyObject *kwnames)
{
    elimination_object *this;
    if (ensure_elimination(self, defining_class, &this) < 0)
    {
        return NULL;
    }

    Py_ssize_t level = 0;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &level, .kwname = "level"},
                {},
            },
            args, nargs, kwnames) < 0)
    {
        return NULL;
    }

    const hybsol_elimination_t *const graph = this->graph;
    if (check_index("level", hybsol_elimination_n_levels(graph), level) < 0)
    {
        return NULL;
    }

    const uint64_t length = hybsol_elimination_level_size(graph, (uint64_t)level);
    uint64_t *const rows = PyMem_Malloc(sizeof(*rows) * (size_t)length);
    if (!rows)
    {
        return PyErr_NoMemory();
    }

    uint64_t written = 0;
    const hybsol_result_t res = hybsol_elimination_level_rows(graph, (uint64_t)level, rows, length, &written);
    if (res != HYBSOL_SUCCESS)
    {
        PyMem_Free(rows);
        return hybsol_raise("level_rows", res);
    }

    PyObject *const out = indices_to_tuple(rows, written);
    PyMem_Free(rows);
    return out;
}

/** A one-argument getter over a single index. */
#define ELIMINATION_INDEX_METHOD(pyname, corename)                                                                     \
    static PyObject *elimination_object_##pyname(PyObject *const self, PyTypeObject *const defining_class,             \
                                                 PyObject *const *args, const Py_ssize_t nargs, PyObject *kwnames)     \
    {                                                                                                                  \
        elimination_object *this;                                                                                      \
        if (ensure_elimination(self, defining_class, &this) < 0)                                                       \
        {                                                                                                              \
            return NULL;                                                                                               \
        }                                                                                                              \
        Py_ssize_t row = 0;                                                                                            \
        if (parse_arguments_check(                                                                                     \
                (cpyutl_argument_t[]){                                                                                 \
                    {.type = CPYARG_TYPE_SSIZE, .p_val = &row, .kwname = "row"},                                       \
                    {},                                                                                                \
                },                                                                                                     \
                args, nargs, kwnames) < 0)                                                                             \
        {                                                                                                              \
            return NULL;                                                                                               \
        }                                                                                                              \
        if (check_index("row", hybsol_elimination_n_blocks(this->graph), row) < 0)                                     \
        {                                                                                                              \
            return NULL;                                                                                               \
        }                                                                                                              \
        return PyLong_FromUnsignedLongLong(corename(this->graph, (uint64_t)row));                                      \
    }

ELIMINATION_INDEX_METHOD(row_length, hybsol_elimination_row_length)
ELIMINATION_INDEX_METHOD(row_n_eliminations, hybsol_elimination_row_n_eliminations)
ELIMINATION_INDEX_METHOD(row_first_level, hybsol_elimination_row_first_level)
ELIMINATION_INDEX_METHOD(row_level, hybsol_elimination_row_level)

/* ------------------------------------------------------------------------- */
/* Queries                                                                    */
/* ------------------------------------------------------------------------- */

#define ELIMINATION_GETTER(pyname, expr)                                                                               \
    static PyObject *elimination_object_get_##pyname(PyObject *const self, void *const Py_UNUSED(closure))             \
    {                                                                                                                  \
        elimination_object *this;                                                                                      \
        if (ensure_elimination(self, NULL, &this) < 0)                                                                 \
        {                                                                                                              \
            return NULL;                                                                                               \
        }                                                                                                              \
        return expr;                                                                                                   \
    }

ELIMINATION_GETTER(n_blocks, PyLong_FromUnsignedLongLong(hybsol_elimination_n_blocks(this->graph)))
ELIMINATION_GETTER(n_columns, PyLong_FromUnsignedLongLong(hybsol_elimination_n_columns(this->graph)))
ELIMINATION_GETTER(n_operations, PyLong_FromUnsignedLongLong(hybsol_elimination_n_operations(this->graph)))
ELIMINATION_GETTER(n_levels, PyLong_FromUnsignedLongLong(hybsol_elimination_n_levels(this->graph)))
ELIMINATION_GETTER(value_bytes, PyLong_FromSize_t(hybsol_elimination_value_bytes(this->graph)))
ELIMINATION_GETTER(total_bytes, PyLong_FromSize_t(hybsol_elimination_total_bytes(this->graph)))

static PyObject *elimination_object_get_precision(PyObject *const self, void *const Py_UNUSED(closure))
{
    elimination_object *this;
    if (ensure_elimination(self, NULL, &this) < 0)
    {
        return NULL;
    }
    return hybsol_precision_member(hybsol_elimination_precision(this->graph));
}

static PyObject *elimination_object_get_failing_block(PyObject *const self, void *const Py_UNUSED(closure))
{
    elimination_object *this;
    if (ensure_elimination(self, NULL, &this) < 0)
    {
        return NULL;
    }
    const uint64_t block = hybsol_elimination_failing_block(this->graph);
    if (block == UINT64_MAX)
    {
        Py_RETURN_NONE;
    }
    return PyLong_FromUnsignedLongLong(block);
}

/* ------------------------------------------------------------------------- */
/* Type definition                                                            */
/* ------------------------------------------------------------------------- */

#define ELIMINATION_METHOD(name, fn, doc)                                                                              \
    {                                                                                                                  \
        .ml_name = name,                                                                                               \
        .ml_meth = (void *)(fn),                                                                                       \
        .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,                                                       \
        .ml_doc = (doc),                                                                                               \
    }

PyType_Spec elimination_type_spec = {
    .name = MODULE_TYPE_NAME(Elimination),
    .basicsize = sizeof(elimination_object),
    .itemsize = 0,
    .flags = Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_DEFAULT,
    .slots =
        (PyType_Slot[]){
            {Py_tp_traverse, heap_type_traverse_type},
            {Py_tp_dealloc, elimination_dealloc},
            {Py_tp_repr, elimination_repr},
            {Py_tp_doc, (void *)elimination_docstring},
            {Py_tp_methods,
             (PyMethodDef[]){
                 ELIMINATION_METHOD("row_columns", elimination_object_row_columns,
                                    elimination_object_row_columns_docstring),
                 ELIMINATION_METHOD("level_rows", elimination_object_level_rows,
                                    elimination_object_level_rows_docstring),
                 ELIMINATION_METHOD(
                     "row_length", elimination_object_row_length,
                     "row_length(row: int) -> int\nBlock row ``row`` holds this many blocks at the end.\n"),
                 ELIMINATION_METHOD("row_n_eliminations", elimination_object_row_n_eliminations,
                                    "row_n_eliminations(row: int) -> int\nHow many eliminations ``row`` performs.\n"),
                 ELIMINATION_METHOD("row_first_level", elimination_object_row_first_level,
                                    "row_first_level(row: int) -> int\nFirst pass ``row`` is processed in.\n"),
                 ELIMINATION_METHOD("row_level", elimination_object_row_level,
                                    "row_level(row: int) -> int\nLast pass ``row`` is processed in.\n"),
                 {},
             }},
            {Py_tp_getset,
             (PyGetSetDef[]){
                 {
                     .name = "n_blocks",
                     .get = elimination_object_get_n_blocks,
                     .doc = "int : Number of blocks.",
                 },
                 {
                     .name = "n_columns",
                     .get = elimination_object_get_n_columns,
                     .doc = "int : Blocks the final pattern holds, fill-in included.",
                 },
                 {
                     .name = "n_operations",
                     .get = elimination_object_get_n_operations,
                     .doc = "int : Operations a factorization of this system records.",
                 },
                 {
                     .name = "n_levels",
                     .get = elimination_object_get_n_levels,
                     .doc = "int : Passes the elimination takes.",
                 },
                 {
                     .name = "value_bytes",
                     .get = elimination_object_get_value_bytes,
                     .doc = "int : Bytes of block storage a decomposition needs.",
                 },
                 {
                     .name = "total_bytes",
                     .get = elimination_object_get_total_bytes,
                     .doc = "int : Bytes this graph itself occupies.",
                 },
                 {
                     .name = "precision",
                     .get = elimination_object_get_precision,
                     .doc = "hybsol.Precision : The precision the analyzed system stores.",
                 },
                 {
                     .name = "failing_block",
                     .get = elimination_object_get_failing_block,
                     .doc = "int or None : Block whose diagonal is identically zero.",
                 },
                 {},
             }},
            {},
        },
};
