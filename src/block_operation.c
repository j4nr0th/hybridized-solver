#include "block_operation.h"

static void block_decomposition_dealloc(block_decomposition_object *self)
{
    PyObject_GC_UnTrack(self);
    PyTypeObject *const type = Py_TYPE(self);
    block_system_object *const system = self->system;
    self->system = NULL;
    Py_XDECREF(system);
    type->tp_free((PyObject *)self);
    Py_DECREF(type);
}

static int ensure_block_decomposition_and_state(PyObject *self, PyTypeObject *defining_class,
                                                const module_state_t **p_state, block_decomposition_object **p_this)
{
    const module_state_t *const state =
        defining_class ? PyType_GetModuleState(defining_class) : module_state_from_type(Py_TYPE(self));
    if (!state)
        return -1;

    if (!PyObject_TypeCheck(self, state->type_block_decomposition))
    {
        PyErr_Format(PyExc_TypeError, "%s expected, got %s.", state->type_block_decomposition->tp_name,
                     Py_TYPE(self)->tp_name);
        return -1;
    }

    *p_state = state;
    *p_this = (block_decomposition_object *)self;
    return 0;
}

static PyObject *block_decomposition_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds)
{
    const module_state_t *state = module_state_from_type(subtype);
    if (!state)
        return NULL;

    block_system_object *system;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!", (char *[]){"", NULL}, state->type_block_system, &system))
        return NULL;

    PyObject *const valid_res = PyObject_CallMethod((PyObject *)system, "is_valid", NULL);
    if (!valid_res)
        return NULL;

    const int is_valid = PyObject_IsTrue(valid_res);
    Py_DECREF(valid_res);

    if (!is_valid)
    {
        PyErr_SetString(PyExc_ValueError, "Block system is not valid.");
        return NULL;
    }

    u64 n_ops;
    operation_t *p_ops;
    const result_t res = block_system_decompose(&system->system, &n_ops, &p_ops);
    if (res != RESULT_SUCCESS)
    {
        PyErr_Format(PyExc_RuntimeError, "Could not decompose the system with error code: %u", res);
        return NULL;
    }

    block_decomposition_object *const self =
        (block_decomposition_object *)subtype->tp_alloc(subtype, (Py_ssize_t)n_ops);
    if (!self)
    {
        PyMem_RawFree(p_ops);
        return NULL;
    }
    self->system = system;
    memcpy(self->operations, p_ops, n_ops * sizeof(operation_t));
    PyMem_RawFree(p_ops);
    Py_INCREF(system);
    return (PyObject *)self;
}

static PyObject *block_decomposition_object_operations(PyObject *self, PyTypeObject *defining_class,
                                                       PyObject *const *Py_UNUSED(args), const Py_ssize_t nargs,
                                                       const PyObject *kwnames)
{
    const module_state_t *state;
    block_decomposition_object *this;
    if (ensure_block_decomposition_and_state(self, defining_class, &state, &this) < 0)
        return NULL;

    if (nargs != 0 || kwnames != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "No arguments accepted.");
        return NULL;
    }

    PyTupleObject *const out = (PyTupleObject *)PyTuple_New(Py_SIZE(this));

    for (u64 i = 0; i < (u64)Py_SIZE(this); ++i)
    {
        const operation_t op = this->operations[i];
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

PyDoc_STRVAR(block_decomposition_object_docstring, "operations() -> tuple[tuple[int, int] | tuple[int]]\n"
                                                   "Get operations as tuples of one or two ints.");

PyObject *block_decomposition_get_system(PyObject *self, void *Py_UNUSED(closure))
{
    const block_decomposition_object *const this = (block_decomposition_object *)self;
    Py_INCREF(this->system);
    return (PyObject *)this->system;
}

PyType_Spec block_decomposition_type_spec = {
    .name = MODULE_TYPE_NAME(BlockDecomposition),
    .basicsize = sizeof(block_decomposition_object),
    .itemsize = sizeof(operation_t),
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_IMMUTABLETYPE,
    .slots =
        (PyType_Slot[]){
            {Py_tp_new, block_decomposition_new},
            {Py_tp_traverse, heap_type_traverse_type},
            {Py_tp_dealloc, block_decomposition_dealloc},
            {Py_tp_getset,
             (PyGetSetDef[]){
                 {

                     .name = "system",
                     .get = (getter)block_decomposition_get_system,
                     .doc = "BlockSystem : System decomposition was made with.",
                 },
                 {},
             }},
            {Py_tp_methods,
             (PyMethodDef[]){
                 {
                     .ml_name = "operations",
                     .ml_meth = (void *)block_decomposition_object_operations,
                     .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
                     .ml_doc = block_decomposition_object_docstring,
                 },
                 {},
             }},
            {},
        }

};
