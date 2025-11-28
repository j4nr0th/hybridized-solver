#include "block_info.h"

static PyObject *block_info_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    long row, col;
    PyObject *py_val;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "llO", (char *[]){"row", "col", "val", NULL}, &row, &col, &py_val))
        return NULL;

    PyArrayObject *const val =
        (PyArrayObject *)PyArray_FROMANY(py_val, NPY_DOUBLE, 2, 2, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!val)
        return NULL;

    const npy_intp total_size = PyArray_SIZE(val);
    block_info_object *const self = (block_info_object *)type->tp_alloc(type, total_size);
    if (!self)
    {
        Py_DECREF(val);
        return NULL;
    }
    self->row = row;
    self->col = col;
    self->rows = PyArray_DIM(val, 0);
    self->cols = PyArray_DIM(val, 1);
    memcpy(self->val, PyArray_DATA(val), total_size * sizeof(f64));
    return (PyObject *)self;
}

static PyObject *block_info_get_row(const block_info_object *self, void *Py_UNUSED(closure))
{
    return PyLong_FromUnsignedLongLong(self->row);
}

static PyObject *block_info_get_col(const block_info_object *self, void *Py_UNUSED(closure))
{
    return PyLong_FromUnsignedLongLong(self->col);
}

static PyObject *block_info_get_shape(const block_info_object *self, void *Py_UNUSED(closure))
{
    return cpyutl_output_create_check(CPYOUT_TYPE_TUPLE,
                                      (const cpyutl_output_t[]){
                                          {.type = CPYOUT_TYPE_PYINT, .value_int = (Py_ssize_t)self->rows},
                                          {.type = CPYOUT_TYPE_PYINT, .value_int = (Py_ssize_t)self->cols},
                                          {},
                                      });
}

static PyObject *block_info_get_value(const block_info_object *self, void *Py_UNUSED(closure))
{
    const npy_intp dims[2] = {(npy_intp)self->rows, (npy_intp)self->cols};
    PyArrayObject *const arr = (PyArrayObject *)PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, (npy_double *)self->val);
    if (!arr)
        return NULL;
    if (PyArray_SetBaseObject(arr, (PyObject *)self) < 0)
    {
        Py_DECREF(arr);
        return NULL;
    }
    Py_INCREF(self);
    return (PyObject *)arr;
}

PyDoc_STRVAR(block_info_doc, "BlockInfo(row: int, col: int, val: numpy.typing.ArrayLike)\nType describing a block.\n");

MODULE_INTERNAL
PyType_Spec block_info_type_spec = {
    .name = MODULE_TYPE_NAME(BlockInfo),
    .basicsize = sizeof(block_info_object),
    .itemsize = sizeof(f64),
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_IMMUTABLETYPE,
    .slots =
        (PyType_Slot[]){
            {Py_tp_new, block_info_new},
            {Py_tp_getset,
             (PyGetSetDef[]){
                 {
                     .name = "row",
                     .get = (getter)block_info_get_row,
                     .doc = "The row index of the block.",
                 },
                 {
                     .name = "col",
                     .get = (getter)block_info_get_col,
                     .doc = "The column index of the block.",
                 },
                 {
                     .name = "shape",
                     .get = (getter)block_info_get_shape,
                     .doc = "The shape of the block.",
                 },
                 {
                     .name = "value",
                     .get = (getter)block_info_get_value,
                     .doc = "The block value.",
                 },
                 {},
             }},
            {Py_tp_traverse, heap_type_traverse_type},
            {Py_tp_doc, (void *)block_info_doc},
            {},
        },
};
