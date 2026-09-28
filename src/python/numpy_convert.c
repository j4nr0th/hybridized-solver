#include "numpy_convert.h"

hybsol_matrix_t hybsol_matrix_from_array(const PyArrayObject *const arr)
{
    const int ndim = PyArray_NDIM(arr);
    CPYUTL_ASSERT(ndim >= 1 && ndim <= 2, "Expected a 1D or 2D array, got a %dD array.", ndim);
    CPYUTL_ASSERT(check_input_array(arr, 0, (const npy_intp[]){}, NPY_DOUBLE,
                                    NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, "matrix") == 0,
                  "Array did not have correct dtype and/or flags!");
    return hybsol_matrix_view((uint64_t)PyArray_DIM(arr, 0), ndim == 2 ? (uint64_t)PyArray_DIM(arr, 1) : 1,
                              PyArray_DATA(arr));
}

PyArrayObject *hybsol_matrix_to_array(const hybsol_matrix_t *const mat)
{
    const npy_intp dims[2] = {(npy_intp)mat->rows, (npy_intp)mat->cols};
    PyArrayObject *const arr = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!arr)
    {
        return NULL;
    }
    memcpy(PyArray_DATA(arr), mat->data, (size_t)(mat->rows * mat->cols) * sizeof(double));
    return arr;
}

int hybsol_double_array(PyObject *const obj, const int ndim, const npy_intp *const dims, const char *const name,
                        PyArrayObject **const arr_out)
{
    PyArrayObject *const arr =
        (PyArrayObject *)PyArray_FROMANY(obj, NPY_DOUBLE, ndim, ndim, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!arr)
    {
        return -1;
    }
    if (check_input_array(arr, (unsigned)ndim, dims, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, name) < 0)
    {
        Py_DECREF(arr);
        return -1;
    }
    *arr_out = arr;
    return 0;
}

int hybsol_uint64_array(PyObject *const obj, const int ndim, const npy_intp *const dims, const char *const name,
                        PyArrayObject **const arr_out)
{
    PyArrayObject *const arr =
        (PyArrayObject *)PyArray_FROMANY(obj, NPY_UINT64, ndim, ndim, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!arr)
    {
        return -1;
    }
    if (check_input_array(arr, (unsigned)ndim, dims, NPY_UINT64, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, name) < 0)
    {
        Py_DECREF(arr);
        return -1;
    }
    *arr_out = arr;
    return 0;
}

int hybsol_prepare_output(PyArrayObject *const out, const int ndim, const npy_intp *const dims, const int dtype,
                          const int flags, const char *const name, PyArrayObject **const arr_out)
{
    if (!out)
    {
        *arr_out = (PyArrayObject *)PyArray_SimpleNew(ndim, dims, dtype);
        return *arr_out ? 0 : -1;
    }

    Py_INCREF(out);
    if (check_input_array(out, (unsigned)ndim, dims, dtype,
                          NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE | flags, name) < 0)
    {
        Py_DECREF(out);
        return -1;
    }
    *arr_out = out;
    return 0;
}
