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

hybsol_fmatrix_t hybsol_fmatrix_from_array(const PyArrayObject *const arr)
{
    const int ndim = PyArray_NDIM(arr);
    CPYUTL_ASSERT(ndim >= 1 && ndim <= 2, "Expected a 1D or 2D array, got a %dD array.", ndim);
    CPYUTL_ASSERT(check_input_array(arr, 0, (const npy_intp[]){}, NPY_FLOAT, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED,
                                    "matrix") == 0,
                  "Array did not have correct dtype and/or flags!");
    return hybsol_fmatrix_view((uint64_t)PyArray_DIM(arr, 0), ndim == 2 ? (uint64_t)PyArray_DIM(arr, 1) : 1,
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

PyArrayObject *hybsol_matrix_wrap(const hybsol_matrix_t *const mat, PyObject *const base)
{
    const npy_intp dims[2] = {(npy_intp)mat->rows, (npy_intp)mat->cols};
    PyArrayObject *const arr = (PyArrayObject *)PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, mat->data);
    if (!arr)
    {
        Py_DECREF(base);
        return NULL;
    }

    // The array aliases memory it did not allocate: the base object is what
    // keeps that memory (and the system it belongs to) alive.
    if (PyArray_SetBaseObject(arr, base) < 0)
    {
        Py_DECREF(arr);
        return NULL;
    }

    PyArray_ENABLEFLAGS(arr, NPY_ARRAY_WRITEABLE);
    return arr;
}

PyArrayObject *hybsol_fmatrix_to_array(const hybsol_fmatrix_t *const mat)
{
    const npy_intp dims[2] = {(npy_intp)mat->rows, (npy_intp)mat->cols};
    PyArrayObject *const arr = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_FLOAT);
    if (!arr)
    {
        return NULL;
    }
    memcpy(PyArray_DATA(arr), mat->data, (size_t)(mat->rows * mat->cols) * sizeof(float));
    return arr;
}

PyArrayObject *hybsol_fmatrix_wrap(const hybsol_fmatrix_t *const mat, PyObject *const base)
{
    const npy_intp dims[2] = {(npy_intp)mat->rows, (npy_intp)mat->cols};
    PyArrayObject *const arr = (PyArrayObject *)PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, mat->data);
    if (!arr)
    {
        Py_DECREF(base);
        return NULL;
    }

    if (PyArray_SetBaseObject(arr, base) < 0)
    {
        Py_DECREF(arr);
        return NULL;
    }

    PyArray_ENABLEFLAGS(arr, NPY_ARRAY_WRITEABLE);
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

int hybsol_float_array(PyObject *const obj, const int ndim, const npy_intp *const dims, const char *const name,
                       PyArrayObject **const arr_out)
{
    // FORCECAST is what makes a Python caller able to hand a double array to a
    // single-precision system: there is only one door in Python, and it is the
    // system's precision that decides the type. The C API keeps the strict
    // rule instead -- there the caller picks a spelling, and picking the wrong
    // one is an error rather than a silent narrowing.
    PyArrayObject *const arr = (PyArrayObject *)PyArray_FROMANY(
        obj, NPY_FLOAT, ndim, ndim, NPY_ARRAY_FORCECAST | NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!arr)
    {
        return -1;
    }
    if (check_input_array(arr, (unsigned)ndim, dims, NPY_FLOAT, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, name) < 0)
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

int hybsol_index_array(PyObject *const obj, const int ndim, const npy_intp *const dims, const char *const name,
                       PyArrayObject **const arr_out)
{
    static const int flags = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED;

    // A `uint64` array is used as-is. Everything that NumPy can reach with a
    // *safe* cast to `int64` (Python ints, `int64`, smaller integers) is taken
    // through that type instead, so that `np.arange(n)` works just as well as
    // an explicit `dtype=np.uint64` array.
    PyArrayObject *arr = (PyArrayObject *)PyArray_FROMANY(obj, NPY_UINT64, ndim, ndim, flags);
    if (!arr)
    {
        PyErr_Clear();
        arr = (PyArrayObject *)PyArray_FROMANY(obj, NPY_INT64, ndim, ndim, flags);
    }
    if (!arr)
    {
        return -1;
    }
    if (check_input_array(arr, (unsigned)ndim, dims, -1, flags, name) < 0)
    {
        Py_DECREF(arr);
        return -1;
    }

    const Py_ssize_t total = (Py_ssize_t)PyArray_SIZE(arr);
    const int64_t *const vals = (const int64_t *)PyArray_DATA(arr);
    for (Py_ssize_t i = 0; i < total; ++i)
    {
        if (vals[i] < 0)
        {
            PyErr_Format(PyExc_ValueError, "Array %s must not contain negative values, but entry %zd is %lld.", name, i,
                         (long long)vals[i]);
            Py_DECREF(arr);
            return -1;
        }
    }

    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(ndim, PyArray_DIMS(arr), NPY_UINT64);
    if (!out)
    {
        Py_DECREF(arr);
        return -1;
    }
    uint64_t *const dst = (uint64_t *)PyArray_DATA(out);
    for (Py_ssize_t i = 0; i < total; ++i)
    {
        dst[i] = (uint64_t)vals[i];
    }
    Py_DECREF(arr);
    *arr_out = out;
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

int hybsol_byte_array(PyObject *const obj, const int ndim, const size_t min_bytes, const char *const name,
                      PyArrayObject **const arr_out)
{
    /* PyArray_FROMANY satisfies a requested writeable flag by copying a
     * read-only input, which would silently defeat the point of handing in a
     * reusable buffer. Reject that up front instead. */
    if (PyArray_Check(obj) && !PyArray_ISWRITEABLE((PyArrayObject *)obj))
    {
        PyErr_Format(PyExc_ValueError, "Array %s must be writable.", name);
        return -1;
    }

    /* Writable and contiguous: the library writes straight into this buffer. */
    PyArrayObject *const arr = (PyArrayObject *)PyArray_FROMANY(
        obj, NPY_UINT8, ndim, ndim, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE);
    if (!arr)
    {
        return -1;
    }

    if ((size_t)PyArray_SIZE(arr) < min_bytes)
    {
        PyErr_Format(PyExc_ValueError, "Array %s holds %zd bytes but at least %zu are needed.", name,
                     (Py_ssize_t)PyArray_SIZE(arr), min_bytes);
        Py_DECREF(arr);
        return -1;
    }

    *arr_out = arr;
    return 0;
}
