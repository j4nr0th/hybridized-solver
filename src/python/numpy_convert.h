/**
 * @file src/python/numpy_convert.h
 * Conversion between :c:type:`hybsol_matrix_t` and NumPy arrays.
 *
 * Unless a function says otherwise, it demands a C-contiguous, aligned array of
 * a fixed dtype, and a ``dims`` entry of ``0`` means "any size".
 */

#pragma once

#include "module.h"

/** View a validated array as a matrix; the view borrows ``arr``'s buffer. */
hybsol_matrix_t hybsol_matrix_from_array(const PyArrayObject *arr);

/** Single-precision twin of :c:func:`hybsol_matrix_from_array`. */
hybsol_fmatrix_t hybsol_fmatrix_from_array(const PyArrayObject *arr);

PyArrayObject *hybsol_matrix_to_array(const hybsol_matrix_t *mat);

/**
 * Wrap a matrix's storage as a writable array.
 *
 * ``base`` is recorded as the array's base object, so the storage stays alive
 * for exactly as long as the array can be written through. One reference to it
 * is consumed whether the wrap succeeds or not.
 */
PyArrayObject *hybsol_matrix_wrap(const hybsol_matrix_t *mat, PyObject *base);

PyArrayObject *hybsol_fmatrix_to_array(const hybsol_fmatrix_t *mat);

/** Single-precision twin of :c:func:`hybsol_matrix_wrap`. */
PyArrayObject *hybsol_fmatrix_wrap(const hybsol_fmatrix_t *mat, PyObject *base);

/**
 * :c:func:`PyArray_FROMANY` plus the input checks in one step, so callers
 * cannot forget the validation or leak the array when it fails.
 *
 * ``dims`` holds one required size per dimension, ``0`` meaning "any".
 */
int hybsol_double_array(PyObject *obj, int ndim, const npy_intp *dims, const char *name, PyArrayObject **arr_out);

/**
 * Single-precision twin of :c:func:`hybsol_double_array`, with one deliberate
 * difference: it casts rather than refusing, so a ``double`` array handed to a
 * single-precision system is narrowed on the way in. The C API keeps the
 * strict rule -- there the caller picks a spelling, and picking the wrong one
 * is an error.
 */
int hybsol_float_array(PyObject *obj, int ndim, const npy_intp *dims, const char *name, PyArrayObject **arr_out);

int hybsol_uint64_array(PyObject *obj, int ndim, const npy_intp *dims, const char *name, PyArrayObject **arr_out);

/**
 * Like :c:func:`hybsol_uint64_array`, but accepts anything NumPy can safely
 * cast to ``int64`` (so ``np.arange(n)`` works), rejecting negative values.
 */
int hybsol_index_array(PyObject *obj, int ndim, const npy_intp *dims, const char *name, PyArrayObject **arr_out);

/**
 * Validate (or allocate) the ``out=`` parameter shared by several methods.
 *
 * When ``out`` is ``NULL`` a new array of the given shape is allocated,
 * otherwise it is checked against ``dtype``, ``dims`` and ``flags``.
 */
int hybsol_prepare_output(PyArrayObject *out, int ndim, const npy_intp *dims, int dtype, int flags, const char *name,
                          PyArrayObject **arr_out);

/**
 * Like :c:func:`hybsol_uint64_array` but for a writable ``uint8`` buffer of at
 * least ``min_bytes`` bytes, i.e. storage the library writes into.
 */
int hybsol_byte_array(PyObject *obj, int ndim, size_t min_bytes, const char *name, PyArrayObject **arr_out);

/**
 * Read an optional output array keyword.
 *
 * Yields ``NULL`` when the keyword was omitted or is ``None``, else a new
 * reference to the given array.
 */
int hybsol_optional_array(PyObject *obj, PyArrayObject **out);
