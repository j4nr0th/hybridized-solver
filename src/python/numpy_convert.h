/**
 * @file src/python/numpy_convert.h
 * Conversion between :c:type:`hybsol_matrix_t` and NumPy arrays.
 *
 * These are the only functions in the extension that know about both the
 * core's view type and NumPy's array type.
 */

#pragma once

#include "module.h"

/**
 * Read a matrix view out of a validated NumPy array.
 *
 * The array must be ``double``, C-contiguous and aligned; the caller keeps
 * its reference for at least as long as the view is used.
 *
 * :param arr: Array to describe, either 1-D (treated as a column) or 2-D.
 * :returns: A view borrowing ``arr``'s buffer.
 */
hybsol_matrix_t hybsol_matrix_from_array(const PyArrayObject *arr);

/**
 * Read a single-precision matrix view out of a validated NumPy array.
 *
 * The single-precision twin of :c:func:`hybsol_matrix_from_array`.
 *
 * :param arr: Array to describe, either 1-D (treated as a column) or 2-D.
 * :returns: A view borrowing ``arr``'s buffer.
 */
hybsol_fmatrix_t hybsol_fmatrix_from_array(const PyArrayObject *arr);

/**
 * Copy a matrix into a freshly allocated 2-D ``double`` array.
 *
 * :param mat: Matrix to copy.
 * :returns: The new array, or ``NULL`` with :c:macro:`PyExc_MemoryError` set.
 */
PyArrayObject *hybsol_matrix_to_array(const hybsol_matrix_t *mat);

/**
 * Wrap a matrix's existing storage as a writable 2-D ``double`` array.
 *
 * The array does not own its memory: ``base`` is recorded as the array's base
 * object so that whatever the storage belongs to stays alive for exactly as
 * long as the array can be written through.
 *
 * :param mat: Matrix whose buffer is wrapped.
 * :param base: Object owning the storage. One reference to it is consumed
 *     whether the wrap succeeds or not.
 * :returns: The array, or ``NULL`` with an exception set.
 */
PyArrayObject *hybsol_matrix_wrap(const hybsol_matrix_t *mat, PyObject *base);

/**
 * Copy a single-precision matrix into a freshly allocated 2-D ``float`` array.
 *
 * :param mat: Matrix to copy.
 * :returns: The new array, or ``NULL`` with :c:macro:`PyExc_MemoryError` set.
 */
PyArrayObject *hybsol_fmatrix_to_array(const hybsol_fmatrix_t *mat);

/**
 * Wrap a single-precision matrix's existing storage as a writable ``float``
 * array, exactly as :c:func:`hybsol_matrix_wrap` does for doubles.
 *
 * :param mat: Matrix whose buffer is wrapped.
 * :param base: Object owning the storage. One reference to it is consumed
 *     whether the wrap succeeds or not.
 * :returns: The array, or ``NULL`` with an exception set.
 */
PyArrayObject *hybsol_fmatrix_wrap(const hybsol_fmatrix_t *mat, PyObject *base);

/**
 * Convert an object to a ``double`` array with the required shape.
 *
 * This is ``PyArray_FROMANY`` plus :c:func:`check_input_array` in one step,
 * so callers cannot forget the validation (or leak the array when the
 * validation fails, as earlier versions of this module did).
 *
 * :param obj: Anything array-like.
 * :param ndim: Required number of dimensions.
 * :param dims: Required size per dimension; ``0`` means "any". Must have
 *     ``ndim`` entries unless ``ndim`` is ``0``.
 * :param name: Argument name used in the error message.
 * :param arr_out: Receives the new reference.
 * :returns: ``0`` on success, ``-1`` with an exception set on failure.
 */
int hybsol_double_array(PyObject *obj, int ndim, const npy_intp *dims, const char *name, PyArrayObject **arr_out);

/**
 * Convert an object to a ``float`` array with the required shape.
 *
 * The single-precision twin of :c:func:`hybsol_double_array`, used for the
 * block values of a :c:enumerator:`HYBSOL_PRECISION_SINGLE` system.
 *
 * :param obj: Anything array-like.
 * :param ndim: Required number of dimensions.
 * :param dims: Required size per dimension; ``0`` means "any".
 * :param name: Argument name used in the error message.
 * :param arr_out: Receives the new reference.
 * :returns: ``0`` on success, ``-1`` with an exception set on failure.
 */
int hybsol_float_array(PyObject *obj, int ndim, const npy_intp *dims, const char *name, PyArrayObject **arr_out);

/**
 * Convert an object to a C-contiguous ``uint64_t`` array.
 *
 * :param obj: Anything array-like.
 * :param ndim: Required number of dimensions.
 * :param dims: Required size per dimension; ``0`` means "any".
 * :param name: Argument name used in the error message.
 * :param arr_out: Receives the new reference.
 * :returns: ``0`` on success, ``-1`` with an exception set on failure.
 */
int hybsol_uint64_array(PyObject *obj, int ndim, const npy_intp *dims, const char *name, PyArrayObject **arr_out);

/**
 * Convert an object to a C-contiguous ``uint64_t`` array of block indices.
 *
 * Identical to :c:func:`hybsol_uint64_array`, except that anything NumPy can
 * safely cast to ``int64`` is accepted as well (so ``np.arange(n)`` works),
 * with negative values rejected.
 *
 * :param obj: Anything array-like.
 * :param ndim: Required number of dimensions.
 * :param dims: Required size per dimension; ``0`` means "any".
 * :param name: Argument name used in the error message.
 * :param arr_out: Receives the new reference.
 * :returns: ``0`` on success, ``-1`` with an exception set on failure.
 */
int hybsol_index_array(PyObject *obj, int ndim, const npy_intp *dims, const char *name, PyArrayObject **arr_out);

/**
 * Validate (or allocate) the ``out=`` parameter shared by several methods.
 *
 * When ``out`` is ``NULL`` a new array with the given shape is allocated,
 * otherwise it is checked against ``dtype``, ``dims`` and ``flags``. On
 * success a reference is stored in ``*arr_out`` for the caller to own.
 *
 * :param out: The user supplied array, or ``NULL``.
 * :param ndim: Number of dimensions.
 * :param dims: Expected sizes per dimension.
 * :param dtype: Expected NumPy dtype.
 * :param flags: Extra array flags required of a user supplied array.
 * :param name: Argument name used in the error message.
 * :param arr_out: Receives the (new) reference.
 * :returns: ``0`` on success, ``-1`` with an exception set on failure.
 */
int hybsol_prepare_output(PyArrayObject *out, int ndim, const npy_intp *dims, int dtype, int flags, const char *name,
                          PyArrayObject **arr_out);

/**
 * Get a writable 1-D ``uint8`` array of at least ``min_bytes``.
 *
 * Used for buffers the library writes into rather than reads, such as a
 * decomposition workspace, so it demands the writeable flag the read-only
 * input helpers do not.
 *
 * @param obj Object to convert; anything NumPy accepts for a uint8 array.
 * @param ndim Number of dimensions; must be 1.
 * @param min_bytes Smallest acceptable size in bytes.
 * @param name Argument name used in the error message.
 * @param arr_out Receives a new reference to the array.
 * @return 0 on success, -1 with an exception set.
 */
int hybsol_byte_array(PyObject *obj, int ndim, size_t min_bytes, const char *name, PyArrayObject **arr_out);
