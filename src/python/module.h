/**
 * @file src/python/module.h
 * Shared declarations of the hybsol Python extension module.
 *
 * Everything that touches Python or NumPy lives under ``src/python``; the
 * solver itself lives under ``src/core`` and is completely independent of
 * both.
 */

#pragma once

#ifdef __GNUC__
#define MODULE_INTERNAL __attribute__((visibility("hidden")))
#endif

#ifndef MODULE_INTERNAL
#define MODULE_INTERNAL
#endif

// Python ssize define
#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#endif

#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif

// Prevent numpy from being re-imported
#ifndef PY_LIMITED_API
#define PY_LIMITED_API 0x030A0000
#endif

#ifndef PY_ARRAY_UNIQUE_SYMBOL
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL _mod
#endif

#include <Python.h>
#include <numpy/ndarrayobject.h>

// This must be after the NumPy include
#include <cpyutl.h>

#include <hybsol/hybsol.h>

/** Per-module state of ``hybsol._mod``. */
typedef struct
{
    PyTypeObject *type_block_system;
    PyTypeObject *type_decomposition;
    PyTypeObject *type_elimination;
    /** ``hybsol.SingularSystemError``; borrowed by the state, owned by the module. */
    PyObject *exc_singular;
} module_state_t;

MODULE_INTERNAL
extern PyModuleDef hybsol_module_def;

/**
 * Fetch the module state belonging to a type defined by this module.
 *
 * :param type: Type created from :c:data:`hybsol_module_def`.
 * :returns: The state, or ``NULL`` with an exception set.
 */
static inline const module_state_t *module_state_from_type(PyTypeObject *type)
{
    PyObject *const mod = PyType_GetModuleByDef(type, &hybsol_module_def);
    if (!mod)
    {
        return NULL;
    }
    return PyModule_GetState(mod);
}

MODULE_INTERNAL
int heap_type_traverse_type(PyObject *self, visitproc visit, void *arg);

/**
 * Map a :c:type:`hybsol_result_t` onto the exception type used for it.
 *
 * :param res: Result code to translate.
 * :returns: A borrowed reference to an exception *type* object.
 */
MODULE_INTERNAL
PyObject *hybsol_exception_type(hybsol_result_t res);

/**
 * Raise the exception associated with a result code.
 *
 * The message is ``"<what>: <hybsol_result_str(res)>"``.
 *
 * :param what: Short description of the operation that failed.
 * :param res: The result code; :c:enumerator:`HYBSOL_SUCCESS` is treated as
 *     :c:enumerator:`HYBSOL_ERROR_INTERNAL`, since nothing failed otherwise.
 * :returns: Always ``NULL``, so callers can ``return hybsol_raise(...)``.
 */
MODULE_INTERNAL
PyObject *hybsol_raise(const char *what, hybsol_result_t res);

/**
 * Raise the exception for a result code, naming the failing block when known.
 *
 * :param what: Short description of the operation that failed.
 * :param res: The result code.
 * :param failing_block: Block index to name, or ``UINT64_MAX`` for none.
 * :returns: Always ``NULL``, so callers can ``return hybsol_raise_block(...)``.
 */
MODULE_INTERNAL
PyObject *hybsol_raise_block(const char *what, const hybsol_result_t res, uint64_t failing_block);

/**
 * Reject a negative thread count.
 *
 * :param n_threads: The value the caller parsed from the arguments.
 * :returns: ``0`` on success, ``-1`` with a :exc:`ValueError` set.
 */
MODULE_INTERNAL
int hybsol_check_n_threads(Py_ssize_t n_threads);

/**
 * Build the :class:`hybsol.Precision` member for a core precision.
 *
 * :param precision: The precision to name.
 * :returns: A new reference, or ``NULL`` with an exception set.
 */
MODULE_INTERNAL
PyObject *hybsol_precision_member(hybsol_precision_t precision);

#ifndef MODULE_TYPE_NAME
#define MODULE_TYPE_NAME(name) ("hybsol._mod." #name)
#endif
