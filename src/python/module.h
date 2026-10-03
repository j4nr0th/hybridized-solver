/**
 * @file src/python/module.h
 * Shared declarations of the hybsol Python extension module.
 *
 * Everything that touches Python or NumPy lives under ``src/python``; the
 * solver itself lives under ``src/core`` and is independent of both.
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
    /** ``hybsol.DeviceError``; borrowed by the state, owned by the module. */
    PyObject *exc_device;
} module_state_t;

MODULE_INTERNAL
extern PyModuleDef hybsol_module_def;

/** Fetch the module state belonging to a type defined by this module. */
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
 * ``type`` is any type of this module; it identifies the interpreter the
 * exception belongs to. The singular-system and device exceptions are created
 * per module instance, so they are read out of that instance's state rather
 * than from a process-wide pointer a second interpreter would share.
 */
MODULE_INTERNAL
PyObject *hybsol_exception_type(PyTypeObject *type, hybsol_result_t res);

/**
 * Raise the exception associated with a result code as
 * ``"<what>: <hybsol_result_str(res)>"``. Always returns ``NULL``, so callers
 * can ``return hybsol_raise(...)``.
 */
MODULE_INTERNAL
PyObject *hybsol_raise(PyTypeObject *type, const char *what, hybsol_result_t res);

/** As :c:func:`hybsol_raise`, naming the failing block when one is known. */
MODULE_INTERNAL
PyObject *hybsol_raise_block(PyTypeObject *type, const char *what, const hybsol_result_t res, uint64_t failing_block);

/** Reject a negative thread count with a :exc:`ValueError`. */
MODULE_INTERNAL
int hybsol_check_n_threads(Py_ssize_t n_threads);

/** Build the :class:`hybsol.Precision` member for a core precision. */
MODULE_INTERNAL
PyObject *hybsol_precision_member(hybsol_precision_t precision);

/**
 * What a backend's Python module is allowed to drive hybsol through.
 *
 * A backend ships as its own extension module and links its own copy of the
 * core, so it cannot share the bindings' helpers by linking them. Instead
 * ``hybsol._mod`` hands them over in the capsule ``hybsol._mod.backend_api``,
 * which a plugin reads once, when it first needs it.
 */
typedef struct hybsol_python_api
{
    /** Wrap a decomposition in this interpreter's :class:`hybsol.Decomposition`. */
    PyObject *(*decomposition_alloc)(PyTypeObject *decomposition_type, hybsol_decomposition_t *decomposition);
    /** Raise :exc:`ValueError` unless the system is structurally valid. */
    int (*require_valid_system)(const hybsol_system_t *system, const char *what);
    /** Raise :exc:`ValueError` for a negative thread count. */
    int (*check_n_threads)(Py_ssize_t n_threads);
    /**
     * Parse a ``precision=`` argument: :exc:`TypeError` for a non-string,
     * :exc:`ValueError` for a name that is neither ``"double"`` nor
     * ``"single"``.
     */
    int (*parse_precision)(PyObject *value, hybsol_precision_t *out);
} hybsol_python_api_t;

/** Build the capsule ``hybsol._mod.backend_api``, holding a static API table. */
MODULE_INTERNAL
PyObject *hybsol_backend_api_capsule(void);

#ifndef MODULE_TYPE_NAME
#define MODULE_TYPE_NAME(name) ("hybsol._mod." #name)
#endif
