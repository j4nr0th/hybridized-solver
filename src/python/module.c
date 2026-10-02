/**
 * @file src/python/module.c
 * Module definition and error mapping of ``hybsol._mod``.
 */

#define PY_ARRAY_UNIQUE_SYMBOL _mod
#include "module.h"

#include "decomposition_type.h"
#include "elimination_type.h"

#include "block_system_type.h"

PyObject *hybsol_exception_type(PyTypeObject *const type, const hybsol_result_t res)
{
    switch (res)
    {
    case HYBSOL_ERROR_OUT_OF_MEMORY:
        return PyExc_MemoryError;

    // Distinguishing a singular system lets a caller retry at a different block
    // granularity without swallowing every other ValueError. The exception is
    // per module instance, so it is looked up through this interpreter's own
    // state rather than a static that every interpreter would share.
    case HYBSOL_ERROR_SINGULAR: {
        const module_state_t *const state = module_state_from_type(type);
        if (state == NULL || state->exc_singular == NULL)
            return PyExc_ValueError;
        return state->exc_singular;
    }

    // A device that could not be reached, lacked a capability or failed at
    // runtime is not a condition of the caller's data: it is the machine.
    case HYBSOL_ERROR_NO_DEVICE:
    case HYBSOL_ERROR_DEVICE_CAPABILITY:
    case HYBSOL_ERROR_DEVICE: {
        const module_state_t *const state = module_state_from_type(type);
        if (state == NULL || state->exc_device == NULL)
            return PyExc_RuntimeError;
        return state->exc_device;
    }

    // Everything else is a condition detected in the data, not a caller mistake.
    default:
        return PyExc_ValueError;
    }
}

PyObject *hybsol_raise(PyTypeObject *const type, const char *const what, const hybsol_result_t res)
{
    PyErr_Format(hybsol_exception_type(type, res), "%s: %s", what, hybsol_result_str(res));
    return NULL;
}

PyObject *hybsol_raise_block(PyTypeObject *const type, const char *const what, const hybsol_result_t res,
                             const uint64_t failing_block)
{
    if (failing_block != UINT64_MAX)
    {
        PyErr_Format(hybsol_exception_type(type, res), "%s: %s (block %llu)", what, hybsol_result_str(res),
                     (unsigned long long)failing_block);
        return NULL;
    }
    return hybsol_raise(type, what, res);
}

PyObject *hybsol_precision_member(const hybsol_precision_t precision)
{
    PyObject *const mod = PyImport_ImportModule("hybsol");
    if (mod == NULL)
    {
        return NULL;
    }
    PyObject *const cls = PyObject_GetAttrString(mod, "Precision");
    Py_DECREF(mod);
    if (cls == NULL)
    {
        return NULL;
    }

    PyObject *const value = PyUnicode_FromString(precision == HYBSOL_PRECISION_SINGLE ? "single" : "double");
    if (value == NULL)
    {
        Py_DECREF(cls);
        return NULL;
    }

    PyObject *const member = PyObject_CallOneArg(cls, value);
    Py_DECREF(value);
    Py_DECREF(cls);
    return member;
}

int hybsol_check_n_threads(const Py_ssize_t n_threads)
{
    if (n_threads < 0)
    {
        PyErr_SetString(PyExc_ValueError, "Number of threads must be non-negative.");
        return -1;
    }
    return 0;
}

/*
 * Teardown. ``exc_singular`` is *borrowed* by the state: PyModule_AddObject
 * stole the reference the exception was created with, so the module dict owns
 * it and releases it. Dropping one here would free the type while the dict
 * still points at it, which is a use-after-free at shutdown rather than
 * anything visible during a run. Clearing the borrow is all that is left.
 */
static void free_module_state(void *const module)
{
    module_state_t *const module_state = (module_state_t *)PyModule_GetState(module);
    *module_state = (module_state_t){};
}

/**
 * The capsule a backend's Python module reads to drive this one. The table
 * points at functions that already exist; this only publishes them.
 */
PyObject *hybsol_backend_api_capsule(void)
{
    static hybsol_python_api_t api = {
        .decomposition_alloc = decomposition_alloc,
        .require_valid_system = require_valid_system,
        .check_n_threads = hybsol_check_n_threads,
        .parse_precision = parse_precision,
    };
    return PyCapsule_New(&api, "hybsol._mod.backend_api", NULL);
}

static int module_exec(PyObject *const mod)
{
    if (PyArray_ImportNumPyAPI() < 0)
    {
        return -1;
    }

    module_state_t *const module_state = (module_state_t *)PyModule_GetState(mod);
    if (!module_state)
    {
        PyErr_SetString(PyExc_RuntimeError, "hybsol._mod has no module state.");
        return -1;
    }

    module_state->type_block_system = cpyutl_add_type_from_spec_to_module(mod, &block_system_type_spec, NULL);
    if (!module_state->type_block_system)
    {
        return -1;
    }

    module_state->type_decomposition = cpyutl_add_type_from_spec_to_module(mod, &decomposition_type_spec, NULL);
    if (!module_state->type_decomposition)
    {
        return -1;
    }

    module_state->type_elimination = cpyutl_add_type_from_spec_to_module(mod, &elimination_type_spec, NULL);
    if (!module_state->type_elimination)
    {
        return -1;
    }

    // On the module so `hybsol.SingularSystemError` resolves; the dict owns it.
    module_state->exc_singular = PyErr_NewException("hybsol.SingularSystemError", PyExc_ValueError, NULL);
    if (module_state->exc_singular == NULL)
    {
        return -1;
    }
    if (PyModule_AddObject(mod, "SingularSystemError", module_state->exc_singular) < 0)
    {
        Py_DECREF(module_state->exc_singular);
        module_state->exc_singular = NULL;
        return -1;
    }

    // On the module so `hybsol.DeviceError` resolves; the dict owns it.
    module_state->exc_device = PyErr_NewException("hybsol.DeviceError", PyExc_RuntimeError, NULL);
    if (module_state->exc_device == NULL)
    {
        return -1;
    }
    if (PyModule_AddObject(mod, "DeviceError", module_state->exc_device) < 0)
    {
        Py_DECREF(module_state->exc_device);
        module_state->exc_device = NULL;
        return -1;
    }

    if (PyModule_AddObject(mod, "backend_api", hybsol_backend_api_capsule()) < 0)
    {
        return -1;
    }

    return 0;
}

PyModuleDef hybsol_module_def = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "hybsol._mod",
    .m_doc = "Extension to improve performance of the hybridized solver.",
    .m_size = sizeof(module_state_t),
    .m_free = free_module_state,
    .m_slots =
        (PyModuleDef_Slot[]){
            {.slot = Py_mod_exec, .value = module_exec},
            {.slot = Py_mod_multiple_interpreters, .value = Py_MOD_MULTIPLE_INTERPRETERS_SUPPORTED},
            {},
        },
};

PyMODINIT_FUNC PyInit__mod(void)
{
    return PyModuleDef_Init(&hybsol_module_def);
}

int heap_type_traverse_type(PyObject *const self, const visitproc visit, void *const arg)
{
    return cpyutl_traverse_heap_type(self, visit, arg);
}
