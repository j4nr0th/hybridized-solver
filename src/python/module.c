/**
 * @file src/python/module.c
 * Module definition and error mapping of ``hybsol._mod``.
 */

#define PY_ARRAY_UNIQUE_SYMBOL _mod
#include "module.h"

#include "block_system_type.h"

// Set during module exec; `hybsol_exception_type` only gets a result code.
static PyObject *hybsol_singular_error = NULL;

PyObject *hybsol_exception_type(const hybsol_result_t res)
{
    switch (res)
    {
    case HYBSOL_SUCCESS:
    case HYBSOL_ERROR_INTERNAL:
        return PyExc_RuntimeError;

    case HYBSOL_ERROR_OUT_OF_MEMORY:
        return PyExc_MemoryError;

    case HYBSOL_ERROR_NOT_DECOMPOSED:
    case HYBSOL_ERROR_ALREADY_DECOMPOSED:
        return PyExc_RuntimeError;

    // A singular system is worth distinguishing: a caller that wants to fall
    // back to a different block granularity can catch it without also swallowing
    // every other ValueError.
    case HYBSOL_ERROR_SINGULAR:
        return hybsol_singular_error != NULL ? hybsol_singular_error : PyExc_ValueError;

    // Everything else is a condition detected in the data rather than a
    // caller mistake (an empty row, too many colors), so it is a ValueError
    // rather than some more exotic type.
    default:
        return PyExc_ValueError;
    }
}

PyObject *hybsol_raise(const char *const what, const hybsol_result_t res)
{
    PyErr_Format(hybsol_exception_type(res), "%s: %s", what, hybsol_result_str(res));
    return NULL;
}

PyObject *hybsol_raise_block(const char *const what, const hybsol_result_t res, const hybsol_system_t *const sys)
{
    const uint64_t block = sys == NULL ? UINT64_MAX : hybsol_system_failing_block(sys);
    if (block != UINT64_MAX)
    {
        PyErr_Format(hybsol_exception_type(res), "%s: %s (block %llu)", what, hybsol_result_str(res),
                     (unsigned long long)block);
        return NULL;
    }
    return hybsol_raise(what, res);
}

static void free_module_state(void *const module)
{
    Py_CLEAR(hybsol_singular_error);
    module_state_t *const module_state = (module_state_t *)PyModule_GetState(module);
    *module_state = (module_state_t){};
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
    hybsol_singular_error = module_state->exc_singular;

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
