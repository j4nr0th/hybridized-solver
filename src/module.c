#define PY_ARRAY_UNIQUE_SYMBOL _mod // TODO: change this symbol to be unique per module
#include "module.h"

//  Numpy
#include <numpy/npy_no_deprecated_api.h>

static void free_module_state(void *module)
{
    module_state_t *const module_state = (module_state_t *)PyModule_GetState(module);
    *module_state = (module_state_t){};
}

static int module_add_steal(PyObject *mod, const char *name, PyObject *obj)
{
    if (!obj)
        return -1;
    const int res = PyModule_AddObjectRef(mod, name, obj);
    Py_XDECREF(obj);
    return res;
}

static int module_exec(PyObject *mod)
{
    if (PyArray_ImportNumPyAPI() < 0)
        return -1;

    module_state_t *const module_state = (module_state_t *)PyModule_GetState(mod);
    if (!module_state)
    {
        return -1;
    }
    (void)module_add_steal;

    return 0;
}

PyModuleDef interplib_module = {
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

PyMODINIT_FUNC PyInit__mod(void) // TODO: rename to match module name
{
    import_array();

    return PyModuleDef_Init(&interplib_module);
}

int heap_type_traverse_type(PyObject *self, const visitproc visit, void *arg)
{
    Py_VISIT(Py_TYPE(self));
    return 0;
}
