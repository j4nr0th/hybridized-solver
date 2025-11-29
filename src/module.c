#define PY_ARRAY_UNIQUE_SYMBOL _mod // TODO: change this symbol to be unique per module
#include "module.h"

//  Numpy
#include "block_info.h"
#include "block_operation.h"
#include "block_system.h"

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
    if (!module_state ||
        (module_state->type_block_info = cpyutl_add_type_from_spec_to_module(mod, &block_info_type_spec, NULL)) ==
            NULL ||
        (module_state->type_block_system = cpyutl_add_type_from_spec_to_module(mod, &block_system_type_spec, NULL)) ==
            NULL ||
        (module_state->type_block_decomposition =
             cpyutl_add_type_from_spec_to_module(mod, &block_decomposition_type_spec, NULL)) == NULL)
    {
        return -1;
    }
    (void)module_add_steal;

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

PyMODINIT_FUNC PyInit__mod(void) // TODO: rename to match module name
{
    import_array();

    return PyModuleDef_Init(&hybsol_module_def);
}

int heap_type_traverse_type(PyObject *self, const visitproc visit, void *arg)
{
    Py_VISIT(Py_TYPE(self));
    return 0;
}
