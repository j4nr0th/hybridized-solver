#pragma once

#ifdef __GNUC__
#define MODULE_INTERNAL __attribute__((visibility("hidden")))
#endif

#ifndef MODULE_INTERNAL
#define MODULE_INTERNAL
#endif

//  Python ssize define
#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#endif

#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif

//  Prevent numpy from being re-imported
#ifndef PY_LIMITED_API
#define PY_LIMITED_API 0x030A0000
#endif

#ifndef PY_ARRAY_UNIQUE_SYMBOL
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL _mod // TODO: change this symbol to be unique per module
#endif

#include <Python.h>
#include <numpy/ndarrayobject.h>

// This must be after the NumPy include
#include <cpyutl.h>

typedef struct
{
    PyTypeObject *type_block_info;
    PyTypeObject *type_block_system;
    PyTypeObject *type_block_decomposition;
} module_state_t;

MODULE_INTERNAL
extern PyModuleDef hybsol_module_def;

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

typedef uint64_t u64;
typedef double_t f64;
typedef uint32_t uint;
typedef uint8_t u8;

#ifndef MODULE_TYPE_NAME
#define MODULE_TYPE_NAME(name) ("hybsol._mod." #name)
#endif

typedef enum
{
    RESULT_SUCCESS,
    RESULT_BLOCK_NOT_IN_SYSTEM,
    RESULT_FAILED_ALLOC,
} result_t;
