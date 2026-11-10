/**
 * @file src/python/decomposition_type.h
 * The :class:`hybsol.Decomposition` extension type.
 */

#pragma once

#include "module.h"

/** Python object wrapping an opaque :c:type:`hybsol_decomposition_t`. */
typedef struct
{
    PyObject_HEAD;
    hybsol_decomposition_t *decomposition;
} decomposition_object;

MODULE_INTERNAL
extern PyType_Spec decomposition_type_spec;

/** Allocate a :class:`hybsol.Decomposition` taking over ``decomposition``. */
MODULE_INTERNAL
PyObject *decomposition_alloc(PyTypeObject *type, hybsol_decomposition_t *decomposition);
