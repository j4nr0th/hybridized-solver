/**
 * @file src/python/elimination_type.h
 * The :class:`hybsol.Elimination` extension type.
 */

#pragma once

#include "module.h"

/** Python object wrapping an opaque :c:type:`hybsol_elimination_t`. */
typedef struct
{
    PyObject_HEAD;
    hybsol_elimination_t *graph;
} elimination_object;

MODULE_INTERNAL
extern PyType_Spec elimination_type_spec;

/** Allocate an :class:`hybsol.Elimination` taking over ``graph``. */
MODULE_INTERNAL
PyObject *elimination_alloc(PyTypeObject *type, hybsol_elimination_t *graph);
