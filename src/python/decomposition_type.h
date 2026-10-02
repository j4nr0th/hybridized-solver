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

/**
 * Allocate a Python :class:`hybsol.Decomposition` owning ``decomposition``.
 *
 * :param type: The heap type to instantiate.
 * :param decomposition: The decomposition to take over; the caller keeps its
 *     own pointer valid and must not destroy it.
 * :returns: A new reference, or ``NULL`` with an exception set.
 */
MODULE_INTERNAL
PyObject *decomposition_alloc(PyTypeObject *type, hybsol_decomposition_t *decomposition);
