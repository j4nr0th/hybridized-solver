/**
 * @file src/python/block_system_type.h
 * The :class:`hybsol.BlockSystem` extension type.
 */

#pragma once

#include "module.h"

/** Python object wrapping an opaque :c:type:`hybsol_system_t`. */
typedef struct
{
    PyObject_HEAD;
    hybsol_system_t *system;
    /**
     * Number of arrays handed out by ``block_storage()`` that are still alive,
     * so methods rebuilding a row's storage can refuse to run while a caller
     * may still be writing through a pointer into it.
     */
    Py_ssize_t n_live_views;
} block_system_object;

MODULE_INTERNAL
extern PyType_Spec block_system_type_spec;

/** Raise :exc:`ValueError` unless ``system`` is structurally valid. */
MODULE_INTERNAL
int require_valid_system(const hybsol_system_t *system, const char *what);

/** Read a ``precision=`` argument; see the definition for the accepted forms. */
MODULE_INTERNAL
int parse_precision(PyObject *value, hybsol_precision_t *out);
