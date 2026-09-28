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
} block_system_object;

MODULE_INTERNAL
extern PyType_Spec block_system_type_spec;
