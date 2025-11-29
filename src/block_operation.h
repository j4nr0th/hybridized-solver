#pragma once

#include "block_system.h"
#include "blocks.h"

typedef struct
{
    PyObject_VAR_HEAD;
    block_system_object *system;
    operation_t operations[];
} block_decomposition_object;

MODULE_INTERNAL
extern PyType_Spec block_decomposition_type_spec;
