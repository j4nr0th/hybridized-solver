#pragma once

#include "blocks.h"
#include "module.h"

typedef struct
{
    PyObject_HEAD;
    block_system_t system;
    u64 nops;
    operation_t *ops;
} block_system_object;

MODULE_INTERNAL
extern PyType_Spec block_system_type_spec;
