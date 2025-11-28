#pragma once

#include "module.h"

typedef struct
{
    PyObject_VAR_HEAD;
    u64 row;
    u64 col;
    u64 rows;
    u64 cols;
    f64 val[];
} block_info_object;

MODULE_INTERNAL
extern PyType_Spec block_info_type_spec;
