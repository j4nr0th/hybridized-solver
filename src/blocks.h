#pragma once

#include "module.h"

typedef struct
{
    u64 rows;
    u64 cols;
    f64 *data;
} matrix_t;

typedef struct
{
    u64 col;
    f64 vals[];
} row_entry_t;

typedef struct
{
    u64 count;
    u64 capacity;
    row_entry_t **entries;
} sys_row_t;

typedef struct
{
    uint n;           // Number of blocks (rows and cols)
    u64 *block_sizes; // Array of block sizes (length n)
    sys_row_t *rows;  // Array of actual rows (length n)
} block_system_t;

MODULE_INTERNAL
int block_system_add_block(const block_system_t *this, u64 idx_row, u64 idx_col, u64 nv, const f64 vals[static nv]);

MODULE_INTERNAL
int block_system_is_valid(const block_system_t *this);

// TODO: test this one
MODULE_INTERNAL
u64 row_array_find_first_geq(u64 size, const row_entry_t *const array[static size], u64 val);
