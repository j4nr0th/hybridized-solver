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
    uint n;             // Number of blocks (rows and cols)
    u64 *block_offsets; // Array of block sizes (length n + 1)
    sys_row_t *rows;    // Array of actual rows (length n)
} block_system_t;

typedef enum
{
    OPERATION_INVDIA,
    OPERATION_ELIMIN,
} operation_type_t;

typedef struct
{
    u64 idx;
} operation_invdia_t;

typedef struct
{
    u64 idx_row;
    u64 idx_col;
} operation_elimin_t;

typedef struct
{
    operation_type_t type;
    union {
        operation_invdia_t invdia;
        operation_elimin_t elimin;
    };
} operation_t;

typedef enum
{
    ORDERING_FIRST,
    ORDERING_GREEDY,
    ORDERING_BALANCED,
} ordering_strategy_t;

MODULE_INTERNAL
int block_system_add_block(const block_system_t *this, u64 idx_row, u64 idx_col, u64 nv, const f64 vals[static nv]);

MODULE_INTERNAL
int block_system_is_valid(const block_system_t *this);

MODULE_INTERNAL
u64 row_array_find_first_geq(u64 size, const row_entry_t *const array[static size], u64 val);

MODULE_INTERNAL
result_t block_system_get_block(const block_system_t *this, u64 idx_row, u64 idx_col, matrix_t *out);

MODULE_INTERNAL
PyArrayObject *matrix_to_array(const matrix_t *mat);

MODULE_INTERNAL
matrix_t matrix_from_array(const PyArrayObject *arr);

MODULE_INTERNAL
void matrix_multiply(const matrix_t *a, const matrix_t *b, const matrix_t *out);

MODULE_INTERNAL
void matrix_multiply_sub_inplace(const matrix_t *a, const matrix_t *b, const matrix_t *out);

MODULE_INTERNAL
void matrix_subtract_inplace(const matrix_t *a, const matrix_t *b);

MODULE_INTERNAL
void matrix_lu_decompose(const matrix_t *m);

MODULE_INTERNAL
void matrix_lu_solve(const matrix_t *m, const matrix_t *b, const matrix_t *out);

MODULE_INTERNAL
result_t block_system_decompose(const block_system_t *this, u64 *pn_ops, operation_t **pp_ops, uint n_threads);

MODULE_INTERNAL
void block_system_apply_diagonal_inverse(const block_system_t *this, u64 idx_row);

MODULE_INTERNAL
void block_system_decompose_diagonal(const block_system_t *this, u64 idx_row);

MODULE_INTERNAL
result_t block_system_eliminate_row(const block_system_t *this, u64 idx_tgt, u64 idx_src);

MODULE_INTERNAL
void block_system_apply_operations(const block_system_t *this, u64 nops, const operation_t ops[static nops], f64 *vec);

MODULE_INTERNAL
void block_system_solve_u(const block_system_t *this, f64 *y);

static inline u64 block_system_get_block_size(const block_system_t *this, const u64 idx_row)
{
    return this->block_offsets[idx_row + 1] - this->block_offsets[idx_row];
}

MODULE_INTERNAL
result_t block_system_compute_ordering_first(u64 n, const sys_row_t rows[const static n], u64 ordering[const n],
                                             u64 max_colors, u64 color_buffer[2 * max_colors]);

MODULE_INTERNAL
result_t block_system_compute_ordering_greedy(u64 n, const sys_row_t rows[const static n], u64 ordering[const n],
                                              u64 max_colors, u64 color_buffer[2 * max_colors]);

MODULE_INTERNAL
result_t block_system_compute_ordering_balanced(u64 n, const sys_row_t rows[const static n], u64 ordering[const n],
                                                u64 max_colors, u64 color_buffer[2 * max_colors]);
