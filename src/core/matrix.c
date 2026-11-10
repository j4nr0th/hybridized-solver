/**
 * @file core/matrix.c
 * Dense matrix kernels: products, differences and LU. ``matrix.inc`` holds them
 * and is included once per public spelling; its header sets out the contract.
 */

#include "internal.h"

// The unsuffixed, double spelling: what hybsol_matrix_t carries.
#define HYBSOL_SCALAR double
#define HYBSOL_ACC double
#define HYBSOL_VIEW hybsol_matrix_t
#define HYBSOL_FN(name) hybsol_matrix_##name
#include "matrix.inc"

// The _f32 spelling, for a system created with HYBSOL_PRECISION_SINGLE.
#define HYBSOL_SCALAR float
#define HYBSOL_ACC float
#define HYBSOL_VIEW hybsol_fmatrix_t
#define HYBSOL_FN(name) hybsol_fmatrix_##name
#include "matrix.inc"
