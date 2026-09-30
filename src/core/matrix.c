/**
 * @file core/matrix.c
 * Dense matrix kernels: products, differences and unpivoted LU.
 *
 * The kernels live in ``matrix.inc``, included once per spelling of the public
 * API; that file's header sets out the contract each inclusion configures.
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
