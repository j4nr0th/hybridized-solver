/**
 * @file src/hip/hip_kernels.cpp
 * The decomposition's kernels, AOT-compiled into the HIP backend.
 *
 * Transcription of ``src/opencl/gpu_kernels.h`` onto HIP: the same loop order,
 * the same accumulation types, no pivoting, vectors always double. Only
 * ``scalar`` differs between the two variants, the type the factors are stored
 * in -- the single one accumulates its products in ``float`` exactly as the
 * ``HYBSOL_ACC`` of the CPU's single instantiation does. The file is compiled
 * twice, once per factor precision: the double pass arrives under
 * -DHYBSOL_HIP_DOUBLE, which is what pastes the _f32/_f64 suffixes onto the
 * extern "C" entry points the host declares in ``hip_internal.h``.
 *
 * A third instantiation, under -DHYBSOL_HIP_DOUBLE -DHYBSOL_HIP_DS -- and
 * -ffp-contract=off, because implicit fusion past the named fmaf re-rounds
 * terms the TwoProd/TwoSum balances depend on -- carries
 * the double factors' arithmetic in double-single pairs (``ds_math.h``) on
 * the FP32 FMA unit, entry points suffixed _f64_ds: the same grids, the same
 * argument lists, the same double storage -- a kernel splits each block it
 * loads, chains the ds operations, and folds the result back to double for
 * the store. Any block with an element above HYBSOL_DS_MAX is a range the
 * pair cannot carry, and that work item does its block in plain double
 * instead, through the same un-#ifdef'd helpers the _f64 pass uses. Which
 * set a launch reaches is the host's choice, so the mode costs nothing here.
 *
 * The two dead helpers of the OpenCL string -- ``eliminate_step`` and
 * ``invert_diagonal_into_row``, superseded by the inline walks in the kernels
 * -- are not transcribed: no kernel calls them.
 */

#include <hip/hip_runtime.h>

#ifdef HYBSOL_HIP_DS
#include "ds_math.h"
#endif

#ifdef HYBSOL_HIP_DOUBLE
typedef double scalar;
#ifdef HYBSOL_HIP_DS
#define HYBSOL_NAME(n) hybsol_##n##_f64_ds
#else
#define HYBSOL_NAME(n) hybsol_##n##_f64
#endif
#else
typedef float scalar;
#define HYBSOL_NAME(n) hybsol_##n##_f32
#endif

/* Work-item coordinates, spelled the way a one- and two-dimensional launch gives them. */
#define HYBSOL_GID0 ((unsigned long)blockIdx.x * blockDim.x + threadIdx.x)
#define HYBSOL_GID1 ((unsigned long)blockIdx.y * blockDim.y + threadIdx.y)

/* What hybsol_operation_t calls the two operations. */
#define HYBSOL_OP_INVERT_DIAGONAL 0u
#define HYBSOL_OP_ELIMINATE 1u

/* Rows/columns of block idx, from the prefix of block sizes. */
static __device__ unsigned long block_size(const unsigned long *off, const unsigned long idx)
{
    return off[idx + 1] - off[idx];
}

/* out = a @ b, in the CPU's i-k-j order. */
static __device__ void mat_multiply(const scalar *a, const scalar *b, scalar *out, const unsigned long rows,
                                    const unsigned long inner, const unsigned long cols)
{
    for (unsigned long i = 0; i < rows; ++i)
    {
        for (unsigned long j = 0; j < cols; ++j)
            out[i * cols + j] = (scalar)0;

        for (unsigned long k = 0; k < inner; ++k)
        {
            const scalar aik = a[i * inner + k];
            for (unsigned long j = 0; j < cols; ++j)
                out[i * cols + j] = (scalar)(out[i * cols + j] + aik * b[k * cols + j]);
        }
    }
}

static __device__ void mat_subtract(scalar *a, const scalar *b, const unsigned long rows, const unsigned long cols)
{
    for (unsigned long i = 0; i < rows * cols; ++i)
        a[i] = (scalar)(a[i] - b[i]);
}

/* Doolittle LU in place, no pivoting: an exactly zero pivot fails. */
static __device__ int lu_decompose(scalar *m, const unsigned long n)
{
    for (unsigned long i = 0; i < n; ++i)
    {
        for (unsigned long j = i; j < n; ++j)
        {
            scalar u_ij = m[i * n + j];
            for (unsigned long k = 0; k < i; ++k)
                u_ij = (scalar)(u_ij - m[i * n + k] * m[k * n + j]);
            m[i * n + j] = u_ij;
        }

        const scalar pivot = m[i * n + i];
        if (pivot == (scalar)0)
            return 1;

        for (unsigned long j = i + 1; j < n; ++j)
        {
            scalar l_ij = m[j * n + i];
            for (unsigned long k = 0; k < i; ++k)
                l_ij = (scalar)(l_ij - m[k * n + i] * m[j * n + k]);
            m[j * n + i] = (scalar)(l_ij / pivot);
        }
    }
    return 0;
}

/* Solve m @ x = b in place on b, a column at a time: the unit diagonal of L, then both halves of U. */
static __device__ void lu_solve(const scalar *m, scalar *b, const unsigned long n, const unsigned long cols)
{
    /* Row scaling solves whole blocks, so the column count is part of the contract. */
    for (unsigned long c = 0; c < cols; ++c)
    {
        /* Row-major: column c is every cols-th element. */
        for (unsigned long i = 0; i < n; ++i)
        {
            scalar v = b[i * cols + c];
            for (unsigned long j = 0; j < i; ++j)
                v = (scalar)(v - m[i * n + j] * b[j * cols + c]);
            b[i * cols + c] = v;
        }

        for (unsigned long i = n; i > 0; --i)
        {
            scalar v = b[(i - 1) * cols + c];
            for (unsigned long j = i; j < n; ++j)
                v = (scalar)(v - m[(i - 1) * n + j] * b[j * cols + c]);
            b[(i - 1) * cols + c] = (scalar)(v / m[(i - 1) * n + i - 1]);
        }
    }
}

/* out -= a @ b with a in factor storage and b, out double vectors. */
static __device__ void vec_multiply_sub(const scalar *a, const double *b, double *out, const unsigned long rows,
                                        const unsigned long inner, const unsigned long cols)
{
    for (unsigned long i = 0; i < rows; ++i)
    {
        for (unsigned long j = 0; j < cols; ++j)
        {
            scalar v = (scalar)0;
            for (unsigned long k = 0; k < inner; ++k)
                v = (scalar)(v + a[i * inner + k] * (scalar)b[k * cols + j]);

            out[i * cols + j] -= (double)v;
        }
    }
}

/* Solve m @ x = b with m in factor storage, b and out double: accumulates double even for single factors. */
static __device__ void vec_lu_solve(const scalar *m, double *b, const unsigned long n)
{
    for (unsigned long i = 0; i < n; ++i)
    {
        double v = b[i];
        for (unsigned long j = 0; j < i; ++j)
            v -= (double)m[i * n + j] * b[j];
        b[i] = v;
    }

    for (unsigned long i = n; i > 0; --i)
    {
        double v = b[i - 1];
        for (unsigned long j = i; j < n; ++j)
            v -= (double)m[(i - 1) * n + j] * b[j];
        b[i - 1] = v / (double)m[(i - 1) * n + i - 1];
    }
}

#ifdef HYBSOL_HIP_DS

/* The ds kernels reuse a pass's scratch bytes for ds blocks, and store pairs into factor slots: one pair is exactly
   as wide as a stored double. */
static_assert(sizeof(ds_t) == sizeof(scalar), "a double-single pair must fill one factor slot to share its scratch");

/*
 * The range guard, run as a block is loaded: any element past HYBSOL_DS_MAX makes a float ``hi`` meaningless, so
 * that work item does the block in plain double instead. A sweep of the block is cheap next to the block walk it
 * gates, and the fallback is what keeps the mode never wrong at any magnitude.
 *
 * ``scalar`` is double in this instantiation, so a double vector segment passes through the same sweep.
 */
static __device__ int ds_fits(const scalar *m, const unsigned long count)
{
    for (unsigned long i = 0; i < count; ++i)
    {
        if (m[i] > (scalar)HYBSOL_DS_MAX || m[i] < -(scalar)HYBSOL_DS_MAX)
            return 0;
    }
    return 1;
}

/* out = a @ b, in the CPU's i-k-j order, the products and the accumulation chains ds: out is a ds block. */
static __device__ void mat_multiply_ds(const scalar *a, const scalar *b, ds_t *out, const unsigned long rows,
                                       const unsigned long inner, const unsigned long cols)
{
    const ds_t zero = {0.0f, 0.0f};
    for (unsigned long i = 0; i < rows; ++i)
    {
        for (unsigned long j = 0; j < cols; ++j)
            out[i * cols + j] = zero;

        for (unsigned long k = 0; k < inner; ++k)
        {
            const ds_t aik = ds_split(a[i * inner + k]);
            for (unsigned long j = 0; j < cols; ++j)
                out[i * cols + j] = ds_add(out[i * cols + j], ds_mul(aik, ds_split(b[k * cols + j])));
        }
    }
}

/* a -= out, a the stored-double block and out the ds product block. */
static __device__ void mat_subtract_ds(scalar *a, const ds_t *b, const unsigned long rows, const unsigned long cols)
{
    for (unsigned long i = 0; i < rows * cols; ++i)
        a[i] = ds_to_double(ds_add(ds_split(a[i]), ds_neg(b[i])));
}

/* Doolittle LU in place on the double storage, ds chains between the loads and stores; the zero pivot fails the
   same way. A stored element loses nothing: a pair's value fits double exactly. */
static __device__ int lu_decompose_ds(scalar *m, const unsigned long n)
{
    for (unsigned long i = 0; i < n; ++i)
    {
        for (unsigned long j = i; j < n; ++j)
        {
            ds_t u_ij = ds_split(m[i * n + j]);
            for (unsigned long k = 0; k < i; ++k)
                u_ij = ds_add(u_ij, ds_neg(ds_mul(ds_split(m[i * n + k]), ds_split(m[k * n + j]))));
            m[i * n + j] = ds_to_double(u_ij);
        }

        if (m[i * n + i] == (scalar)0)
            return 1;
        const ds_t pivot = ds_split(m[i * n + i]);

        for (unsigned long j = i + 1; j < n; ++j)
        {
            ds_t l_ij = ds_split(m[j * n + i]);
            for (unsigned long k = 0; k < i; ++k)
                l_ij = ds_add(l_ij, ds_neg(ds_mul(ds_split(m[k * n + i]), ds_split(m[j * n + k]))));
            m[j * n + i] = ds_to_double(ds_div(l_ij, pivot));
        }
    }
    return 0;
}

/* Solve m @ x = b in place on the stored block b, a column at a time: ds chains, and the diagonals divided with
   ds_div rather than the hardware pipe. */
static __device__ void lu_solve_ds(const scalar *m, scalar *b, const unsigned long n, const unsigned long cols)
{
    for (unsigned long c = 0; c < cols; ++c)
    {
        for (unsigned long i = 0; i < n; ++i)
        {
            ds_t v = ds_split(b[i * cols + c]);
            for (unsigned long j = 0; j < i; ++j)
                v = ds_add(v, ds_neg(ds_mul(ds_split(m[i * n + j]), ds_split(b[j * cols + c]))));
            b[i * cols + c] = ds_to_double(v);
        }

        for (unsigned long i = n; i > 0; --i)
        {
            ds_t v = ds_split(b[(i - 1) * cols + c]);
            for (unsigned long j = i; j < n; ++j)
                v = ds_add(v, ds_neg(ds_mul(ds_split(m[(i - 1) * n + j]), ds_split(b[j * cols + c]))));
            b[(i - 1) * cols + c] = ds_to_double(ds_div(v, ds_split(m[(i - 1) * n + i - 1])));
        }
    }
}

/* out -= a @ b, a the factor block and b, out double vectors: the pair replaces double as the accumulator. */
static __device__ void vec_multiply_sub_ds(const scalar *a, const double *b, double *out, const unsigned long rows,
                                           const unsigned long inner, const unsigned long cols)
{
    const ds_t zero = {0.0f, 0.0f};
    for (unsigned long i = 0; i < rows; ++i)
    {
        for (unsigned long j = 0; j < cols; ++j)
        {
            ds_t v = zero;
            for (unsigned long k = 0; k < inner; ++k)
                v = ds_add(v, ds_mul(ds_split(a[i * inner + k]), ds_split(b[k * cols + j])));

            out[i * cols + j] -= ds_to_double(v);
        }
    }
}

/* Solve m @ x = b with b the double vector: the vectors stay double end to end, the chain is ds. */
static __device__ void vec_lu_solve_ds(const scalar *m, double *b, const unsigned long n)
{
    for (unsigned long i = 0; i < n; ++i)
    {
        ds_t v = ds_split(b[i]);
        for (unsigned long j = 0; j < i; ++j)
            v = ds_add(v, ds_neg(ds_mul(ds_split(m[i * n + j]), ds_split(b[j]))));
        b[i] = ds_to_double(v);
    }

    for (unsigned long i = n; i > 0; --i)
    {
        ds_t v = ds_split(b[i - 1]);
        for (unsigned long j = i; j < n; ++j)
            v = ds_add(v, ds_neg(ds_mul(ds_split(m[(i - 1) * n + j]), ds_split(b[j]))));
        b[i - 1] = ds_to_double(ds_div(v, ds_split(m[(i - 1) * n + i - 1])));
    }
}

#endif /* HYBSOL_HIP_DS */

/* One pass's eliminations: a work item per (block row, source block) of the grid's second dimension. */
/* Occurrences that factor a diagonal instead are left to hybsol_diagonal_pass. */
/* Each source entry has its own destination, so no two work items write the same block. */
extern "C" __global__ void HYBSOL_NAME(eliminate_pass)(const unsigned long *level_rows, const unsigned long *level_k,
                                            const unsigned long *row_n_elim, const unsigned long *row_entry_offset,
                                            const unsigned long *entry_col, const unsigned long *entry_val_offset,
                                            const unsigned long *block_offsets, scalar *vals, scalar *scratch,
                                            const unsigned long pass_first, const unsigned long pass_count,
                                            const unsigned long sources_stride, const unsigned long scratch_stride)
{
    const unsigned long which = HYBSOL_GID0;
    if (which >= pass_count)
        return;
    const unsigned long j = pass_first + which;
    const unsigned long m = HYBSOL_GID1;

    const unsigned long row = level_rows[j];
    const unsigned long step = level_k[j];
    if (step >= row_n_elim[row])
        return; /* This occurrence factors the diagonal, not a source. */

    const unsigned long at_tgt = row_entry_offset[row];
    const unsigned long at_step = at_tgt + step;
    const unsigned long col_src = entry_col[at_step];
    const unsigned long at_src = row_entry_offset[col_src];
    const unsigned long b = row_n_elim[col_src] + 1 + m;
    if (b >= row_entry_offset[col_src + 1] - at_src)
        return;

    const unsigned long entry_src = at_src + b;
    const unsigned long col = entry_col[entry_src];

    /* This source entry's destination: the CPU row kernel's walk, whose results ascend with m. */
    unsigned long dest = step + 1;
    for (unsigned long done = 0; done < m; ++done)
    {
        const unsigned long previous = entry_col[at_src + row_n_elim[col_src] + 1 + done];
        while (entry_col[at_tgt + dest] < previous)
            ++dest;
        ++dest;
    }
    while (entry_col[at_tgt + dest] < col)
        ++dest;

    const unsigned long size_tgt = block_size(block_offsets, row);
    const unsigned long size_src = block_size(block_offsets, col_src);
    const unsigned long n_cols = block_size(block_offsets, col);
    scalar *const out = scratch + (which * sources_stride + m) * scratch_stride;

    /* A subtraction, never an assignment: a column gained only at this step sits at zero. */
    const scalar *const left = vals + entry_val_offset[at_step];
    const scalar *const right = vals + entry_val_offset[entry_src];
    scalar *const dest_block = vals + entry_val_offset[at_tgt + dest];
#ifdef HYBSOL_HIP_DS
    if (ds_fits(left, size_tgt * size_src) && ds_fits(right, size_src * n_cols)
        && ds_fits(dest_block, size_tgt * n_cols))
    {
        /* The scratch slot holds the ds block where the double one would sit: same bytes. */
        ds_t *const out_ds = (ds_t *)out;
        mat_multiply_ds(left, right, out_ds, size_tgt, size_src, n_cols);
        mat_subtract_ds(dest_block, out_ds, size_tgt, n_cols);
    }
    else
#endif
    {
        mat_multiply(left, right, out, size_tgt, size_src, n_cols);
        mat_subtract(dest_block, out, size_tgt, n_cols);
    }
}

/* One work item per block row whose diagonal this pass factors; the LU stays whole. */
extern "C" __global__ void HYBSOL_NAME(diagonal_pass)(const unsigned long *diag_rows, const unsigned long *row_n_elim,
                                           const unsigned long *row_entry_offset,
                                           const unsigned long *entry_val_offset, const unsigned long *block_offsets,
                                           scalar *vals, int *fail_j, const unsigned long pass_first,
                                           const unsigned long pass_count)
{
    const unsigned long which = HYBSOL_GID0;
    if (which >= pass_count)
        return;
    const unsigned long row = diag_rows[pass_first + which];

    const unsigned long at = row_entry_offset[row];
    const unsigned long i_diag = row_n_elim[row];
    const unsigned long size = block_size(block_offsets, row);
    scalar *const diag = vals + entry_val_offset[at + i_diag];
    int failed;
#ifdef HYBSOL_HIP_DS
    if (ds_fits(diag, size * size))
        failed = lu_decompose_ds(diag, size);
    else
#endif
        failed = lu_decompose(diag, size);
    if (failed != 0)
        atomicMin(fail_j, (int)(HYBSOL_GID0));
}

/* Scale every block above a diagonal by that diagonal's inverse: one work item per (block row, block). */
extern "C" __global__ void HYBSOL_NAME(scale_pass)(const unsigned long *scale_rows, const unsigned long *scale_tail,
                                        const unsigned long *row_n_elim, const unsigned long *row_entry_offset,
                                        const unsigned long *entry_col, const unsigned long *entry_val_offset,
                                        const unsigned long *block_offsets, scalar *vals,
                                        const unsigned long pass_first, const unsigned long pass_count)
{
    const unsigned long which = HYBSOL_GID0;
    if (which >= pass_count)
        return;
    const unsigned long row = scale_rows[pass_first + which];
    const unsigned long tail = scale_tail[pass_first + which];
    const unsigned long at = row_entry_offset[row];
    const unsigned long i_diag = row_n_elim[row];
    const unsigned long slot = i_diag + 1 + HYBSOL_GID1;
    if (HYBSOL_GID1 >= tail)
        return;

    const unsigned long size = block_size(block_offsets, row);
    const unsigned long size_tail = block_size(block_offsets, entry_col[at + slot]);
    scalar *const lu = vals + entry_val_offset[at + i_diag];
    scalar *const rhs = vals + entry_val_offset[at + slot];
#ifdef HYBSOL_HIP_DS
    if (ds_fits(lu, size * size) && ds_fits(rhs, size * size_tail))
        lu_solve_ds(lu, rhs, size, size_tail);
    else
#endif
        lu_solve(lu, rhs, size, size_tail);
}

/* The solve's forward substitution: the same passes, one work item per row. */
extern "C" __global__ void HYBSOL_NAME(forward_pass)(const unsigned long *level_rows, const unsigned long *level_k,
                                          const unsigned long *row_n_elim, const unsigned long *row_level,
                                          const unsigned long *row_entry_offset, const unsigned long *entry_col,
                                          const unsigned long *entry_val_offset, const unsigned long *block_offsets,
                                          const scalar *vals, double *vec, const unsigned long pass,
                                          const unsigned long from, const unsigned long pass_rows)
{
    const unsigned long j = from + HYBSOL_GID0;
    /* The grid is padded to whole work groups; from is absolute and pass_rows a count. */
    if (j >= from + pass_rows)
        return;
    const unsigned long row = level_rows[j];
    const unsigned long step = level_k[j];
    const unsigned long at = row_entry_offset[row];
    const unsigned long size_row = block_size(block_offsets, row);

    if (step < row_n_elim[row])
    {
        const unsigned long entry = at + step;
        const unsigned long col = entry_col[entry];
        const unsigned long size_col = block_size(block_offsets, col);
        const scalar *const a = vals + entry_val_offset[entry];
        const double *const in = vec + block_offsets[col];
        double *const out_segment = vec + block_offsets[row];
#ifdef HYBSOL_HIP_DS
        if (ds_fits(a, size_row * size_col) && ds_fits(in, size_col))
            vec_multiply_sub_ds(a, in, out_segment, size_row, size_col, (unsigned long)1);
        else
#endif
            vec_multiply_sub(a, in, out_segment, size_row, size_col, (unsigned long)1);
    }
    if (pass == row_level[row])
    {
        const scalar *const lu = vals + entry_val_offset[at + row_n_elim[row]];
        double *const segment = vec + block_offsets[row];
#ifdef HYBSOL_HIP_DS
        if (ds_fits(lu, size_row * size_row) && ds_fits(segment, size_row))
            vec_lu_solve_ds(lu, segment, size_row);
        else
#endif
            vec_lu_solve(lu, segment, size_row);
    }
}

/* Back substitution: serial on the CPU, and so one work item here. */
extern "C" __global__ void HYBSOL_NAME(back_substitute)(const unsigned long *row_entry_offset, const unsigned long *entry_col,
                                             const unsigned long *entry_val_offset, const unsigned long *block_offsets,
                                             const scalar *vals, double *vec, const unsigned long n)
{
    if (HYBSOL_GID0 != 0)
        return;

    for (unsigned long i = n; i > 0; --i)
    {
        const unsigned long row = i - 1;
        const unsigned long at = row_entry_offset[row];
        const unsigned long count = row_entry_offset[row + 1] - at;
        const unsigned long size_row = block_size(block_offsets, row);
        for (unsigned long t = count; t > 0; --t)
        {
            const unsigned long entry = at + t - 1;
            const unsigned long col = entry_col[entry];
            if (row >= col)
                break;

            const unsigned long size_col = block_size(block_offsets, col);
            const scalar *const a = vals + entry_val_offset[entry];
            const double *const in = vec + block_offsets[col];
            double *const out_segment = vec + block_offsets[row];
#ifdef HYBSOL_HIP_DS
            if (ds_fits(a, size_row * size_col) && ds_fits(in, size_col))
                vec_multiply_sub_ds(a, in, out_segment, size_row, size_col, (unsigned long)1);
            else
#endif
                vec_multiply_sub(a, in, out_segment, size_row, size_col, (unsigned long)1);
        }
    }
}

/* The replay of a recorded operation list: serial there, so serial here. */
extern "C" __global__ void HYBSOL_NAME(apply_ops)(const unsigned long *row_entry_offset, const unsigned long *entry_col,
                                       const unsigned long *entry_val_offset, const unsigned long *block_offsets,
                                       const scalar *vals, double *vec, const unsigned int *op_type,
                                       const unsigned long *op_row, const unsigned long *op_col,
                                       const unsigned long n_ops)
{
    if (HYBSOL_GID0 != 0)
        return;

    for (unsigned long i = 0; i < n_ops; ++i)
    {
        const unsigned long row = op_row[i];
        const unsigned long at = row_entry_offset[row];
        const unsigned long count = row_entry_offset[row + 1] - at;
        const unsigned long size_row = block_size(block_offsets, row);
        if (op_type[i] == HYBSOL_OP_ELIMINATE)
        {
            const unsigned long col = op_col[i];
            unsigned long lo = 0;
            unsigned long hi = count;
            while (lo < hi)
            {
                const unsigned long mid = lo + (hi - lo) / 2;
                if (entry_col[at + mid] < col)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            const unsigned long entry = at + lo;
            const unsigned long size_col = block_size(block_offsets, col);
            const scalar *const a = vals + entry_val_offset[entry];
            const double *const in = vec + block_offsets[col];
            double *const out_segment = vec + block_offsets[row];
#ifdef HYBSOL_HIP_DS
            if (ds_fits(a, size_row * size_col) && ds_fits(in, size_col))
                vec_multiply_sub_ds(a, in, out_segment, size_row, size_col, (unsigned long)1);
            else
#endif
                vec_multiply_sub(a, in, out_segment, size_row, size_col, (unsigned long)1);
        }
        else
        {
            unsigned long lo = 0;
            unsigned long hi = count;
            while (lo < hi)
            {
                const unsigned long mid = lo + (hi - lo) / 2;
                if (entry_col[at + mid] < row)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            const unsigned long entry = at + lo;
            const scalar *const lu = vals + entry_val_offset[entry];
            double *const segment = vec + block_offsets[row];
#ifdef HYBSOL_HIP_DS
            if (ds_fits(lu, size_row * size_row) && ds_fits(segment, size_row))
                vec_lu_solve_ds(lu, segment, size_row);
            else
#endif
                vec_lu_solve(lu, segment, size_row);
        }
    }
}
