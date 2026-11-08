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
 * The two dead helpers of the OpenCL string -- ``eliminate_step`` and
 * ``invert_diagonal_into_row``, superseded by the inline walks in the kernels
 * -- are not transcribed: no kernel calls them.
 */

#include <hip/hip_runtime.h>

#ifdef HYBSOL_HIP_DOUBLE
typedef double scalar;
#define HYBSOL_NAME(n) hybsol_##n##_f64
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
    mat_multiply(vals + entry_val_offset[at_step], vals + entry_val_offset[entry_src], out, size_tgt, size_src,
                 n_cols);
    mat_subtract(vals + entry_val_offset[at_tgt + dest], out, size_tgt, n_cols);
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
    if (lu_decompose(vals + entry_val_offset[at + i_diag], size) != 0)
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
    lu_solve(vals + entry_val_offset[at + i_diag], vals + entry_val_offset[at + slot], size,
             block_size(block_offsets, entry_col[at + slot]));
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
        vec_multiply_sub(vals + entry_val_offset[entry], vec + block_offsets[col], vec + block_offsets[row], size_row,
                         block_size(block_offsets, col), (unsigned long)1);
    }
    if (pass == row_level[row])
    {
        vec_lu_solve(vals + entry_val_offset[at + row_n_elim[row]], vec + block_offsets[row], size_row);
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
            vec_multiply_sub(vals + entry_val_offset[entry], vec + block_offsets[col], vec + block_offsets[row],
                             size_row, size_col, (unsigned long)1);
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
            vec_multiply_sub(vals + entry_val_offset[entry], vec + block_offsets[col], vec + block_offsets[row],
                             size_row, block_size(block_offsets, col), (unsigned long)1);
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
            vec_lu_solve(vals + entry_val_offset[entry], vec + block_offsets[row], size_row);
        }
    }
}
