/**
 * @file src/hip/hip_kernels.cpp
 * The decomposition's kernels, AOT-compiled into the HIP backend.
 *
 * Transcription of ``src/opencl/gpu_kernels.h`` onto HIP: the same loop order,
 * the same accumulation types, no pivoting, vectors always double. Only
 * ``scalar`` differs between the two variants, the type the factors are stored
 * in -- the single one accumulates its products in ``float`` exactly as the
 * ``HYBSOL_ACC`` of the CPU's single instantiation does. The file is compiled
 * once per factor precision, plus the two Ozaki instantiations below: the
 * double pass arrives under -DHYBSOL_HIP_DOUBLE, which is what pastes the
 * _f32/_f64 suffixes onto the extern "C" entry points the host declares in
 * ``hip_internal.h``.
 *
 * Two further instantiations, each under -DHYBSOL_HIP_DOUBLE plus
 * -DHYBSOL_HIP_OZ_I8 or -DHYBSOL_HIP_OZ2 (and -ffp-contract=off: the slice
 * windows are exact
 * integer windows of exact quotients, and a fused multiply-add the
 * compiler invents past the named fma re-rounds a term that balance
 * depends on), carry the two block-GEMM passes -- ``eliminate_pass`` and
 * ``scale_pass`` only -- on the RDNA4 matrix cores by Ozaki mantissa
 * slicing, entry points suffixed _f64_ozi8 and _f64_oz2. The first is
 * Ozaki-I, both operands sliced into the int32 tile -- the realization
 * the selection harness clocked fastest, its FP16-slice sibling having
 * lost the head to head (1.3x at 32-wide blocks) and been cut; the
 * second is Ozaki-II, asymmetric: the left operand keeps its five FP16
 * slices while the right is a single slice under its block scale, ~11
 * bits of accuracy kept and the mode meant for the inner solve under
 * iterative refinement -- the harness prints its residual, never gates
 * it. Each takes the double factors and builds the shared scale of every
 * operand region itself, as tight as what the tile walk is about to read;
 * the slices and every product between them are exact by construction, and
 * only the pruned slice-pair band is lost -- see the Ozaki section below for
 * the numerics. Every other pass,
 * and every work item whose blocks are too big for the tile walk or carry a
 * magnitude outside the sliced window, keeps the plain double arithmetic the
 * file already has; the host simply never launches these entry points on a
 * device without WMMA, and the WMMA walk itself is compiled out on a
 * non-gfx12 target so one source still builds for gfx1036.
 *
 * The two dead helpers of the OpenCL string -- ``eliminate_step`` and
 * ``invert_diagonal_into_row``, superseded by the inline walks in the kernels
 * -- are not transcribed: no kernel calls them.
 */

#include <hip/hip_runtime.h>

/* One spelling for "this object carries an Ozaki block-GEMM path": the
   argument lists, wave walks and fast paths are identical across the
   realizations, which differ only inside oz_block_gemm. */
#if defined(HYBSOL_HIP_OZ_I8) || defined(HYBSOL_HIP_OZ2)
#define HYBSOL_HIP_OZ
#endif

#ifdef HYBSOL_HIP_DOUBLE
typedef double scalar;
#if defined(HYBSOL_HIP_OZ_I8)
#define HYBSOL_NAME(n) hybsol_##n##_f64_ozi8
#elif defined(HYBSOL_HIP_OZ2)
#define HYBSOL_NAME(n) hybsol_##n##_f64_oz2
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

#ifdef HYBSOL_HIP_OZ

/* --------------------------------------------------------------------- */
/* Ozaki on the RDNA4 matrix cores (Ozaki-I int8, Ozaki-II asymmetric)     */
/* --------------------------------------------------------------------- */
/*
 * Scheme (Ozaki et al. 2012; Ootomo et al. for the FP16-slice/FP32-accumulator
 * pairing; arXiv:2409.13313 for INT8 units): every stored block carries one
 * power-of-two shared scale, a strict ceiling of the block's largest element,
 * so every element x has |x/scale| < 1 exactly (a zero block gets scale 1 and
 * every slice comes out zero). Each element then deconstructs into k slices of
 * tau bits: slice i the integer nearest the residue the slices before it left,
 * x/scale = sum_i s_i 2^{-tau i} to within 2^{-tau k}. Each window step is
 * exact in double -- m = x/scale is a power-of-two scaling, |residue| bounds
 * every later window below 2^{tau-1}, and subtract-then-shift keeps both
 * steps unrounded (all windows are integers far inside the 53-bit significand);
 * -ffp-contract=off on this object (C1's lesson: implicit
 * fusion re-rounds a term an exact transform's balance depends on) makes sure
 * no *other* contraction joins in. Nothing the matrix cores touch ever
 * rounds: the slice scales at least double the strict power-of-two ceiling
 * of their block, so |x/scale| <= 1/2 keeps every window at or under
 * 2^(tau-1) -- |s| <= 1024 in _Float16's 11-bit significand, |s| <= 64 for
 * signed int8 -- integers the input type carries exactly, and the
 * combine folds each tile's i32/f32 accumulator out in double with the
 * power-of-two weight 2^{-tau(i+j)}, scaling back by scale_a * scale_b.
 * Ozaki-I loses only the pruned band: the dropped slice pairs' total
 * contribution sits below 2^{-50} of a block's magnitude, the deepest
 * kept pair at 2^{-tau*10} -- 43 of 64 products, the Ozaki-I pruning at
 * target band t ~ 36 the digest's cost table counts. (An FP16-slice
 * realization of the same scheme, tau = 11 and 15 of 25 products kept,
 * was compiled for the selection harness's head to head and cut when it
 * lost: 1.3x slower at 32-wide blocks.)
 *
 * Accumulation exactness (why the FP16-window walks split their k
 * dimension): the i8 path's int32 accumulator carries 16 * 64^2 = 2^16
 * per tile and pair exactly, whatever the blocks' scales. A float
 * accumulator is exact for integers only up to 2^24: leading windows
 * reach 2^10, so a product reaches 2^20 and a full 16-term sum brushes
 * that ceiling. The walk therefore runs each pair as groups of at most
 * four products -- quartets for the leading pair, halves whenever
 * exactly one operand is leading, singles otherwise -- each group a
 * separate WMMA call whose partial sums stay at or under 4 * 2^20 <
 * 2^22, and the double combine adds the groups. No f32 rounding ever
 * happens; the only error in the whole path is the pruned band.
 *
 * Ozaki-II (the _f64_oz2 object) keeps this whole machinery and thins one
 * side: the left operand is still sliced k = 5, but the right is a single
 * window -- its top 11 bits under the same halved-ceiling scale -- so a
 * product pair is (a_u, b_0) with weight 2^{-tau(u+1)}, five pairs instead
 * of fifteen, and the only error is the right operand's dropped residue
 * (~2^-12 relative). That is refinement-partner accuracy, not a solve's;
 * the harness prints it and never gates it. The same group-split k walk
 * keeps every f32 partial an exact integer: the single b window is at most
 * 2^10, so a group sum still sits under 2^22.
 *
 * Fragments: one wavefront (32 lanes) carries one 16x16 output tile, the
 * layout the Phase D probe pinned down numerically at gfx1201 -- lane L holds
 * A's row L%16 at k = k0 + 8(L/16) + e and B's column L%16 at the same k
 * (e = 0..7, the guide's reference-kernel convention: the hardware pairs A's
 * slot e with B's slot e of the same lane, so only the two loads' consistency
 * matters -- the i8 byte order inside the two registers passed either way for
 * the same reason), and register g of the accumulator is row (L/16)*8 + g,
 * column L%16. The host launches these entry points 1-D with workgroups of
 * 64 lanes -- exactly two wavefronts, one work item each -- and the kernel
 * recovers the (which, m) pair the plain kernels walk from the wave index.
 */

/* Slice constants and the per-block magnitude guard, shared by the two
   Ozaki instantiations. */

/**
 * The magnitudes an Ozaki work item agrees to slice, as a window on a block's
 * largest element.
 *
 * The shared block scale is a power of two from the block's own maximum, and
 * the extraction below shifts a slice window by up to the full slice span; a
 * scale beyond ~2^332 puts the tile's result past what the double combine can
 * hold once the two operands' scales multiply, so an entry block reaching this
 * magnitude leaves the work item to the plain double path. Powers of two up to
 * 2^333 carry every window exactly. The floor is the same ceiling mirrored:
 * at 2^-332 the product of the two scales underflows to zero, and a block
 * sliced at that scale contributes nothing at all -- a wrong answer with
 * nothing wrong in it. An all-zero block slices at zero whatever its scale, so
 * it fits.
 */
#define HYBSOL_OZ_MAX_ENTRY 1e100
#define HYBSOL_OZ_MIN_ENTRY 1e-100

/**
 * Slice parameters (Ozaki et al. 2012; Ootomo et al. for the
 * FP16-slice/FP32-accumulator pairing; arXiv:2409.13313 for INT8 units):
 * tau mantissa bits per slice, k = ceil(53 / tau) slices to cover a double.
 * The int8 pair drives the Ozaki-I slices; the FP16 pair drives the windows
 * Ozaki-II slices one operand into.
 */
#define HYBSOL_OZ_F16_TAU 11
#define HYBSOL_OZ_F16_SLICES 5
#define HYBSOL_OZ_I8_TAU 7
#define HYBSOL_OZ_I8_SLICES 8

/** Whether a block sits in the magnitude window the slicing carries. */
static __device__ inline int oz_block_fits(const double *const m, const unsigned long count)
{
    double maxabs = 0.0;
    for (unsigned long i = 0; i < count; ++i)
    {
        const double v = m[i] < 0.0 ? -m[i] : m[i];
        if (v > maxabs)
            maxabs = v;
    }
    if (maxabs == 0.0)
        return 1;
    return maxabs <= HYBSOL_OZ_MAX_ENTRY && maxabs >= HYBSOL_OZ_MIN_ENTRY;
}

#if defined(__gfx1200__) || defined(__gfx1201__)

using oz_v8half = _Float16 [[clang::ext_vector_type(8)]];
using oz_v8float = float [[clang::ext_vector_type(8)]];
using oz_v2int = int [[clang::ext_vector_type(2)]];
using oz_v8int = int [[clang::ext_vector_type(8)]];

/*
 * Kept slice pairs for the Ozaki-I walk, zero-based (u = slice index - 1):
 * the int8 realization keeps u+v <= 8 (43 of 64). Ozaki-II needs no table:
 * its pairs are simply (u, 0).
 */
static __device__ __constant__ const unsigned char oz_i8_pairs[43][2] = {
    {0, 0}, {0, 1}, {1, 0}, {0, 2}, {1, 1}, {2, 0}, {0, 3}, {1, 2}, {2, 1}, {3, 0}, {0, 4}, {1, 3}, {2, 2}, {3, 1},
    {4, 0}, {0, 5}, {1, 4}, {2, 3}, {3, 2}, {4, 1}, {5, 0}, {0, 6}, {1, 5}, {2, 4}, {3, 3}, {4, 2}, {5, 1}, {6, 0},
    {0, 7}, {1, 6}, {2, 5}, {3, 4}, {4, 3}, {5, 2}, {6, 1}, {7, 0}, {1, 7}, {2, 6}, {3, 5}, {4, 4}, {5, 3}, {6, 2},
    {7, 1}
};

/* Pair weight 2^{-tau(u+v+2)}: exact powers of two, spelled out. */
static __device__ __constant__ const double oz_f16_weights[5] = {2.384185791015625e-07, 1.1641532182693481e-10,
                                                                 5.684341886080802e-14, 2.7755575615628914e-17,
                                                                 1.3552527156068805e-20};
static __device__ __constant__ const double oz_i8_weights[9] = {
    6.103515625e-05,        4.76837158203125e-07,      3.725290298461914e-09,    2.9103830456733704e-11,
    2.2737367544323206e-13, 1.7763568394002505e-15,    1.3877787807814457e-17,   1.0842021724855044e-19,
    8.470329472543003e-22};

/*
 * The scale a block is actually sliced at: two times the strict power-of-two
 * ceiling of the largest element the walk is about to read (frexp answers it
 * exactly), floored at 2 so an all-zero region still divides. The extra factor
 * two is what makes rounding safe: |x/scale| <= 1/2 lands every leading
 * window at or under 2^(tau-1) -- |s_1| <= 1024 in _Float16 and |s_1| <= 64
 * in signed int8 -- where a nearest-integer step can never reach the +128 a
 * signed byte cannot spell. Slicing still spans 50 (f16) or 56 (i8) bits of
 * the significand, far past the pruned band.
 *
 * The ceiling is the region's own, taken as it is read: eliminations rewrite
 * blocks in both directions, and a factor block divided by its pivot drops
 * from its block's scale to multipliers of order one. A scale looser than the
 * region by 2^k costs k bits of every slice, which is how a shared exponent
 * picked once -- at create, from the initial values -- turns a
 * 1e12-scaled system into a 1e-4 answer: the block's whole mantissa sits below
 * the window span. Nothing here inherits from the host; the operand in front
 * of the tile is the only thing a scale may be built from.
 */
static __device__ double oz_scale_of(const double *const m, const unsigned long count)
{
    double maxabs = 0.0;
    for (unsigned long i = 0; i < count; ++i)
    {
        const double v = m[i] < 0.0 ? -m[i] : m[i];
        if (v > maxabs)
            maxabs = v;
    }
    if (maxabs == 0.0)
        return 2.0;
    int e = 0;
    (void)frexp(maxabs, &e); /* 2^(e-1) <= maxabs < 2^e */
    return ldexp(1.0, e + 1);
}

/* The same ceiling over a strided region: the first ``cols`` entries of each of
   the first ``rows`` rows, ``stride`` entries apart -- what a tile walk reads
   out of a stored block whose row is wider than the tile. */
static __device__ double oz_scale_of_region(const double *const m, const unsigned rows, const unsigned cols,
                                            const unsigned stride)
{
    double maxabs = 0.0;
    for (unsigned r = 0; r < rows; ++r)
    {
        const double *const line = m + (size_t)r * stride;
        for (unsigned c = 0; c < cols; ++c)
        {
            const double v = line[c] < 0.0 ? -line[c] : line[c];
            if (v > maxabs)
                maxabs = v;
        }
    }
    if (maxabs == 0.0)
        return 2.0;
    int e = 0;
    (void)frexp(maxabs, &e);
    return ldexp(1.0, e + 1);
}

/* The k slice windows of one element m = x/scale, zero-based index = slice-1. */
static __device__ inline void oz_window_f16(const double m, _Float16 *const out)
{
    double q = m * 2048.0; /* 2^tau */
    #pragma unroll
    for (int i = 0; i < HYBSOL_OZ_F16_SLICES; ++i)
    {
        out[i] = (_Float16)nearbyint(q); /* every window at or under 2^(tau-1): exact in f16 */
        if (i + 1 < HYBSOL_OZ_F16_SLICES)
            q = (q - (double)out[i]) * 2048.0; /* the residue, shifted: both steps exact */
    }
}

static __device__ inline void oz_window_i8(const double m, signed char *const out)
{
    double q = m * 128.0; /* 2^tau */
    #pragma unroll
    for (int i = 0; i < HYBSOL_OZ_I8_SLICES; ++i)
    {
        out[i] = (signed char)nearbyint(q); /* the halved window keeps every slice at or under 64 */
        if (i + 1 < HYBSOL_OZ_I8_SLICES)
            q = (q - (double)out[i]) * 128.0; /* the residue, shifted: both steps exact */
    }
}

/*
 * out = a @ b: a is rows_a x inner stored with row stride a_stride, b is
 * inner x cols_b dense but only its first rows_b (<= inner) rows are stored
 * -- rows past that read as zero, which keeps edge panels from touching
 * memory past their block and is exact whenever the caller's operand needs
 * only those rows. rows_a, inner and cols_b all <= 32, and one wavefront
 * tiles the output in ceil(/16)^2 WMMA tiles. The scales are powers of two
 * that dominate the two blocks' largest entries by at least a factor two
 * (never zero). The Ozaki-II walk splits each pair's k dimension into
 * groups of at most four products so every float partial sum stays an
 * exact integer below 2^24 -- see the accumulation note at the head of
 * this section; the Ozaki-I int8 walk needs no split, its int32
 * accumulator is exact for every 16-term sum.
 */

/* The int8 walk: slices at or under 127 in absolute value, every product and
   every 16-term partial sum carried exactly by the int32 accumulator (the
   signed-operand form, unclamped), so one WMMA call per pair and k tile. */
static __device__ void oz_block_gemm_i8(const double *a, const double scale_a, const unsigned rows_a,
                                        const unsigned a_stride, const unsigned inner, const double *b,
                                        const double scale_b, const unsigned rows_b, const unsigned cols_b,
                                        double *out)
{
    const unsigned lane = threadIdx.x & 31u;
    const unsigned kbase_lane = 8u * (lane >> 4);
    const double inv_a = 1.0 / scale_a, inv_b = 1.0 / scale_b;
    for (unsigned ti = 0; ti * 16u < rows_a; ++ti)
    {
        for (unsigned tj = 0; tj * 16u < cols_b; ++tj)
        {
            double acc[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            for (unsigned tk = 0; tk * 16u < inner; ++tk)
            {
                const unsigned row = ti * 16u + (lane & 15u);
                const unsigned col = tj * 16u + (lane & 15u);
                signed char asl[8][HYBSOL_OZ_I8_SLICES];
                signed char bsl[8][HYBSOL_OZ_I8_SLICES];
                #pragma unroll
                for (unsigned e = 0; e < 8u; ++e)
                {
                    const unsigned kk = tk * 16u + kbase_lane + e;
                    if (row < rows_a && kk < inner)
                        oz_window_i8(a[row * a_stride + kk] * inv_a, &asl[e][0]);
                    else
                    {
                        #pragma unroll
                        for (int s = 0; s < HYBSOL_OZ_I8_SLICES; ++s)
                            asl[e][s] = (signed char)0;
                    }
                    if (col < cols_b && kk < inner && kk < rows_b)
                        oz_window_i8(b[kk * cols_b + col] * inv_b, &bsl[e][0]);
                    else
                    {
                        #pragma unroll
                        for (int s = 0; s < HYBSOL_OZ_I8_SLICES; ++s)
                            bsl[e][s] = (signed char)0;
                    }
                }
                #pragma unroll
                for (unsigned p = 0; p < 43u; ++p)
                {
                    const unsigned u = oz_i8_pairs[p][0], v = oz_i8_pairs[p][1];
                    oz_v2int frag_a = {}, frag_b = {};
                    unsigned long long ra = 0, rb = 0;
                    #pragma unroll
                    for (unsigned e = 0; e < 8u; ++e)
                    {
                        ra |= ((unsigned long long)(unsigned char)asl[e][u]) << (8u * e);
                        rb |= ((unsigned long long)(unsigned char)bsl[e][v]) << (8u * e);
                    }
                    frag_a[0] = (int)(ra & 0xffffffffu);
                    frag_a[1] = (int)(ra >> 32);
                    frag_b[0] = (int)(rb & 0xffffffffu);
                    frag_b[1] = (int)(rb >> 32);
                    oz_v8int c = {};
                    c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, frag_a, true, frag_b, c, false);
                    const double w = oz_i8_weights[u + v];
                    #pragma unroll
                    for (unsigned g = 0; g < 8u; ++g)
                        acc[g] += (double)c[g] * w;
                }
            }
            const double weight = scale_a * scale_b;
            const unsigned crow = ti * 16u + (lane >> 4) * 8u;
            const unsigned ccol = tj * 16u + (lane & 15u);
            #pragma unroll
            for (unsigned g = 0; g < 8u; ++g)
            {
                if (crow + g < rows_a && ccol < cols_b)
                    out[(crow + g) * cols_b + ccol] = acc[g] * weight;
            }
        }
    }
}

/*
 * The Ozaki-II walk: the left operand keeps its five FP16 slices, the
 * right is its single leading window -- top 11 bits under the same
 * halved power-of-two ceiling, exact in FP16 as an integer at or under
 * 2^10 -- so five pairs (u, 0) carry the whole product and the right
 * operand's residue is the mode's designed loss. Grouping is the f16
 * walk's: quartets when both windows lead, halves otherwise, every f32
 * partial an exact integer under 2^22.
 */
static __device__ void oz_block_gemm_oz2(const double *a, const double scale_a, const unsigned rows_a,
                                         const unsigned a_stride, const unsigned inner, const double *b,
                                         const double scale_b, const unsigned rows_b, const unsigned cols_b,
                                         double *out)
{
    const unsigned lane = threadIdx.x & 31u;
    const unsigned kbase_lane = 8u * (lane >> 4); /* the lane group's first k inside a k tile */
    const double inv_a = 1.0 / scale_a, inv_b = 1.0 / scale_b; /* powers of two: exact */
    for (unsigned ti = 0; ti * 16u < rows_a; ++ti)
    {
        for (unsigned tj = 0; tj * 16u < cols_b; ++tj)
        {
            double acc[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            for (unsigned tk = 0; tk * 16u < inner; ++tk)
            {
                const unsigned row = ti * 16u + (lane & 15u);
                const unsigned col = tj * 16u + (lane & 15u);
                _Float16 asl[8][HYBSOL_OZ_F16_SLICES];
                _Float16 bwin[8];
                #pragma unroll
                for (unsigned e = 0; e < 8u; ++e)
                {
                    const unsigned kk = tk * 16u + kbase_lane + e;
                    if (row < rows_a && kk < inner)
                        oz_window_f16(a[row * a_stride + kk] * inv_a, &asl[e][0]);
                    else
                    {
                        #pragma unroll
                        for (int s = 0; s < HYBSOL_OZ_F16_SLICES; ++s)
                            asl[e][s] = (_Float16)0;
                    }
                    if (col < cols_b && kk < inner && kk < rows_b)
                        bwin[e] = (_Float16)nearbyint(b[kk * cols_b + col] * inv_b * 2048.0); /* the one window: exact */
                    else
                        bwin[e] = (_Float16)0;
                }
                #pragma unroll
                for (unsigned u = 0; u < HYBSOL_OZ_F16_SLICES; ++u)
                {
                    const unsigned groups = (u == 0u) ? 4u : 2u;
                    #pragma unroll
                    for (unsigned gq = 0; gq < 4u; ++gq)
                    {
                        if (gq >= groups)
                            break;
                        const unsigned span = 16u / groups;
                        const unsigned klo = span * gq;
                        oz_v8half frag_a = {}, frag_b = {};
                        #pragma unroll
                        for (unsigned e = 0; e < 8u; ++e)
                        {
                            if (kbase_lane + e < klo || kbase_lane + e >= klo + span)
                                continue;
                            frag_a[e] = asl[e][u];
                            frag_b[e] = bwin[e];
                        }
                        oz_v8float c = {};
                        c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(frag_a, frag_b, c);
                        const double w = oz_f16_weights[u]; /* 2^{-tau(u+1)}: the b window weighs one slice */
                        #pragma unroll
                        for (unsigned g = 0; g < 8u; ++g)
                            acc[g] += (double)c[g] * w;
                    }
                }
            }
            const double weight = scale_a * scale_b;
            const unsigned crow = ti * 16u + (lane >> 4) * 8u;
            const unsigned ccol = tj * 16u + (lane & 15u);
            #pragma unroll
            for (unsigned g = 0; g < 8u; ++g)
            {
                if (crow + g < rows_a && ccol < cols_b)
                    out[(crow + g) * cols_b + ccol] = acc[g] * weight;
            }
        }
    }
}

/* The dispatch the kernels call: whichever realization this object compiled. */
static __device__ void oz_block_gemm(const double *a, const double scale_a, const unsigned rows_a,
                                     const unsigned a_stride, const unsigned inner, const double *b,
                                     const double scale_b, const unsigned rows_b, const unsigned cols_b,
                                     double *out)
{
#ifdef HYBSOL_HIP_OZ2
    oz_block_gemm_oz2(a, scale_a, rows_a, a_stride, inner, b, scale_b, rows_b, cols_b, out);
#else
    oz_block_gemm_i8(a, scale_a, rows_a, a_stride, inner, b, scale_b, rows_b, cols_b, out);
#endif
}

/*
 * scale_pass on the matrix cores. Inverting each unit-triangular LU panel is
 * a finite Neumann series: with W = I - Lpanel, Lpanel = I - W is unit lower
 * triangular, so W is strictly lower triangular and W^16 = 0 identically at
 * any magnitude -- hence sum_{m<16} W^m is Lpanel^-1 exactly in exact
 * arithmetic. The series runs as its doubling product form, four P <- P + P
 * @ W^steps and three W <- W @ W squarings through the same tile walk; P and
 * W are rescanned for fresh strict-ceiling scales after every step, so their
 * windows stay legal however far the intermediates grow. Rows and
 * columns past the block's own edge carry the identity through W and P, and
 * only rows inside the edge are ever applied to a right-hand side. The two
 * GEMM walks around the inversion -- folding the earlier panels' eliminations
 * into this panel's right-hand side, and applying the inverse -- read the
 * stored blocks with their own rescanned scales (eliminations grow entries
 * past the operands' own magnitudes). Accumulation between GEMMs
 * stays in double, matching the plain path's scratch contract.
 *
 * scratch layout, one stride per work item: G (16 x size_tail), P (16x16),
 * W (16x16) -- 16*max_block + 512 doubles, the stride the host sized.
 */
static __device__ void oz_scale_run(scalar *lu, scalar *rhs, const unsigned size, const unsigned size_tail,
                                    double *scratch)
{
    double *const g = scratch;
    /* G holds every tile walk's output -- 16 rows by size_tail columns for
     * the block walks, 16 by 16 for the Neumann steps -- so it starts at a
     * region of max(16*size_tail, 256) doubles; P and W follow, one stride
     * per work item, the host having sized the stride as 16*max_block + 512
     * whenever Ozaki is live. */
    double *const p = scratch + (size_tail > 15u ? 16u * size_tail : 256u);
    double *const wmat = p + 16u * 16u;
    const unsigned panels = (size + 15u) / 16u;
    for (unsigned pp = 0; pp < panels; ++pp)
    {
        const unsigned prow = pp * 16u;
        const unsigned prows = (size - prow < 16u) ? size - prow : 16u;

        for (unsigned cp = 0; cp < pp; ++cp)
        {
            const double scale_lu = oz_scale_of_region(lu + prow * size + cp * 16u, prows, 16u, size);
            const double scale_cp = oz_scale_of(rhs + cp * 16u * size_tail, 16u * size_tail);
            oz_block_gemm(lu + prow * size + cp * 16u, scale_lu, prows, size, 16u, rhs + cp * 16u * size_tail,
                          scale_cp, 16u, size_tail, g);
            for (unsigned r = 0; r < prows; ++r)
                for (unsigned c = 0; c < size_tail; ++c)
                    rhs[(prow + r) * size_tail + c] -= g[r * size_tail + c];
        }

        /* W = I - L: zero diagonal, minus the strictly-lower (r > c) LU
         * entries -- nothing from U's upper half, or W would not be strictly
         * lower triangular and the series would diverge. It is sliced at its
         * own ceiling below, not the LU block's: the multipliers it is built
         * from are quotients, and their scale has nothing to do with the
         * block's; the series sums raw W^m, so P = L^-1. */
        #pragma unroll
        for (unsigned r = 0; r < 16u; ++r)
            for (unsigned c = 0; c < 16u; ++c)
                wmat[r * 16u + c] = (r > c && r < prows && prow + c < size)
                                        ? -lu[(prow + r) * size + prow + c]
                                        : 0.0;
        #pragma unroll
        for (unsigned r = 0; r < 16u; ++r)
            for (unsigned c = 0; c < 16u; ++c)
                p[r * 16u + c] = (r == c ? 1.0 : 0.0);

        /* P accumulates the product form of the finite geometric series:
         * (I + W)(I + W^2)(I + W^4)(I + W^8) = sum_{m<16} W^m exactly for a
         * nilpotent W -- four accumulate steps and three squarings, each a
         * tile walk through the G region, instead of fifteen steps of a
         * chain that would only be a truncated series step by step. P and W
         * are rescanned for strict power-of-two scales between steps, so
         * the windows stay legal however large the intermediate powers. */
        double scale_p = oz_scale_of(p, 256u); /* |I| <= 1 < 2 */
        double scale_w = oz_scale_of(wmat, 256u);
        for (unsigned step = 0; step < 4u; ++step)
        {
            oz_block_gemm(p, scale_p, 16u, 16u, 16u, wmat, scale_w, 16u, 16u, g);
            for (unsigned i = 0; i < 256u; ++i)
                p[i] += g[i];
            if (step + 1u < 4u)
            {
                oz_block_gemm(wmat, scale_w, 16u, 16u, 16u, wmat, scale_w, 16u, 16u, g);
                for (unsigned i = 0; i < 256u; ++i)
                    wmat[i] = g[i];
                scale_w = oz_scale_of(wmat, 256u);
            }
            scale_p = oz_scale_of(p, 256u);
        }

        /* What is left is the block-diagonal solve: this panel's right-hand
         * side, folded clean by the loop above, gets replaced by P @ it --
         * replaced, not added to: L_pp^-1 acts on the panel as a whole. The
         * rows_b bound keeps the edge panel's rows past the block's end out
         * of both the reads and the product (P is lower triangular, so those
         * rows only ever feed output rows outside the edge, which are
         * dropped). */
        const double scale_pp = oz_scale_of(rhs + prow * size_tail, prows * size_tail);
        oz_block_gemm(p, scale_p, 16u, 16u, 16u, rhs + prow * size_tail, scale_pp, prows, size_tail, g);
        for (unsigned r = 0; r < prows; ++r)
            for (unsigned c = 0; c < size_tail; ++c)
                rhs[(prow + r) * size_tail + c] = g[r * size_tail + c];
    }

    /* The Neumann series finished each panel's L solve; the U half is a
     * back-substitution over the panel structure, walked the other way: by
     * the time panel pp is reached, every panel below it is final, so one
     * dense tile walk per (pp, cp) subtracts U_pp,cp @ x_cp, and a 16x16
     * triangular finish -- cheap next to the walks, and the divisions are
     * the same scalar ones the diagonal pass itself does in double. */
    for (unsigned pp = panels; pp > 0; --pp)
    {
        const unsigned prow = (pp - 1u) * 16u;
        const unsigned prows = (size - prow < 16u) ? size - prow : 16u;

        for (unsigned cp = pp; cp < panels; ++cp)
        {
            const unsigned prows_cp = (size - cp * 16u < 16u) ? size - cp * 16u : 16u;
            /* The tile is 16 wide but the block's last column panel is not, and past its
               edge the walk reads neighbouring rows of the same block -- contributions it
               masks off with rows_b, but a scale scan must not swallow them: a LU block
               holds multipliers of order one next to U entries at the block's own scale,
               and a ceiling taken from the wrong half slices the right half to zero. */
            const double scale_lu = oz_scale_of_region(lu + prow * size + cp * 16u, prows, prows_cp, size);
            const double scale_cp = oz_scale_of(rhs + cp * 16u * size_tail, prows_cp * size_tail);
            oz_block_gemm(lu + prow * size + cp * 16u, scale_lu, prows, size, 16u, rhs + cp * 16u * size_tail,
                          scale_cp, prows_cp, size_tail, g);
            for (unsigned r = 0; r < prows; ++r)
                for (unsigned c = 0; c < size_tail; ++c)
                    rhs[(prow + r) * size_tail + c] -= g[r * size_tail + c];
        }

        for (unsigned c = 0; c < size_tail; ++c)
            for (unsigned r = prows; r > 0; --r)
            {
                double acc = rhs[(prow + r - 1) * size_tail + c];
                for (unsigned j = r; j < prows; ++j)
                    acc -= lu[(prow + r - 1) * size + prow + j] * rhs[(prow + j) * size_tail + c];
                rhs[(prow + r - 1) * size_tail + c] = acc / lu[(prow + r - 1) * size + prow + r - 1];
            }
    }
}

#endif /* __gfx1200__ || __gfx1201__ */

#endif /* HYBSOL_HIP_OZ */

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
#ifdef HYBSOL_HIP_OZ
    /* The Ozaki builds run one whole wavefront per work item: the host
       launches 1-D with workgroups of 64 lanes, and the wave's index carries
       the (which, m) pair the plain kernels walk with two grid dimensions.
       The pair is wave-uniform by construction, so every lane agrees.
       Load-bearing alongside it: the scratch slot and the destination block
       are indexed by that pair alone, so this wave owns them outright --
       changing the group size or the scratch indexing without the ownership
       with it would put two waves in one slot. */
    const unsigned long wave = HYBSOL_GID0 >> 5;
    const unsigned long which = wave / sources_stride;
    const unsigned long m = wave % sources_stride;
    if (which >= pass_count)
        return;
    const unsigned long j = pass_first + which;
#else
    const unsigned long which = HYBSOL_GID0;
    if (which >= pass_count)
        return;
    const unsigned long j = pass_first + which;
    const unsigned long m = HYBSOL_GID1;
#endif

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
#ifdef HYBSOL_HIP_OZ
#if defined(__gfx1200__) || defined(__gfx1201__)
    /* The matrix cores when the whole pair fits the tile walk and every entry
       sits in the range the slicing carries; the host only reaches this entry
       point on a WMMA device with blocks that big, but the per-pair bounds
       here are what keep a mixed system — one small pair among large ones —
       correct. Anything else falls through to the plain double walk. */
    if (size_tgt <= 32u && size_src <= 32u && n_cols <= 32u && oz_block_fits(left, size_tgt * size_src)
        && oz_block_fits(right, size_src * n_cols))
    {
        oz_block_gemm(left, oz_scale_of(left, size_tgt * size_src), size_tgt, size_src, size_src, right,
                      oz_scale_of(right, size_src * n_cols), size_src, n_cols, out);
        mat_subtract(dest_block, out, size_tgt, n_cols);
        return;
    }
#endif
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
    failed = lu_decompose(diag, size);
    if (failed != 0)
        atomicMin(fail_j, (int)(pass_first + HYBSOL_GID0));
}

/* Scale every block above a diagonal by that diagonal's inverse: one work item per (block row, block). */
extern "C" __global__ void HYBSOL_NAME(scale_pass)(const unsigned long *scale_rows, const unsigned long *scale_tail,
                                        const unsigned long *row_n_elim, const unsigned long *row_entry_offset,
                                        const unsigned long *entry_col, const unsigned long *entry_val_offset,
                                        const unsigned long *block_offsets, scalar *vals,
                                        const unsigned long pass_first, const unsigned long pass_count
#ifdef HYBSOL_HIP_OZ
                                            ,
                                            scalar *scratch, const unsigned long tail_stride,
                                            const unsigned long scratch_stride
#endif
                                            )
{
#ifdef HYBSOL_HIP_OZ
    /* One wavefront per work item, the wave's index carrying the (row, tail
       slot) pair -- see the note at the eliminate entry's head. The scratch
       slot below (wave * scratch_stride) and the rhs block it scales belong
       to this wave alone; the same ownership caveat applies. */
    const unsigned long wave = HYBSOL_GID0 >> 5;
    const unsigned long which = wave / tail_stride;
    const unsigned long gid1 = wave % tail_stride;
    if (which >= pass_count)
        return;
#else
    const unsigned long which = HYBSOL_GID0;
    if (which >= pass_count)
        return;
    const unsigned long gid1 = HYBSOL_GID1;
#endif
    const unsigned long row = scale_rows[pass_first + which];
    const unsigned long tail = scale_tail[pass_first + which];
    const unsigned long at = row_entry_offset[row];
    const unsigned long i_diag = row_n_elim[row];
    const unsigned long slot = i_diag + 1 + gid1;
    if (gid1 >= tail)
        return;

    const unsigned long size = block_size(block_offsets, row);
    const unsigned long size_tail = block_size(block_offsets, entry_col[at + slot]);
    scalar *const lu = vals + entry_val_offset[at + i_diag];
    scalar *const rhs = vals + entry_val_offset[at + slot];
#ifdef HYBSOL_HIP_OZ
#if defined(__gfx1200__) || defined(__gfx1201__)
    /* The matrix-core solve when the pair fits the tile walk and every entry
       sits in the range the slicing carries; the Neumann inversion of the
       panel replaces the plain forward solve one-for-one, the same storage,
       the same zero-pivot failure already reported by the diagonal pass. */
    if (size <= 32u && size_tail <= 32u && oz_block_fits(lu, size * size) && oz_block_fits(rhs, size * size_tail))
    {
        oz_scale_run(lu, rhs, size, size_tail, scratch + wave * scratch_stride);
        return;
    }
#endif
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
        vec_multiply_sub(a, in, out_segment, size_row, size_col, (unsigned long)1);
    }
    if (pass == row_level[row])
    {
        const scalar *const lu = vals + entry_val_offset[at + row_n_elim[row]];
        double *const segment = vec + block_offsets[row];
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
            vec_lu_solve(lu, segment, size_row);
        }
    }
}
