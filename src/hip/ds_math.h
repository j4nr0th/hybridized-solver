/**
 * @file src/hip/ds_math.h
 * Double-single arithmetic for the HIP device code.
 *
 * A 64-bit factor value is carried as a pair of 32-bit floats, a leading
 * ``hi`` and a trailing ``lo``, and every operation runs on the FP32 FMA
 * unit -- the point being that RDNA4's FP64 pipe is a small fraction of the
 * FP32 pipe's rate, so a value split in two and manipulated through exact
 * error-free transforms beats the native pipe by far more than the lost
 * bits cost. The scheme is the classical double-single arithmetic of
 * Squassabia et al.: TwoSum keeps an addition's rounding error, TwoProd --
 * via the fused multiply-add, exact for the leading product -- keeps a
 * multiplication's, and chains of them carry ~14 significant decimal
 * digits, enough for the backend's 10-11-digit target.
 *
 * Device-only: like the kernels that include it, this is compiled by the
 * HIP compiler, not the host C one -- and only correct without implicit
 * multiply-add fusion: every transform here balances a rounded product
 * against its captured error, so the object that compiles these kernels is
 * built -ffp-contract=off (see ``CMakeLists.txt``); ``fmaf`` itself, which
 * the source asks for by name, is exempt and stays. The range guard the kernels apply --
 * fall back to native arithmetic for any block with an element of magnitude
 * above HYBSOL_DS_MAX -- is a property of the *pair*, not of these helpers:
 * a magnitude past a float's range makes ``hi`` infinite and every transform
 * below meaningless.
 */

#ifndef HYBSOL_DS_MATH_H
#define HYBSOL_DS_MATH_H

#include <math.h>

/** A value as a leading float and the float residual under it. */
typedef struct ds_pair
{
    float hi;
    float lo;
} ds_t;

/**
 * The largest magnitude the double-single kernels agree to carry.
 *
 * Well under a float's overflow, so a product of two carried values still
 * fits the FP32 exponent range the transforms assume.
 */
#define HYBSOL_DS_MAX 1e30

/* ------------------------------------------------------------------------- */
/* Ozaki slicing constants (shared by the two Ozaki kernel instantiations)     */
/* ------------------------------------------------------------------------- */

/**
 * The largest magnitude an Ozaki work item agrees to slice.
 *
 * The shared block scale is a power of two from the block's own maximum, and
 * the extraction below shifts a slice window by up to the full slice span; a
 * scale beyond ~2^332 would overflow the double that carries it, so an entry
 * block reaching this magnitude leaves the work item to the plain double
 * path, exactly as the pair guard above leaves a magnitude past
 * HYBSOL_DS_MAX to the native path. Powers of two up to 2^333 carry every
 * window exactly.
 */
#define HYBSOL_OZ_MAX_ENTRY 1e100

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

/** Whether every element of a block sits in the range the slicing carries. */
static __device__ inline int oz_block_fits(const double *const m, const unsigned long count)
{
    for (unsigned long i = 0; i < count; ++i)
    {
        if (m[i] <= -HYBSOL_OZ_MAX_ENTRY || m[i] >= HYBSOL_OZ_MAX_ENTRY)
            return 0;
    }
    return 1;
}

/**
 * Split an FP64 into the pair.
 *
 * Exact in the sense the scheme assumes: the residual of a double under its
 * own float rounding is exactly representable while it is normal, and
 * ``hi + lo`` then reproduces ``x`` to within a double-single's ~48
 * mantissa bits -- far past the precision a factorization keeps anyway.
 */
static __device__ inline ds_t ds_split(const double x)
{
    ds_t r;
    r.hi = (float)x;
    r.lo = (float)(x - (double)r.hi);
    return r;
}

/** Fold a pair back to FP64 for a store; the sum is exact in double. */
static __device__ inline double ds_to_double(const ds_t a)
{
    return (double)a.hi + (double)a.lo;
}

/** The negation, exact and bit-free on the pair. */
static __device__ inline ds_t ds_neg(const ds_t a)
{
    ds_t r;
    r.hi = -a.hi;
    r.lo = -a.lo;
    return r;
}

/**
 * Sum two pairs.
 *
 * Knuth's TwoSum on the leading floats captures the rounding error, which
 * joins the trailing halves and folds back in a second rounding step --
 * the result carries the exact sum to within a ulp of ``hi``.
 */
static __device__ inline ds_t ds_add(const ds_t a, const ds_t b)
{
    ds_t r;
    const float s = a.hi + b.hi;
    const float v = s - a.hi;
    float e = (a.hi - (s - v)) + (b.hi - v);
    e += a.lo + b.lo;
    r.hi = s + e;
    r.lo = e - (r.hi - s);
    return r;
}

/**
 * Product of two pairs.
 *
 * TwoProd: the fused multiply-add answers the product's error exactly, the
 * two cross terms of the trailing halves join it, and the pair re-forms
 * around the rounded product.
 */
static __device__ inline ds_t ds_mul(const ds_t a, const ds_t b)
{
    ds_t r;
    const float p = a.hi * b.hi;
    const float e = fmaf(a.hi, b.hi, -p) + a.hi * b.lo + a.lo * b.hi;
    r.hi = p + e;
    r.lo = e - (r.hi - p);
    return r;
}

/**
 * Quotient of two pairs.
 *
 * The leading float quotient, then one refinement pass: TwoProd on the
 * trial quotient, the leftover over the divisor, folded in by TwoSum --
 * the textbook double-single division truncated where its error drops
 * below a ulp of ``hi``.
 */
static __device__ inline ds_t ds_div(const ds_t a, const ds_t b)
{
    const float q = a.hi / b.hi;

    ds_t prod;
    prod.hi = q * b.hi;
    prod.lo = fmaf(q, b.hi, -prod.hi) + q * b.lo;

    const ds_t r = ds_add(a, ds_neg(prod));
    const float q2 = r.hi / b.hi;

    ds_t res;
    res.hi = q + q2;
    res.lo = (q - res.hi) + q2;
    return res;
}

#endif /* HYBSOL_DS_MATH_H */
