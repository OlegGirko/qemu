/*
 *  AArch64 specific helpers
 *
 *  Copyright (c) 2013 Alexander Graf <agraf@suse.de>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, see <http://www.gnu.org/licenses/>.
 */

#include "cpu.h"
#include "exec/gdbstub.h"
#include "helper.h"
#include "qemu/host-utils.h"
#include "sysemu/sysemu.h"
#include "qemu/bitops.h"

/* C2.4.7 Multiply and divide */
/* special cases for 0 and LLONG_MIN are mandated by the standard */
uint64_t HELPER(udiv64)(uint64_t num, uint64_t den)
{
    if (den == 0) {
        return 0;
    }
    return num / den;
}

int64_t HELPER(sdiv64)(int64_t num, int64_t den)
{
    if (den == 0) {
        return 0;
    }
    if (num == LLONG_MIN && den == -1) {
        return LLONG_MIN;
    }
    return num / den;
}

uint64_t HELPER(clz64)(uint64_t x)
{
    return clz64(x);
}

uint64_t HELPER(cls64)(uint64_t x)
{
    return clrsb64(x);
}

uint32_t HELPER(cls32)(uint32_t x)
{
    return clrsb32(x);
}

uint64_t HELPER(rbit64)(uint64_t x)
{
    /* assign the correct byte position */
    x = bswap64(x);

    /* assign the correct nibble position */
    x = ((x & 0xf0f0f0f0f0f0f0f0ULL) >> 4)
        | ((x & 0x0f0f0f0f0f0f0f0fULL) << 4);

    /* assign the correct bit position */
    x = ((x & 0x8888888888888888ULL) >> 3)
        | ((x & 0x4444444444444444ULL) >> 1)
        | ((x & 0x2222222222222222ULL) << 1)
        | ((x & 0x1111111111111111ULL) << 3);

    return x;
}

/* Convert a softfloat float_relation_ (as returned by
 * the float*_compare functions) to the correct ARM
 * NZCV flag state.
 */
static inline uint32_t float_rel_to_flags(int res)
{
    uint64_t flags;
    switch (res) {
    case float_relation_equal:
        flags = PSTATE_Z | PSTATE_C;
        break;
    case float_relation_less:
        flags = PSTATE_N;
        break;
    case float_relation_greater:
        flags = PSTATE_C;
        break;
    case float_relation_unordered:
    default:
        flags = PSTATE_C | PSTATE_V;
        break;
    }
    return flags;
}

uint64_t HELPER(vfp_cmps_a64)(float32 x, float32 y, void *fp_status)
{
    return float_rel_to_flags(float32_compare_quiet(x, y, fp_status));
}

uint64_t HELPER(vfp_cmpes_a64)(float32 x, float32 y, void *fp_status)
{
    return float_rel_to_flags(float32_compare(x, y, fp_status));
}

uint64_t HELPER(vfp_cmpd_a64)(float64 x, float64 y, void *fp_status)
{
    return float_rel_to_flags(float64_compare_quiet(x, y, fp_status));
}

uint64_t HELPER(vfp_cmped_a64)(float64 x, float64 y, void *fp_status)
{
    return float_rel_to_flags(float64_compare(x, y, fp_status));
}

float32 HELPER(vfp_mulxs)(float32 a, float32 b, void *fpstp)
{
    float_status *fpst = fpstp;

    if ((float32_is_zero(a) && float32_is_infinity(b)) ||
        (float32_is_infinity(a) && float32_is_zero(b))) {
        /* 2.0 with the sign bit set to sign(A) XOR sign(B) */
        return make_float32((1U << 30) |
                            ((float32_val(a) ^ float32_val(b)) & (1U << 31)));
    }
    return float32_mul(a, b, fpst);
}

float64 HELPER(vfp_mulxd)(float64 a, float64 b, void *fpstp)
{
    float_status *fpst = fpstp;

    if ((float64_is_zero(a) && float64_is_infinity(b)) ||
        (float64_is_infinity(a) && float64_is_zero(b))) {
        /* 2.0 with the sign bit set to sign(A) XOR sign(B) */
        return make_float64((1ULL << 62) |
                            ((float64_val(a) ^ float64_val(b)) & (1ULL << 63)));
    }
    return float64_mul(a, b, fpst);
}

uint64_t HELPER(simd_tbl)(CPUARMState *env, uint64_t result, uint64_t indices,
                          uint32_t rn, uint32_t numregs)
{
    /* Helper function for SIMD TBL and TBX. We have to do the table
     * lookup part for the 64 bits worth of indices we're passed in.
     * result is the initial results vector (either zeroes for TBL
     * or some guest values for TBX), rn the register number where
     * the table starts, and numregs the number of registers in the table.
     * We return the results of the lookups.
     */
    int shift;

    for (shift = 0; shift < 64; shift += 8) {
        int index = extract64(indices, shift, 8);
        if (index < 16 * numregs) {
            /* Convert index (a byte offset into the virtual table
             * which is a series of 128-bit vectors concatenated)
             * into the correct vfp.regs[] element plus a bit offset
             * into that element, bearing in mind that the table
             * can wrap around from V31 to V0.
             */
            int elt = (rn * 2 + (index >> 3)) % 64;
            int bitidx = (index & 7) * 8;
            uint64_t val = extract64(env->vfp.regs[elt], bitidx, 8);

            result = deposit64(result, shift, 8, val);
        }
    }
    return result;
}

/* Helper function for 64 bit polynomial multiply case:
 * perform PolynomialMult(op1, op2) and return either the top or
 * bottom half of the 128 bit result.
 */
uint64_t HELPER(neon_pmull_64_lo)(uint64_t op1, uint64_t op2)
{
    int bitnum;
    uint64_t res = 0;

    for (bitnum = 0; bitnum < 64; bitnum++) {
        if (op1 & (1ULL << bitnum)) {
            res ^= op2 << bitnum;
        }
    }
    return res;
}
uint64_t HELPER(neon_pmull_64_hi)(uint64_t op1, uint64_t op2)
{
    int bitnum;
    uint64_t res = 0;

    /* bit 0 of op1 can't influence the high 64 bits at all */
    for (bitnum = 1; bitnum < 64; bitnum++) {
        if (op1 & (1ULL << bitnum)) {
            res ^= op2 >> (64 - bitnum);
        }
    }
    return res;
}

/* 64bit/double versions of the neon float compare functions */
uint64_t HELPER(neon_ceq_f64)(float64 a, float64 b, void *fpstp)
{
    float_status *fpst = fpstp;
    return -float64_eq_quiet(a, b, fpst);
}

uint64_t HELPER(neon_cge_f64)(float64 a, float64 b, void *fpstp)
{
    float_status *fpst = fpstp;
    return -float64_le(b, a, fpst);
}

uint64_t HELPER(neon_cgt_f64)(float64 a, float64 b, void *fpstp)
{
    float_status *fpst = fpstp;
    return -float64_lt(b, a, fpst);
}

/* Reciprocal step and sqrt step. Note that unlike the A32/T32
 * versions, these do a fully fused multiply-add or
 * multiply-add-and-halve.
 */
#define float32_two make_float32(0x40000000)
#define float32_three make_float32(0x40400000)
#define float32_one_point_five make_float32(0x3fc00000)

#define float64_two make_float64(0x4000000000000000ULL)
#define float64_three make_float64(0x4008000000000000ULL)
#define float64_one_point_five make_float64(0x3FF8000000000000ULL)

float32 HELPER(recpsf_f32)(float32 a, float32 b, void *fpstp)
{
    float_status *fpst = fpstp;

    a = float32_chs(a);
    if ((float32_is_infinity(a) && float32_is_zero(b)) ||
        (float32_is_infinity(b) && float32_is_zero(a))) {
        return float32_two;
    }
    return float32_muladd(a, b, float32_two, 0, fpst);
}

float64 HELPER(recpsf_f64)(float64 a, float64 b, void *fpstp)
{
    float_status *fpst = fpstp;

    a = float64_chs(a);
    if ((float64_is_infinity(a) && float64_is_zero(b)) ||
        (float64_is_infinity(b) && float64_is_zero(a))) {
        return float64_two;
    }
    return float64_muladd(a, b, float64_two, 0, fpst);
}

float32 HELPER(rsqrtsf_f32)(float32 a, float32 b, void *fpstp)
{
    float_status *fpst = fpstp;

    a = float32_chs(a);
    if ((float32_is_infinity(a) && float32_is_zero(b)) ||
        (float32_is_infinity(b) && float32_is_zero(a))) {
        return float32_one_point_five;
    }
    return float32_muladd(a, b, float32_three, float_muladd_halve_result, fpst);
}

float64 HELPER(rsqrtsf_f64)(float64 a, float64 b, void *fpstp)
{
    float_status *fpst = fpstp;

    a = float64_chs(a);
    if ((float64_is_infinity(a) && float64_is_zero(b)) ||
        (float64_is_infinity(b) && float64_is_zero(a))) {
        return float64_one_point_five;
    }
    return float64_muladd(a, b, float64_three, float_muladd_halve_result, fpst);
}

/* Reciprocal functions */

/* FPRecipEstimate()
 *
 * This is a slightly different front-end to process the 64 bit float
 * but the actual estimation algolrithm is shared with the NEON equivalent.
 */

typedef struct {
    float64  f;
    uint64_t val64;
    uint64_t sbit;
    int64_t exp;
    uint64_t frac;
} ARMUnpackedF64;

typedef struct {
    float32  f;
    uint32_t val32;
    uint32_t sbit;
    int32_t exp;
    uint32_t frac;
} ARMUnpackedF32;

/* FPUnpack, also may squash input denormals */
static ARMUnpackedF64 unpack_f64(float64 a, bool check, float_status *fpst)
{
    ARMUnpackedF64 result;
    if (check) {
        result.f = float64_squash_input_denormal(a, fpst);
    } else {
        result.f = a;
    }
    result.val64 = float64_val(result.f);
    result.sbit = 0x8000000000000000ULL & result.val64;
    result.exp = extract64(result.val64, 52, 11);
    result.frac = extract64(result.val64, 0, 52);
    return result;
}

static ARMUnpackedF32 unpack_f32(float32 a, bool check, float_status *fpst)
{
    ARMUnpackedF32 result;
    if (check) {
        result.f = float32_squash_input_denormal(a, fpst);
    } else {
        result.f = a;
    }
    result.val32 = float32_val(result.f);
    result.sbit = 0x80000000ULL & result.val32;
    result.exp = extract32(result.val32, 23, 8);
    result.frac = extract32(result.val32, 0, 23);
    return result;
}

static bool handle_float32_nan(float32 *num, float_status *fpst)
{
    float32 a = *num;

    if (float32_is_any_nan(a)) {
        if (float32_is_signaling_nan(a)) {
            float_raise(float_flag_invalid, fpst);
            *num = float32_maybe_silence_nan(a);
        }
        if (fpst->default_nan_mode) {
            *num = float32_default_nan;
        }
        return true;
    }
    return false;
}

static bool handle_float64_nan(float64 *num, float_status *fpst)
{
    float64 a = *num;

    if (float64_is_any_nan(a)) {
        if (float64_is_signaling_nan(a)) {
            float_raise(float_flag_invalid, fpst);
            *num = float64_maybe_silence_nan(a);
        }
        if (fpst->default_nan_mode) {
            *num = float64_default_nan;
        }
        return true;
    }
    return false;
}

static bool do_we_round_to_infinity(float_status *fpst, bool sign_bit)
{
    switch (fpst->float_rounding_mode) {
    case float_round_nearest_even: /* Round to Nearest */
        return true;
    case float_round_up: /* Round to +Inf */
        return !sign_bit;
    case float_round_down: /* Round to -Inf */
        return sign_bit;
    case float_round_to_zero: /* Round to Zero */
        return false;
    }

    g_assert_not_reached();
    return false;
}

/* Common wrapper to call recip_estimate */
static ARMUnpackedF64 call_recip_estimate(ARMUnpackedF64 num,
                                          int exp_offset, float_status *fpst) {
    ARMUnpackedF64 result;
    float64 scaled, estimate;

    /* Generate the scaled number for the estimate function */
    if (num.exp == 0) {
        if (extract64(num.frac, 51, 1) == 0) {
            num.exp = -1;
            num.frac = extract64(num.frac, 0, 50) << 2;
        } else {
            num.frac = extract64(num.frac, 0, 51) << 1;
        }
    }

    /* scaled = '0' : '01111111110' : fraction<51:44> : Zeros(44); */
    scaled = make_float64((0x3feULL << 52)
                          |extract64(num.frac, 44, 8) << 44);

    estimate = recip_estimate(scaled, fpst);

    /* Build new result */
    result = unpack_f64(estimate, false, fpst);
    result.exp = exp_offset - num.exp;

    if (result.exp == 0) {
        result.frac = 1ULL << 51 | extract64(result.frac, 1, 51);
    } else if (result.exp == -1) {
        result.frac = 1ULL << 50 | extract64(result.frac, 2, 50);
        result.exp = 0;
    }

    return result;
}


float32 HELPER(frecpe_f32)(float32 input, void *fpstp)
{
    float_status *fpst = fpstp;
    ARMUnpackedF32 f32;
    ARMUnpackedF64 f64, r64;

    f32 = unpack_f32(input, true, fpst);

    if (handle_float32_nan(&f32.f, fpst)) {
        return f32.f;
    } else if (float32_is_infinity(f32.f)) {
        return float32_set_sign(float32_zero, float32_is_neg(f32.f));
    } else if (float32_is_zero(f32.f)) {
        float_raise(float_flag_divbyzero, fpst);
        return float32_set_sign(float32_infinity, float32_is_neg(f32.f));
    } else if ((f32.val32 & ~(1ULL << 31)) < (1ULL << 21)) {
        /* Abs(value) < 2.0^-128 */
        float_raise(float_flag_overflow | float_flag_inexact, fpst);
        if (do_we_round_to_infinity(fpst, f32.sbit)) {
            return float32_set_sign(float32_infinity, float32_is_neg(f32.f));
        } else {
            return float32_set_sign(float32_maxnorm, float32_is_neg(f32.f));
        }
    } else if (f32.exp >= 253 && fpst->flush_to_zero) {
        float_raise(float_flag_underflow, fpst);
        return float32_set_sign(float32_zero, float32_is_neg(f32.f));
    }

    f64.exp = f32.exp;
    f64.frac = (int64_t)(f32.frac) << 29;
    r64 = call_recip_estimate(f64, 253, fpst);

    /* result = sign : result_exp<7:0> : fraction<51:29>; */
    return make_float32(f32.sbit |
                        (r64.exp & 0xff) << 23 |
                        extract64(r64.frac, 29, 24));
}


float64 HELPER(frecpe_f64)(float64 input, void *fpstp)
{
    float_status *fpst = fpstp;
    ARMUnpackedF64 f64, r64;

    f64 = unpack_f64(input, true, fpst);

    /* Deal with any special cases */
    if (handle_float64_nan(&f64.f, fpst)) {
        return f64.f;
    } else if (float64_is_infinity(f64.f)) {
        return float64_set_sign(float64_zero, float64_is_neg(f64.f));
    } else if (float64_is_zero(f64.f)) {
        float_raise(float_flag_divbyzero, fpst);
        return float64_set_sign(float64_infinity, float64_is_neg(f64.f));
    } else if ((f64.val64 & ~(1ULL << 63)) < (1ULL << 50)) {
        /* Abs(value) < 2.0^-1024 */
        float_raise(float_flag_overflow | float_flag_inexact, fpst);
        if (do_we_round_to_infinity(fpst, f64.sbit)) {
            return float64_set_sign(float64_infinity, float64_is_neg(f64.f));
        } else {
            return float64_set_sign(float64_maxnorm, float64_is_neg(f64.f));
        }
    } else if (f64.exp >= 1023 && fpst->flush_to_zero) {
        float_raise(float_flag_underflow, fpst);
        return float64_set_sign(float64_zero, float64_is_neg(f64.f));
    }

    r64 = call_recip_estimate(f64, 2045, fpst);

    /* result = sign : result_exp<10:0> : fraction<51:0> */
    return make_float64(f64.sbit |
                        ((r64.exp & 0x7ff) << 52) |
                        r64.frac);
}

/* Floating-point reciprocal exponent - see FPRecpX in ARM ARM */
float32 HELPER(frecpx_f32)(float32 a, void *fpstp)
{
    ARMUnpackedF32 f32 = unpack_f32(a, false, NULL);
    float_status *fpst = fpstp;

    if (handle_float32_nan(&f32.f, fpst)) {
        return f32.f;
    }

    if (f32.exp == 0) {
        return make_float32(f32.sbit | (0xfe << 23));
    } else {
        return make_float32(f32.sbit | (~f32.exp & 0xff) << 23);
    }
}

float64 HELPER(frecpx_f64)(float64 a, void *fpstp)
{
    ARMUnpackedF64 f64 = unpack_f64(a, false, NULL);
    float_status *fpst = fpstp;

    if (handle_float64_nan(&a, fpst)) {
        return a;
    }

    if (f64.exp == 0) {
        return make_float64(f64.sbit | (0x7feULL << 52));
    } else {
        return make_float64(f64.sbit | (~f64.exp & 0x7ffULL) << 52);
    }
}
