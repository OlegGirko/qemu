#include "cpu.h"
#include "exec/gdbstub.h"
#include "helper.h"
#include "qemu/host-utils.h"
#include "sysemu/sysemu.h"
#include "qemu/bitops.h"

uint32_t HELPER(pstate_add)(uint32_t pstate, uint64_t a1, uint64_t a2, uint64_t ar)
{
    int64_t sr = ar;

    pstate &= ~(PSTATE_N | PSTATE_Z | PSTATE_C | PSTATE_V);

    if (sr < 0) {
        pstate |= PSTATE_N;
    }

    if (!ar) {
        pstate |= PSTATE_Z;
    }

    if (ar < a1) {
        pstate |= PSTATE_C;
    } else if (ar != (a1 + a2) && ar == a1) {
	/* If result isn't what we expect it must be because we added
	   in some carry.  If so we also produce a carry when ar == a1. */
	pstate |= PSTATE_C;
    }

    if ((int64_t)((a1 ^ a2 ^ -1) & (a1 ^ ar)) < 0) {
        pstate |= PSTATE_V;
    }

    return pstate;
}

uint32_t HELPER(pstate_add32)(uint32_t pstate, uint64_t x1, uint64_t x2, uint64_t xr)
{
    uint32_t a1 = x1;
    uint32_t a2 = x2;
    uint32_t ar = xr;

    int32_t sr = ar;

    pstate &= ~(PSTATE_N | PSTATE_Z | PSTATE_C | PSTATE_V);

    if (sr < 0) {
        pstate |= PSTATE_N;
    }

    if (!ar) {
        pstate |= PSTATE_Z;
    }

    if (ar < a1) {
        pstate |= PSTATE_C;
    } else if (ar != (a1 + a2) && ar == a1) {
	pstate |= PSTATE_C;
    }

    if ((int32_t)((a1 ^ a2 ^ -1) & (a1 ^ ar)) < 0) {
        pstate |= PSTATE_V;
    }

    return pstate;
}

uint32_t HELPER(pstate_sub)(uint32_t pstate, uint64_t a1, uint64_t a2, uint64_t ar)
{
    pstate = helper_pstate_add(pstate, a1, ~a2, ar);
    return pstate;
}

uint32_t HELPER(pstate_sub32)(uint32_t pstate, uint64_t x1, uint64_t x2, uint64_t xr)
{
    uint32_t a1 = x1;
    uint32_t a2 = x2;
    uint32_t ar = xr;

    pstate = helper_pstate_add32(pstate, a1, ~a2, ar);
    return pstate;
}

uint32_t HELPER(cond)(uint32_t pstate, uint32_t cond)
{
    uint32_t r;

    switch (cond >> 1) {
    case 0:
        r = pstate & PSTATE_Z;
        break;
    case 1:
        r = pstate & PSTATE_C;
        break;
    case 2:
        r = pstate & PSTATE_N;
        break;
    case 3:
        r = pstate & PSTATE_V;
        break;
    case 4:
        r = (pstate & PSTATE_C) && !(pstate & PSTATE_Z);
        break;
    case 5:
        r = (((pstate & PSTATE_N) ? 1 : 0) == ((pstate & PSTATE_V) ? 1 : 0));
        break;
    case 6:
        r = (((pstate & PSTATE_N) ? 1 : 0) == ((pstate & PSTATE_V) ? 1 : 0))
               && !(pstate & PSTATE_Z);
        break;
    case 7:
    default:
        /* ALWAYS */
        r = 1;
        break;
    }

    if ((cond & 1) && (cond != 0xf)) {
        r = !r;
    }

//fprintf(stderr, "cond pstate=%x r = %d\n", pstate, r);

    return !!r;
}

static int get_bits(uint32_t inst, int start, int len)
{
    return (inst >> start) & ((1 << len) - 1);
}

uint32_t HELPER(ccmp)(uint32_t pstate, uint32_t insn, uint64_t op1, uint64_t op2)
{
  int flags;
  int cond = get_bits(insn, 12, 4);
  if (helper_cond(pstate, cond)) {
      uint64_t res;
      if (get_bits(insn, 30, 1)) {
	  op2 = ~op2;
	  res = op1 + op2 + 1;
      } else
	res = op1 + op2;
      if (get_bits(insn, 31, 1))
	flags = helper_pstate_add (pstate, op1, op2, res);
      else
	flags = helper_pstate_add32 (pstate, op1, op2, res);
  } else
    flags = get_bits(insn, 0, 4);
  return flags;
}

uint64_t HELPER(csel)(uint32_t pstate, uint32_t insn, uint64_t n, uint64_t m)
{
    bool else_inc = get_bits(insn, 10, 1);
    int cond = get_bits(insn, 12, 4);
    bool else_inv = get_bits(insn, 30, 1);
    bool is_32bit = !get_bits(insn, 31, 1);
    uint64_t r;

    if (helper_cond(pstate, cond)) {
        r = n;
        goto out;
    }

    r = m;
    if (else_inv) {
        r = ~r;
    }
    if (else_inc) {
        r++;
    }

out:
    if (is_32bit) {
        r = (uint32_t)r;
    }

    return r;
}

void HELPER(tb_flush)(CPUARMState *env)
{
    tb_flush(env);
}

uint64_t HELPER(sign_extend)(uint64_t x, uint64_t is_signed, uint64_t mask)
{
    if (x & is_signed) {
        x |= mask;
    }

    return x;
}

uint64_t HELPER(udiv64)(uint64_t num, uint64_t den)
{
    if (den == 0)
      return 0;
    return num / den;
}

int64_t HELPER(sdiv64)(int64_t num, int64_t den)
{
    if (den == 0)
      return 0;
    if (num == LLONG_MIN && den == -1)
      return LLONG_MIN;
    return num / den;
}

uint64_t HELPER(umulh)(uint64_t n, uint64_t m)
{
    uint64_t rl, rh;

    mulu64(&rl, &rh, n ,m);
    return rh;
}

uint64_t HELPER(smulh)(uint64_t n, uint64_t m)
{
    uint64_t rl, rh;

    muls64(&rl, &rh, n ,m);
    return rh;
}

uint64_t HELPER(rbit64)(uint64_t x)
{
    x =  ((x & 0xff00000000000000ULL) >> 56)
       | ((x & 0x00ff000000000000ULL) >> 40)
       | ((x & 0x0000ff0000000000ULL) >> 24)
       | ((x & 0x000000ff00000000ULL) >> 8)
       | ((x & 0x00000000ff000000ULL) << 8)
       | ((x & 0x0000000000ff0000ULL) << 24)
       | ((x & 0x000000000000ff00ULL) << 40)
       | ((x & 0x00000000000000ffULL) << 56);
    x =  ((x & 0xf0f0f0f0f0f0f0f0ULL) >> 4)
       | ((x & 0x0f0f0f0f0f0f0f0fULL) << 4);
    x =  ((x & 0x8888888888888888ULL) >> 3)
       | ((x & 0x4444444444444444ULL) >> 1)
       | ((x & 0x2222222222222222ULL) << 1)
       | ((x & 0x1111111111111111ULL) << 3);
    return x;
}

uint64_t HELPER(clz64)(uint64_t x)
{
    return clz64(x);
}

float64 HELPER(rintd)(float64 x, void *fp_status)
{
    return float64_round_to_int(x, fp_status);
}

float32 HELPER(rints)(float32 x, void *fp_status)
{
    return float32_round_to_int(x, fp_status);
}

void HELPER(set_rmode)(uint32_t rmode, void *fp_status)
{
    switch (rmode) {
    case ROUND_MODE_TIEEVEN:
    default:
        rmode = float_round_nearest_even;
        break;
    case ROUND_MODE_UP:
        rmode = float_round_up;
        break;
    case ROUND_MODE_DOWN:
        rmode = float_round_down;
        break;
    case ROUND_MODE_ZERO:
        rmode = float_round_to_zero;
        break;
    /* XXX add fpcr rounding (exact and not exact) */
    }

    set_float_rounding_mode(rmode, fp_status);
}

uint64_t HELPER(simd_op3s)(uint64_t op1, uint64_t op2, uint32_t insn)
{
    int size = get_bits(insn, 22, 2);
    int opcode = get_bits(insn, 11, 5);
    bool is_u = get_bits(insn, 29, 1);
    /* The neon helpers < 64bit use i32 values, to which we simply
       truncate per call.  */
    switch ((opcode << 4) | (size << 1) | is_u) {
    /* SSHL / USHL */
    case 0x80: return helper_neon_shl_s8 (op1, op2);
    case 0x81: return helper_neon_shl_u8 (op1, op2);
    case 0x82: return helper_neon_shl_s16 (op1, op2);
    case 0x83: return helper_neon_shl_u16 (op1, op2);
    case 0x84: return helper_neon_shl_s32 (op1, op2);
    case 0x85: return helper_neon_shl_u32 (op1, op2);
    case 0x86: return helper_neon_shl_s64 (op1, op2);
    case 0x87: return helper_neon_shl_u64 (op1, op2);

    /* SRSHL / URSHL (rounding) */
    case 0xa0: return helper_neon_rshl_s8 (op1, op2);
    case 0xa1: return helper_neon_rshl_u8 (op1, op2);
    case 0xa2: return helper_neon_rshl_s16 (op1, op2);
    case 0xa3: return helper_neon_rshl_u16 (op1, op2);
    case 0xa4: return helper_neon_rshl_s32 (op1, op2);
    case 0xa5: return helper_neon_rshl_u32 (op1, op2);
    case 0xa6: return helper_neon_rshl_s64 (op1, op2);
    case 0xa7: return helper_neon_rshl_u64 (op1, op2);
    default: return 0;
    }
}

uint64_t HELPER(simd_op3s_env)(CPUARMState *env, uint64_t op1, uint64_t op2,
			      uint32_t insn)
{
    int size = get_bits(insn, 22, 2);
    int opcode = get_bits(insn, 11, 5);
    bool is_u = get_bits(insn, 29, 1);
    switch ((opcode << 4) | (size << 1) | is_u) {
    /* SQSHL / UQSHL (saturating) */
    case 0x90: return helper_neon_qshl_s8 (env, op1, op2);
    case 0x91: return helper_neon_qshl_u8 (env, op1, op2);
    case 0x92: return helper_neon_qshl_s16 (env, op1, op2);
    case 0x93: return helper_neon_qshl_u16 (env, op1, op2);
    case 0x94: return helper_neon_qshl_s32 (env, op1, op2);
    case 0x95: return helper_neon_qshl_u32 (env, op1, op2);
    case 0x96: return helper_neon_qshl_s64 (env, op1, op2);
    case 0x97: return helper_neon_qshl_u64 (env, op1, op2);

    /* SQRSHL / UQRSHL (sat + round) */
    case 0xb0: return helper_neon_qrshl_s8 (env, op1, op2);
    case 0xb1: return helper_neon_qrshl_u8 (env, op1, op2);
    case 0xb2: return helper_neon_qrshl_s16 (env, op1, op2);
    case 0xb3: return helper_neon_qrshl_u16 (env, op1, op2);
    case 0xb4: return helper_neon_qrshl_s32 (env, op1, op2);
    case 0xb5: return helper_neon_qrshl_u32 (env, op1, op2);
    case 0xb6: return helper_neon_qrshl_s64 (env, op1, op2);
    case 0xb7: return helper_neon_qrshl_u64 (env, op1, op2);
    default: return 0;
    }
}

void HELPER(simd_tbl)(CPUARMState *env, uint32_t insn)
{
    int rd = get_bits(insn, 0, 5);
    int rn = get_bits(insn, 5, 5);
    int rm = get_bits(insn, 16, 5);
    bool is_tbl = get_bits(insn, 12, 1);
    int len = get_bits(insn, 13, 2);
    bool is_q = get_bits(insn, 30, 1);
    int shift, pass;
    int regs = len + 1;
    uint64_t indices;

    for (pass = 0; pass < (is_q ? 2 : 1); pass++) {
	indices = env->vfp.regs[rm * 2 + pass];
	uint64_t val = 0;
	for (shift = 0; shift < 64; shift += 8) {
	    int index = (indices >> shift) & 0xff;
	    if (index < 16 * regs) {
		/* Index counts in bytes, tabelem in 64bits,
		   and it's defined modulo.  */
		int tabelem = rn * 2 + (index >> 3);
		uint64_t tmp;
		if (tabelem >= 64)
		  tabelem -= 64;
		tmp = env->vfp.regs[tabelem] >> ((index & 7) << 3) & 0xff;
		val |= tmp << shift;
	    } else if (!is_tbl) {
		val |= env->vfp.regs[rd * 2 + pass] & (0xffUL << shift);
	    }
	}
	env->vfp.regs[rd * 2 + pass] = val;
    }
    if (!is_q)
      env->vfp.regs[rd * 2 + 1] = 0;
}

void HELPER(cmp_addr)(uint64_t addr, uint64_t pc, uint32_t insn)
{
  if (addr == 0x4003af6000ULL)
    qemu_log("cmp_addr==0x4003af6000ULL at 0x%lx, insn=0x%08x", pc, insn);
}
