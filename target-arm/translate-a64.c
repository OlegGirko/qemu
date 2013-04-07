/*
 *  ARM translation
 *
 *  Copyright (c) 2003 Fabrice Bellard
 *  Copyright (c) 2005-2007 CodeSourcery
 *  Copyright (c) 2007 OpenedHand, Ltd.
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
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <inttypes.h>

#include "cpu.h"
#include "disas/disas.h"
#include "tcg-op.h"
#include "qemu/log.h"
#include "translate.h"
#include "qemu/host-utils.h"

#include "helper.h"
#define GEN_HELPER 1
#include "helper.h"

// #define DEBUG_SINGLESTEP 1
// #define DEBUG_FLUSH 1

static TCGv_i64 cpu_X[32];
static TCGv_i64 cpu_pc;
static TCGv_i64 cpu_sp;
static TCGv_i32 pstate;

static TCGv_i64 cpu_exclusive_addr;
static TCGv_i64 cpu_exclusive_val;
static TCGv_i64 cpu_exclusive_high;
#ifdef CONFIG_USER_ONLY
static TCGv_i64 cpu_exclusive_test;
static TCGv_i64 cpu_exclusive_info;
#endif

static const char *regnames[] =
    { "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7",
      "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15",
      "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23",
      "x24", "x25", "x26", "x27", "x28", "x29", "lr", "xzr" };

/* initialize TCG globals.  */
void a64_translate_init(void)
{
    int i;

    cpu_pc = tcg_global_mem_new_i64(TCG_AREG0,
                                    offsetof(CPUARMState, pc),
                                    "pc");
    cpu_sp = tcg_global_mem_new_i64(TCG_AREG0,
                                    offsetof(CPUARMState, sp),
                                    "sp");
    for (i = 0; i < 32; i++) {
        cpu_X[i] = tcg_global_mem_new_i64(TCG_AREG0,
                                          offsetof(CPUARMState, xregs[i]),
                                          regnames[i]);
    }

    pstate = tcg_global_mem_new_i32(TCG_AREG0,
                                    offsetof(CPUARMState, pstate),
                                    "pstate");

    cpu_exclusive_addr = tcg_global_mem_new_i64(TCG_AREG0,
        offsetof(CPUARMState, exclusive_addr), "exclusive_addr");
    cpu_exclusive_val = tcg_global_mem_new_i64(TCG_AREG0,
        offsetof(CPUARMState, exclusive_val), "exclusive_val");
    cpu_exclusive_high = tcg_global_mem_new_i64(TCG_AREG0,
        offsetof(CPUARMState, exclusive_high), "exclusive_high");
#ifdef CONFIG_USER_ONLY
    cpu_exclusive_test = tcg_global_mem_new_i64(TCG_AREG0,
        offsetof(CPUARMState, exclusive_test), "exclusive_test");
    cpu_exclusive_info = tcg_global_mem_new_i64(TCG_AREG0,
        offsetof(CPUARMState, exclusive_info), "exclusive_info");
#endif
}

void cpu_dump_state_a64(CPUARMState *env, FILE *f, fprintf_function cpu_fprintf,
                        int flags)
{
    int i;

    cpu_fprintf(f, "PC=%016"PRIx64"  SP=%016"PRIx64"\n", env->pc, env->sp);
    for(i = 0; i < 31; i++) {
        cpu_fprintf(f, "X%02d=%016"PRIx64, i, env->xregs[i]);
        if ((i % 4) == 3)
            cpu_fprintf(f, "\n");
        else
            cpu_fprintf(f, " ");
    }
    cpu_fprintf(f, "XZR=%016"PRIx64"\n", env->xregs[31]);
    cpu_fprintf(f, "PSTATE=%c%c%c%c\n",
        env->pstate & PSTATE_N ? 'n' : '.',
        env->pstate & PSTATE_Z ? 'z' : '.',
        env->pstate & PSTATE_C ? 'c' : '.',
        env->pstate & PSTATE_V ? 'v' : '.');
    cpu_fprintf(f, "\n");

    if (1) { //flags & CPU_DUMP_FPU) {
        int numvfpregs = 32;
        for (i = 0; i < numvfpregs; i++) {
            uint64_t v = float64_val(env->vfp.regs[i * 2]);
            uint64_t v1 = float64_val(env->vfp.regs[(i * 2) + 1]);
            if (!v && !v1) continue;
            cpu_fprintf(f, "d%02d.0=%016" PRIx64 " " "d%02d.0=%016" PRIx64 "\n",
                        i, v, i, v1);
        }
        cpu_fprintf(f, "FPSCR: %08x\n", (int)env->vfp.xregs[ARM_VFP_FPSCR]);
    }
}

static int get_bits(uint32_t inst, int start, int len)
{
    return (inst >> start) & ((1 << len) - 1);
}

static int get_sbits(uint32_t inst, int start, int len)
{
    int r = get_bits(inst, start, len);
    if (r & (1 << (len - 1))) {
        /* Extend the MSB 1 to the higher bits */
        r |= -1 & ~((1ULL << len) - 1);
    }
    return r;
}

static int get_reg(uint32_t inst)
{
    return get_bits(inst, 0, 5);
}

static TCGv_i64 cpu_reg(int reg)
{
    if (reg == 31) {
        /* XXX leaks temps */
        return tcg_const_i64(0);
    } else {
        return cpu_X[reg];
    }
}

static TCGv_i64 cpu_reg_sp(int reg)
{
    if (reg == 31) {
        return cpu_sp;
    } else {
        return cpu_X[reg];
    }
}

static TCGv_ptr get_fpstatus_ptr(void)
{
    TCGv_ptr statusptr = tcg_temp_new_ptr();
    int offset;

    offset = offsetof(CPUARMState, vfp.standard_fp_status);
    tcg_gen_addi_ptr(statusptr, cpu_env, offset);

    return statusptr;
}

static void clear_fpreg(int dest)
{
    int freg_offs = offsetof(CPUARMState, vfp.regs[dest * 2]);
    TCGv_i64 tcg_zero = tcg_const_i64(0);

    tcg_gen_st_i64(tcg_zero, cpu_env, freg_offs);
    tcg_gen_st_i64(tcg_zero, cpu_env, freg_offs + sizeof(float64));
}

static int get_mem_index(DisasContext *s)
{
    /* XXX only user mode for now */
    return 1;
}

void gen_a64_set_pc_im(uint64_t val)
{
    tcg_gen_movi_i64(cpu_pc, val);
}

static void gen_exception(int excp)
{
    TCGv_i32 tmp = tcg_temp_new_i32();
    tcg_gen_movi_i32(tmp, excp);
    gen_helper_exception(cpu_env, tmp);
    tcg_temp_free_i32(tmp);
}

static void gen_exception_insn(DisasContext *s, int offset, int excp)
{
    gen_a64_set_pc_im(s->pc - offset);
    gen_exception(excp);
    s->is_jmp = DISAS_JUMP;
}

static void _unallocated_encoding(DisasContext *s)
{
    fprintf(stderr, "Unknown instruction: %#x\n",
            arm_ldl_code(current_cpu->env_ptr, s->pc - 4, s->bswap_code));
    gen_exception_insn(s, 4, EXCP_UDEF);
}

#define unallocated_encoding(s) do { \
    fprintf(stderr, "unallocated encoding at line: %d\n", __LINE__); \
    _unallocated_encoding(s); \
    } while(0)

static void reserved(DisasContext *s, uint32_t insn, int start, int len,
                     uint32_t val)
{
    uint32_t field = get_bits(insn, start, len);
    if (field != val) {
        fprintf(stderr, "Broken ins: %08x %08x %08x", insn, field, val);
        unallocated_encoding(s);
    }
}

static inline void gen_goto_tb(DisasContext *s, int n, uint64_t dest)
{
    TranslationBlock *tb;

    tb = s->tb;
    if ((tb->pc & TARGET_PAGE_MASK) == (dest & TARGET_PAGE_MASK)) {
        tcg_gen_goto_tb(n);
        gen_a64_set_pc_im(dest);
        tcg_gen_exit_tb((tcg_target_long)tb + n);
    } else {
        gen_a64_set_pc_im(dest);
        tcg_gen_exit_tb(0);
    }
}

static void handle_b(DisasContext *s, uint32_t insn)
{
    uint64_t addr = s->pc - 4 + (get_sbits(insn, 0, 26) << 2);

    if (get_bits(insn, 31, 1)) {
        /* BL */
        tcg_gen_movi_i64(cpu_reg(30), s->pc);
    }
    gen_goto_tb(s, 0, addr);
    s->is_jmp = DISAS_TB_JUMP;
}

static void handle_br(DisasContext *s, uint32_t insn)
{
    int branch_type = get_bits(insn, 21, 2);
    int source = get_bits(insn, 5, 5);

    switch (branch_type) {
    case 0: /* JMP */
        break;
    case 1: /* CALL */
        tcg_gen_movi_i64(cpu_reg(30), s->pc);
        break;
    case 2: /* RET */
        source = 30;
        break;
    case 3:
        unallocated_encoding(s);
        return;
    }

    tcg_gen_mov_i64(cpu_pc, cpu_reg(source));
    s->is_jmp = DISAS_JUMP;
}

static void handle_condb(DisasContext *s, uint32_t insn)
{
    uint64_t addr = s->pc - 4 + (get_sbits(insn, 5, 19) << 2);
    int cond = get_bits(insn, 0, 4);
    int no_match;
    TCGv_i32 tcg_zero = tcg_const_i32(0);
    TCGv_i32 tcg_cond = tcg_const_i32(cond);
    TCGv_i32 tcg_condmatch = tcg_temp_new_i32();

    no_match = gen_new_label();

    gen_helper_cond(tcg_condmatch, pstate, tcg_cond);
    tcg_gen_brcond_i32(TCG_COND_EQ, tcg_condmatch, tcg_zero, no_match);

    gen_goto_tb(s, 0, addr);

    gen_set_label(no_match);
    gen_goto_tb(s, 1, s->pc);

    tcg_temp_free_i32(tcg_zero);
    tcg_temp_free_i32(tcg_cond);
    tcg_temp_free_i32(tcg_condmatch);

    s->is_jmp = DISAS_TB_JUMP;
}

static void handle_cb(DisasContext *s, uint32_t insn)
{
    uint64_t addr = s->pc - 4 + (get_sbits(insn, 5, 19) << 2);
    bool is_zero = !get_bits(insn, 24, 1);
    bool is_32bit = !get_bits(insn, 31, 1);
    int dest = get_reg(insn);
    int no_match;
    TCGv_i64 tcg_cmp, tcg_zero;

    tcg_cmp = tcg_temp_new_i64();
    tcg_zero = tcg_const_i64(0);

    if (is_32bit) {
        tcg_gen_ext32u_i64(tcg_cmp, cpu_reg(dest));
    } else {
        tcg_gen_mov_i64(tcg_cmp, cpu_reg(dest));
    }

    no_match = gen_new_label();
    if (is_zero) {
        tcg_gen_brcond_i64(TCG_COND_NE, tcg_cmp, tcg_zero, no_match);
    } else {
        tcg_gen_brcond_i64(TCG_COND_EQ, tcg_cmp, tcg_zero, no_match);
    }
    gen_goto_tb(s, 0, addr);

    gen_set_label(no_match);
    gen_goto_tb(s, 1, s->pc);

    s->is_jmp = DISAS_TB_JUMP;

    tcg_temp_free_i64(tcg_cmp);
    tcg_temp_free_i64(tcg_zero);
}

static void handle_tbz(DisasContext *s, uint32_t insn)
{
    uint64_t addr = s->pc - 4 + (get_sbits(insn, 5, 14) << 2);
    bool is_one = get_bits(insn, 24, 1);
    int shift = get_bits(insn, 19, 5) | (get_bits(insn, 31, 1) << 5);
    int source = get_reg(insn);
    int no_match;
    uint64_t mask = 1ULL << shift;
    TCGv_i64 tcg_cmp, tcg_mask;

    tcg_cmp = tcg_temp_new_i64();
    tcg_mask = tcg_const_i64(mask);
    tcg_gen_and_i64(tcg_cmp, cpu_reg(source), tcg_mask);

    no_match = gen_new_label();
    if (is_one) {
        tcg_gen_brcond_i64(TCG_COND_NE, tcg_cmp, tcg_mask, no_match);
    } else {
        tcg_gen_brcond_i64(TCG_COND_EQ, tcg_cmp, tcg_mask, no_match);
    }
    gen_goto_tb(s, 0, addr);
    tcg_gen_exit_tb(0);

    gen_set_label(no_match);
    gen_goto_tb(s, 1, s->pc);

    tcg_temp_free_i64(tcg_cmp);
    tcg_temp_free_i64(tcg_mask);

    s->is_jmp = DISAS_TB_JUMP;
}

static void handle_msr(DisasContext *s, uint32_t insn)
{
    int dest = get_reg(insn);
    int op0 = get_bits(insn, 19, 2);
    int op1 = get_bits(insn, 16, 3);
    int crm = get_bits(insn, 8, 4);
    int crn = get_bits(insn, 12, 4);
    int op2 = get_bits(insn, 5, 3);

    /* XXX what are these? */
    if (op0 == 3 && op1 == 3 && op2 == 2 && !crm && crn == 13) {
        tcg_gen_st_i64(cpu_reg(dest), cpu_env, offsetof(CPUARMState, sr.tpidr_el0));
    } else if (op0 == 3 && op1 == 3 && (op2 == 0 || op2 == 1) && crm == 4 && crn == 4) {
        /* XXX this is wrong! */
        tcg_gen_st32_i64(cpu_reg(dest), cpu_env,
            offsetof(CPUARMState, vfp.xregs[ARM_VFP_FPSCR]));
    } else {
        fprintf(stderr, "MSR: %d %d %d %d %d\n", op0, op1, op2, crm, crn);
        unallocated_encoding(s);
    }
}

static void handle_mrs(DisasContext *s, uint32_t insn)
{
    int dest = get_reg(insn);
    int op0 = get_bits(insn, 19, 2);
    int op1 = get_bits(insn, 16, 3);
    int crm = get_bits(insn, 8, 4);
    int crn = get_bits(insn, 12, 4);
    int op2 = get_bits(insn, 5, 3);

    /* XXX what are these? */
    if (op0 == 3 && op1 == 3 && op2 == 2 && !crm && crn == 13) {
        tcg_gen_ld_i64(cpu_reg(dest), cpu_env, offsetof(CPUARMState, sr.tpidr_el0));
    } else if (op0 == 3 && op1 == 3 && (op2 == 0 || op2 == 1) && crm == 4 && crn == 4) {
        /* XXX this is wrong! */
        tcg_gen_ld32u_i64(cpu_reg(dest), cpu_env,
            offsetof(CPUARMState, vfp.xregs[ARM_VFP_FPSCR]));
    } else if (op0 == 3 && op1 == 3 && op2 == 1 && crm == 0 && crn == 0) {
        /* ctr_el0 */
        /* CTR_EL0 [3:0] contains log2 of icache line size in words.
           CTR_EL0 [19:16] contains log2 of dcache line size in words.  */
        tcg_gen_movi_i64(cpu_reg(dest), 0x30003);
    } else if (op0 == 3 && op1 == 3 && op2 == 7 && crm == 0 && crn == 0) {
	/* dczid_el0 */
	/* [3:0] contain log2(blocksize cleared by "dc zva" instruction)
	   [4] is tested by glibc, and if set dc zva isn't used. */
	tcg_gen_movi_i64(cpu_reg(dest), 0x10);
    } else {
        fprintf(stderr, "MRS: %d %d %d %d %d\n", op0, op1, op2, crm, crn);
        unallocated_encoding(s);
    }
}

static void handle_sys(DisasContext *s, uint32_t insn)
{
    int rt = get_reg(insn);
    int op2 = get_bits(insn, 5, 3);
    int crn = get_bits(insn, 12, 4);
    int op1 = get_bits(insn, 16, 3);
    int op0 = get_bits(insn, 19, 2);
    bool is_l = get_bits(insn, 21, 1);
    /* XXX simply ignore sys for now, need to start worrying when we implement
           system emulation */
    /* XXX Some sys insns can also be done in userspace, e.g. 'dc zva',
       which clears memory (part of the data-cache sys ops).  */
    if ((insn & ~0xf) == 0xd50b7420) {
	unallocated_encoding(s);
	return;
    }
    if (op1 == 3 && crn == 3 && !is_l && op0 == 0 && op2 == 2 && rt == 31) {
	/* CLREX */
	tcg_gen_movi_i64(cpu_exclusive_addr, -1);
    }
}

/* PC relative address calculation */
static void handle_adr(DisasContext *s, uint32_t insn)
{
    int reg = get_reg(insn);
    int is_page = get_bits(insn, 31, 1);
    uint64_t imm;
    uint64_t base;

    imm = get_sbits(insn, 5, 19) << 2;
    imm |= get_bits(insn, 29, 2);

    base = s->pc - 4;
    if (is_page) {
        /* ADRP (page based) */
        base &= ~0xFFFULL;
        imm <<= 12;
    }

    tcg_gen_movi_i64(cpu_reg(reg), base + imm);
}

static void handle_movi(DisasContext *s, uint32_t insn)
{
    int reg = get_reg(insn);
    uint64_t imm = get_bits(insn, 5, 16);
    int is_32bit = !get_bits(insn, 31, 1);
    int is_k = get_bits(insn, 29, 1);
    int is_n = !get_bits(insn, 30, 1);
    int pos = get_bits(insn, 21, 2) << 4;
    TCGv_i64 tcg_imm;

    reserved(s, insn, 23, 1, 1);

    if (is_k && is_n) {
        unallocated_encoding(s);
    }

    if (is_k) {
        tcg_imm = tcg_const_i64(imm);
        tcg_gen_deposit_i64(cpu_reg(reg), cpu_reg(reg), tcg_imm, pos, 16);
        tcg_temp_free_i64(tcg_imm);
    } else {
        tcg_gen_movi_i64(cpu_reg(reg), imm << pos);
    }

    if (is_n) {
        tcg_gen_not_i64(cpu_reg(reg), cpu_reg(reg));
    }

    if (is_32bit) {
        tcg_gen_ext32u_i64(cpu_reg(reg), cpu_reg(reg));
    }
}

static uint64_t replicate(uint64_t mask, int esize)
{
    int i;
    uint64_t out_mask = 0;
    for (i = 0; (i * esize) < 64; i++) {
        out_mask = out_mask | (mask << (i * esize));
    }
    return out_mask;
}

static uint64_t bitmask(int len)
{
    if (len == 64) {
        return -1;
    }
    return (1ULL << len) - 1;
}

#define MASK_TMASK 0
#define MASK_WMASK 1
static uint64_t decode_mask(int immn, int imms, int immr, int type)
{
    uint64_t mask;
    int len = 31 - clz32((immn << 6) | (~imms & 0x3f));
    int esize = 1 << len;
    int levels = (esize - 1) & 0x3f;
    int s = imms & levels;
    int r = immr & levels;
    int d = (s - r) & 0x3f;

    if (type == MASK_WMASK) {
        mask = bitmask(s + 1);
        mask = ((mask >> r) | (mask << (esize - r)));
        mask &= bitmask(esize);
        mask = replicate(mask, esize);
    } else {
        mask = bitmask(d + 1);
    }

    return mask;
}

#if 0
static uint64_t decode_tmask(int immn, int imms, int immr)
{
    return decode_mask(immn, imms, immr, MASK_TMASK);
}
#endif

static uint64_t decode_wmask(int immn, int imms, int immr)
{
    return decode_mask(immn, imms, immr, MASK_WMASK);
}

static void handle_orri(DisasContext *s, uint32_t insn)
{
    int is_32bit = !get_bits(insn, 31, 1);
    int is_n = get_bits(insn, 22, 1);
    int opc = get_bits(insn, 29, 2);
    int dest = get_reg(insn);
    int source = get_bits(insn, 5, 5);
    int imms = get_bits(insn, 10, 6);
    int immr = get_bits(insn, 16, 6);
    TCGv_i64 tcg_dst;
    TCGv_i64 tcg_op2;
    bool setflags = false;
    uint64_t wmask;

    if (is_32bit && is_n) {
        /* reserved */
        unallocated_encoding(s);
    }

    if (opc == 0x3) {
        setflags = true;
    }

    if (setflags) {
        tcg_dst = cpu_reg(dest);
    } else {
        tcg_dst = cpu_reg_sp(dest);
    }

    wmask = decode_wmask(is_n, imms, immr);
    if (is_32bit) {
        wmask = (uint32_t)wmask;
    }
    tcg_op2 = tcg_const_i64(wmask);

    switch (opc) {
    case 0x3:
    case 0x0:
        tcg_gen_and_i64(tcg_dst, cpu_reg(source), tcg_op2);
        break;
    case 0x1:
        tcg_gen_or_i64(tcg_dst, cpu_reg(source), tcg_op2);
        break;
    case 0x2:
        tcg_gen_xor_i64(tcg_dst, cpu_reg(source), tcg_op2);
        break;
    }

    if (is_32bit) {
        tcg_gen_ext32u_i64(tcg_dst, tcg_dst);
    }

    if (setflags) {
        gen_helper_pstate_add(pstate, pstate, tcg_dst, cpu_reg(31), tcg_dst);
    }

    tcg_temp_free_i64(tcg_op2);
}

static TCGv_i64 get_shift(int reg, int shift_type, TCGv_i64 tcg_shift,
                          int is_32bit)
{
    TCGv_i64 r;

    r = tcg_temp_new_i64();

    /* XXX carry_out */
    /* Careful with the width.  We work on 64bit, but must make sure
       that we zero-extend the result on out, and ignore any upper bits,
       that might still be set in that register.  */
    switch (shift_type) {
    case 0: /* LSL */
	/* left shift is easy, simply zero-extend on out */
        tcg_gen_shl_i64(r, cpu_reg(reg), tcg_shift);
	if (is_32bit)
	  tcg_gen_ext32u_i64 (r, r);
        break;
    case 1: /* LSR */
	/* For logical right shift we zero extend first, to zero
	   the upper bits.  We don't need to extend on out.  */
	if (is_32bit) {
	    tcg_gen_ext32u_i64 (r, cpu_reg(reg));
	    tcg_gen_shr_i64 (r, r, tcg_shift);
	} else
	  tcg_gen_shr_i64(r, cpu_reg(reg), tcg_shift);
        break;
    case 2: /* ASR */
	/* For arithmetic right shift we sign extend first, then shift,
	   and then need to clear the upper bits again.  */
        if (is_32bit) {
            TCGv_i64 tcg_tmp = tcg_temp_new_i64();
            tcg_gen_ext32s_i64(tcg_tmp, cpu_reg(reg));
            tcg_gen_sar_i64(r, tcg_tmp, tcg_shift);
	    tcg_gen_ext32u_i64 (r, r);
            tcg_temp_free_i64(tcg_tmp);
        } else {
            tcg_gen_sar_i64(r, cpu_reg(reg), tcg_shift);
        }
        break;
    case 3: /* ROR */
	/* For rotation extending doesn't help, we really have to use
	   a 32bit rotate.  */
	if (is_32bit) {
	    TCGv_i32 tmp = tcg_temp_new_i32();
            tcg_gen_trunc_i64_i32(tmp, cpu_reg(reg));
	    tcg_gen_rotr_i32(tmp, tmp, tcg_shift);
            tcg_gen_extu_i32_i64(r, tmp);
            tcg_temp_free_i32(tmp);
	} else
	  tcg_gen_rotr_i64(r, cpu_reg(reg), tcg_shift);
        break;
    }

    return r;
}

static TCGv_i64 get_shifti(int reg, int shift_type, int shift, int is_32bit)
{
    TCGv_i64 tcg_shift;
    TCGv_i64 r;

    if (!shift) {
        r = tcg_temp_new_i64();
	if (is_32bit)
	  tcg_gen_ext32u_i64 (r, cpu_reg(reg));
	else
	  tcg_gen_mov_i64(r, cpu_reg(reg));
        return r;
    }

    tcg_shift = tcg_const_i64(shift);
    r = get_shift(reg, shift_type, tcg_shift, is_32bit);
    tcg_temp_free_i64(tcg_shift);

    return r;
}

static void handle_orr(DisasContext *s, uint32_t insn)
{
    int is_32bit = !get_bits(insn, 31, 1);
    int dest = get_reg(insn);
    int source = get_bits(insn, 5, 5);
    int rm = get_bits(insn, 16, 5);
    int shift_amount = get_sbits(insn, 10, 6);
    int is_n = get_bits(insn, 21, 1);
    int shift_type = get_bits(insn, 22, 2);
    int opc = get_bits(insn, 29, 2);
    bool setflags = (opc == 0x3);
    TCGv_i64 tcg_op2;
    TCGv_i64 tcg_dest;

    if (is_32bit && (shift_amount < 0)) {
        /* reserved value */
        unallocated_encoding(s);
    }

    /* MOV is dest = xzr & (source & ~0) */
    if (!shift_amount && source == 0x1f) {
        if (is_32bit) {
            tcg_gen_ext32u_i64(cpu_reg_sp(dest), cpu_reg(rm));
        } else {
            tcg_gen_mov_i64(cpu_reg_sp(dest), cpu_reg(rm));
        }
        if (is_n) {
            tcg_gen_not_i64(cpu_reg_sp(dest), cpu_reg_sp(dest));
        }
        if (is_32bit) {
            tcg_gen_ext32u_i64(cpu_reg_sp(dest), cpu_reg_sp(dest));
        }
        return;
    }

    tcg_op2 = get_shifti(rm, shift_type, shift_amount & (is_32bit ? 31 : 63),
                         is_32bit);
    if (is_n) {
        tcg_gen_not_i64(tcg_op2, tcg_op2);
    }

    tcg_dest = cpu_reg(dest);
    switch (opc) {
    case 0x3:
        setflags = true;
        /* fall through */
    case 0x0:
        tcg_gen_and_i64(tcg_dest, cpu_reg(source), tcg_op2);
        break;
    case 0x1:
        tcg_gen_or_i64(tcg_dest, cpu_reg(source), tcg_op2);
        break;
    case 0x2:
        tcg_gen_xor_i64(tcg_dest, cpu_reg(source), tcg_op2);
        break;
    }

    if (is_32bit) {
        tcg_gen_ext32u_i64(tcg_dest, tcg_dest);
    }

    if (setflags) {
        gen_helper_pstate_add(pstate, pstate, tcg_dest, cpu_reg(31), tcg_dest);
    }

    tcg_temp_free_i64(tcg_op2);
}

static void setflags_add(bool sub_op, bool is_32bit, TCGv_i64 src,
                         TCGv_i64 op2, TCGv_i64 res)
{
    if (sub_op) {
        if (is_32bit) {
            gen_helper_pstate_sub32(pstate, pstate, src, op2, res);
        } else {
            gen_helper_pstate_sub(pstate, pstate, src, op2, res);
        }
    } else {
        if (is_32bit) {
            gen_helper_pstate_add32(pstate, pstate, src, op2, res);
        } else {
            gen_helper_pstate_add(pstate, pstate, src, op2, res);
        }
    }
}

static void reg_extend(TCGv_i64 tcg_offset, int option, int shift, int reg)
{
    int extsize = get_bits(option, 0, 2);
    bool is_signed = get_bits(option, 2, 1);

    if (is_signed) {
        switch (extsize) {
        case 0:
            tcg_gen_ext8s_i64(tcg_offset, cpu_reg(reg));
            break;
        case 1:
            tcg_gen_ext16s_i64(tcg_offset, cpu_reg(reg));
            break;
        case 2:
            tcg_gen_ext32s_i64(tcg_offset, cpu_reg(reg));
            break;
        case 3:
            tcg_gen_mov_i64(tcg_offset, cpu_reg(reg));
            break;
        }
    } else {
        switch (extsize) {
        case 0:
            tcg_gen_ext8u_i64(tcg_offset, cpu_reg(reg));
            break;
        case 1:
            tcg_gen_ext16u_i64(tcg_offset, cpu_reg(reg));
            break;
        case 2:
            tcg_gen_ext32u_i64(tcg_offset, cpu_reg(reg));
            break;
        case 3:
            tcg_gen_mov_i64(tcg_offset, cpu_reg(reg));
            break;
        }
    }

    if (shift) {
        tcg_gen_shli_i64(tcg_offset, tcg_offset, shift);
    }
}

static void handle_add(DisasContext *s, uint32_t insn)
{
    int dest = get_reg(insn);
    int source = get_bits(insn, 5, 5);
    int shift_amount = get_sbits(insn, 10, 6);
    int rm = get_bits(insn, 16, 5);
    bool extend = get_bits(insn, 21, 1);
    int shift_type = get_bits(insn, 22, 2);
    bool setflags = get_bits(insn, 29, 1);
    bool sub_op = get_bits(insn, 30, 1);
    bool is_32bit = !get_bits(insn, 31, 1);
    bool is_carry = (get_bits(insn, 24, 5) == 0x1a);
    int extend_type = 0;
    TCGv_i64 tcg_op2;
    TCGv_i64 tcg_src = tcg_temp_new_i64();
    TCGv_i64 tcg_dst;
    TCGv_i64 tcg_result = tcg_temp_new_i64();

    if (extend && shift_type) {
        unallocated_encoding(s);
    }

    tcg_gen_mov_i64(tcg_src, cpu_reg(source));
    tcg_dst = cpu_reg(dest);
    if (extend) {
        extend_type = (shift_amount >> 3) & 0x7;
	shift_amount &= 7;
        if (shift_amount > 4) {
            /* reserved value */
            unallocated_encoding(s);
        }
	tcg_gen_mov_i64(tcg_src, cpu_reg_sp(source));
        if (!setflags) {
            tcg_dst = cpu_reg_sp(dest);
        }
    } else {
        if (shift_type == 3) {
            /* reserved value */
            unallocated_encoding(s);
        }
        if (is_32bit && (shift_amount < 0)) {
            /* reserved value */
            unallocated_encoding(s);
        }
    }

    if (extend) {
        tcg_op2 = tcg_temp_new_i64();
        reg_extend(tcg_op2, extend_type, shift_amount, rm);
    } else {
        tcg_op2 = get_shifti(rm, shift_type, shift_amount, is_32bit);
    }

    if (is_32bit) {
        tcg_gen_ext32s_i64(tcg_src, tcg_src);
        tcg_gen_ext32s_i64(tcg_op2, tcg_op2);
    }

    if (sub_op) {
        tcg_gen_sub_i64(tcg_result, tcg_src, tcg_op2);
    } else {
        tcg_gen_add_i64(tcg_result, tcg_src, tcg_op2);
    }

    if (is_carry) {
        TCGv_i64 tcg_carry = tcg_temp_new_i64();
        tcg_gen_shri_i64(tcg_carry, pstate, PSTATE_C_SHIFT);
        tcg_gen_andi_i64(tcg_carry, tcg_carry, 1);
        tcg_gen_add_i64(tcg_result, tcg_result, tcg_carry);
        if (sub_op) {
            tcg_gen_subi_i64(tcg_result, tcg_result, 1);
        }
        tcg_temp_free_i64(tcg_carry);
    }

    if (setflags) {
        setflags_add(sub_op, is_32bit, tcg_src, tcg_op2, tcg_result);
    }

    if (is_32bit) {
        tcg_gen_ext32u_i64(tcg_dst, tcg_result);
    } else {
        tcg_gen_mov_i64(tcg_dst, tcg_result);
    }

    tcg_temp_free_i64(tcg_src);
    tcg_temp_free_i64(tcg_op2);
    tcg_temp_free_i64(tcg_result);
}

static void ldst_do_vec_int(DisasContext *s, int freg_offs, TCGv_i64 tcg_addr,
                            int size, bool is_store)
{
    TCGv_i64 tcg_tmp = tcg_temp_new_i64();

    if (is_store) {
        switch (size) {
        case 0:
            tcg_gen_ld8u_i64(tcg_tmp, cpu_env, freg_offs);
            tcg_gen_qemu_st8(tcg_tmp, tcg_addr, get_mem_index(s));
            break;
        case 1:
            tcg_gen_ld16u_i64(tcg_tmp, cpu_env, freg_offs);
            tcg_gen_qemu_st16(tcg_tmp, tcg_addr, get_mem_index(s));
            break;
        case 2:
            tcg_gen_ld32u_i64(tcg_tmp, cpu_env, freg_offs);
            tcg_gen_qemu_st32(tcg_tmp, tcg_addr, get_mem_index(s));
            break;
        case 4:
            tcg_gen_ld_i64(tcg_tmp, cpu_env, freg_offs);
            tcg_gen_qemu_st64(tcg_tmp, tcg_addr, get_mem_index(s));
            freg_offs += sizeof(uint64_t);
            tcg_gen_addi_i64(tcg_addr, tcg_addr, sizeof(uint64_t));
            /* fall through */
        case 3:
            tcg_gen_ld_i64(tcg_tmp, cpu_env, freg_offs);
            tcg_gen_qemu_st64(tcg_tmp, tcg_addr, get_mem_index(s));
            break;
        }
    } else {
        switch (size) {
        case 0:
            tcg_gen_qemu_ld8u(tcg_tmp, tcg_addr, get_mem_index(s));
            tcg_gen_st8_i64(tcg_tmp, cpu_env, freg_offs);
            break;
        case 1:
            tcg_gen_qemu_ld16u(tcg_tmp, tcg_addr, get_mem_index(s));
            tcg_gen_st16_i64(tcg_tmp, cpu_env, freg_offs);
            break;
        case 2:
            tcg_gen_qemu_ld32u(tcg_tmp, tcg_addr, get_mem_index(s));
            tcg_gen_st32_i64(tcg_tmp, cpu_env, freg_offs);
            break;
        case 4:
            tcg_gen_qemu_ld64(tcg_tmp, tcg_addr, get_mem_index(s));
            tcg_gen_st_i64(tcg_tmp, cpu_env, freg_offs);
            freg_offs += sizeof(uint64_t);
            tcg_gen_addi_i64(tcg_addr, tcg_addr, sizeof(uint64_t));
            /* fall through */
        case 3:
            tcg_gen_qemu_ld64(tcg_tmp, tcg_addr, get_mem_index(s));
            tcg_gen_st_i64(tcg_tmp, cpu_env, freg_offs);
            break;
        }
    }

    tcg_temp_free_i64(tcg_tmp);
}

static void ldst_do_vec(DisasContext *s, int dest, TCGv_i64 tcg_addr_real,
                        int size, bool is_store)
{
    TCGv_i64 tcg_addr = tcg_temp_new_i64();
    int freg_offs = offsetof(CPUARMState, vfp.regs[dest * 2]);

    /* we don't want to modify the caller's tcg_addr */
    tcg_gen_mov_i64(tcg_addr, tcg_addr_real);

    if (!is_store) {
        /* normal ldst clears non-loaded bits */
        clear_fpreg(dest);
    }

    ldst_do_vec_int(s, freg_offs, tcg_addr, size, is_store);

    tcg_temp_free(tcg_addr);
}

static void ldst_do_gpr(DisasContext *s, int dest, TCGv_i64 tcg_addr, int size,
                        bool is_store, bool is_signed)
{
    if (is_store) {
        switch (size) {
        case 0:
            tcg_gen_qemu_st8(cpu_reg(dest), tcg_addr, get_mem_index(s));
            break;
        case 1:
            tcg_gen_qemu_st16(cpu_reg(dest), tcg_addr, get_mem_index(s));
            break;
        case 2:
            tcg_gen_qemu_st32(cpu_reg(dest), tcg_addr, get_mem_index(s));
            break;
        case 3:
            tcg_gen_qemu_st64(cpu_reg(dest), tcg_addr, get_mem_index(s));
            break;
        }
    } else {
        if (is_signed) {
            /* XXX check what impact regsize has */
            switch (size) {
            case 0:
                tcg_gen_qemu_ld8s(cpu_reg(dest), tcg_addr, get_mem_index(s));
                break;
            case 1:
                tcg_gen_qemu_ld16s(cpu_reg(dest), tcg_addr, get_mem_index(s));
                break;
            case 2:
                tcg_gen_qemu_ld32s(cpu_reg(dest), tcg_addr, get_mem_index(s));
                break;
            case 3:
                tcg_gen_qemu_ld64(cpu_reg(dest), tcg_addr, get_mem_index(s));
                break;
            }
        } else {
            switch (size) {
            case 0:
                tcg_gen_qemu_ld8u(cpu_reg(dest), tcg_addr, get_mem_index(s));
                break;
            case 1:
                tcg_gen_qemu_ld16u(cpu_reg(dest), tcg_addr, get_mem_index(s));
                break;
            case 2:
                tcg_gen_qemu_ld32u(cpu_reg(dest), tcg_addr, get_mem_index(s));
                break;
            case 3:
                tcg_gen_qemu_ld64(cpu_reg(dest), tcg_addr, get_mem_index(s));
                break;
            }
        }
    }
}

static void ldst_do(DisasContext *s, int dest, TCGv_i64 tcg_addr, int size,
                    bool is_store, bool is_signed, bool is_vector)
{
    if (is_vector) {
        ldst_do_vec(s, dest, tcg_addr, size, is_store);
    } else {
        ldst_do_gpr(s, dest, tcg_addr, size, is_store, is_signed);
    }
}

static void handle_stp(DisasContext *s, uint32_t insn)
{
    int rt = get_reg(insn);
    int rn = get_bits(insn, 5, 5);
    int rt2 = get_bits(insn, 10, 5);
    int offset = get_sbits(insn, 15, 7);
    int is_store = !get_bits(insn, 22, 1);
    int is_vector = get_bits(insn, 26, 1);
    int is_signed = get_bits(insn, 30, 1);
    int is_32bit = !get_bits(insn, 31, 1);
    int type = get_bits(insn, 23, 2);
    TCGv_i64 tcg_addr;
    bool postindex;
    bool wback;
    int size = is_32bit ? 2 : 3;

    if (is_vector) {
        size = 2 + get_bits(insn, 30, 2);
    }

    switch (type) {
    default:
    case 0:
        postindex = false;
        wback = false;
        break;
    case 1: /* STP (post-index) */
        postindex = true;
        wback = true;
        break;
    case 2: /* STP (signed offset */
        postindex = false;
        wback = false;
        break;
    case 3: /* STP (pre-index) */
        postindex = false;
        wback = true;
        break;
    }

    if (is_signed && !is_32bit) {
        unallocated_encoding(s);
        return;
    }

    offset <<= size;

    tcg_addr = tcg_temp_new_i64();
    if (rn == 31) {
        /* XXX check SP alignment */
    }
    tcg_gen_mov_i64(tcg_addr, cpu_reg_sp(rn));

    if (!postindex) {
        tcg_gen_addi_i64(tcg_addr, tcg_addr, offset);
    }

    ldst_do(s, rt, tcg_addr, size, is_store, is_signed, is_vector);
    tcg_gen_addi_i64(tcg_addr, tcg_addr, 1 << size);
    ldst_do(s, rt2, tcg_addr, size, is_store, is_signed, is_vector);
    tcg_gen_subi_i64(tcg_addr, tcg_addr, 1 << size);

    if (wback) {
        if (postindex) {
            tcg_gen_addi_i64(tcg_addr, tcg_addr, offset);
        }
        tcg_gen_mov_i64(cpu_reg_sp(rn), tcg_addr);
    }

    tcg_temp_free_i64(tcg_addr);
}

/* Load/Store exclusive instructions are implemented by remembering
   the value/address loaded, and seeing if these are the same
   when the store is performed. This should be sufficient to implement
   the architecturally mandated semantics, and avoids having to monitor
   regular stores.

   In system emulation mode only one CPU will be running at once, so
   this sequence is effectively atomic.  In user emulation mode we
   throw an exception and handle the atomic operation elsewhere.  */
static void gen_load_exclusive(DisasContext *s, int rt, int rt2,
                               TCGv_i64 addr, int size, bool is_pair)
{
    TCGv_i64 tmp = tcg_temp_new_i64();

    switch (size) {
    case 0:
	tcg_gen_qemu_ld8u(tmp, addr, get_mem_index(s));
        break;
    case 1:
	tcg_gen_qemu_ld16u(tmp, addr, get_mem_index(s));
        break;
    case 2:
	tcg_gen_qemu_ld32u(tmp, addr, get_mem_index(s));
	break;
    case 3:
	tcg_gen_qemu_ld64(tmp, addr, get_mem_index(s));
        break;
    default:
        abort();
    }
    tcg_gen_mov_i64(cpu_exclusive_val, tmp);
    tcg_gen_mov_i64(cpu_reg(rt), tmp);
    if (is_pair) {
	TCGv_i64 addr2 = tcg_temp_new_i64();
        tcg_gen_addi_i32(addr2, addr, 1 << size);
	if (size == 2)
	  tcg_gen_qemu_ld32u(tmp, addr2, get_mem_index(s));
	else
	  tcg_gen_qemu_ld64(tmp, addr2, get_mem_index(s));
	tcg_temp_free_i64(addr2);
        tcg_gen_mov_i64(cpu_exclusive_high, tmp);
	tcg_gen_mov_i64(cpu_reg(rt2), tmp);
    }

    tcg_temp_free_i64(tmp);
    tcg_gen_mov_i64(cpu_exclusive_addr, addr);
}

#ifdef CONFIG_USER_ONLY
static void gen_store_exclusive(DisasContext *s, int rd, int rt, int rt2,
                                TCGv_i64 addr, int size, int is_pair)
{
    tcg_gen_mov_i64(cpu_exclusive_test, addr);
    tcg_gen_movi_i64(cpu_exclusive_info,
                     size | is_pair << 2 | (rd << 4) | (rt << 9) | (rt2 << 14));
    gen_exception_insn(s, 4, EXCP_STREX);
}
#else
#error implement gen_store_exclusive for system mode (see target-arm/translate.c)
#endif

static void handle_ldarx(DisasContext *s, uint32_t insn)
{
    int rt = get_reg(insn);
    int rn = get_bits(insn, 5, 5);
    int rt2 = get_bits(insn, 10, 5);
    int is_atomic = !get_bits(insn, 15, 1);
    int rs = get_bits(insn, 16, 5);
    int is_pair = get_bits(insn, 21, 1);
    int is_store = !get_bits(insn, 22, 1);
    int is_excl = !get_bits(insn, 23, 1);
    int size = get_bits(insn, 30, 2);
    TCGv_i64 tcg_addr;

    tcg_addr = tcg_temp_new_i64();
    if (rn == 31) {
        /* XXX check SP alignment */
    }
    tcg_gen_mov_i64(tcg_addr, cpu_reg_sp(rn));

    if (is_atomic) {
        /* XXX add locking */
    }

    if (is_excl) {
	if (!is_store) {
	    gen_load_exclusive (s, rt, rt2, tcg_addr, size, is_pair);
	} else {
	    gen_store_exclusive (s, rs, rt, rt2, tcg_addr, size, is_pair);
	}
    } else {
	ldst_do_gpr(s, rt, tcg_addr, size, is_store, false);
	if (is_pair) {
	    tcg_gen_addi_i64(tcg_addr, tcg_addr, 1 << size);
	    ldst_do_gpr(s, rt2, tcg_addr, size, is_store, false);
	}
    }

    tcg_temp_free_i64(tcg_addr);
}

static void ldst_calc_index(DisasContext *s, TCGv_i64 tcg_addr,
                            bool is_reg_offset, int offset, int size)
{
    int option = get_bits(offset, 1, 3);
    bool is_shift = get_bits(offset, 0, 1);
    int shift = size;
    int rn = get_bits(offset, 4, 5);
    TCGv_i64 tcg_offset;

    if (!is_reg_offset) {
        tcg_offset = tcg_const_i64(offset);
        goto add_offset;
    }

    /* offset in register */
    if (!(option & 2)) {
        unallocated_encoding(s);
        return;
    }

    if (!is_shift) {
        shift = 0;
    }

    tcg_offset = tcg_temp_new_i64();
    reg_extend(tcg_offset, option, shift, rn);

add_offset:
    tcg_gen_add_i64(tcg_addr, tcg_addr, tcg_offset);
    tcg_temp_free_i64(tcg_offset);
}

static void handle_literal(DisasContext *s, uint32_t insn)
{
    int dest = get_reg(insn);
    int opc = get_bits(insn, 30, 2);
    int64_t imm = get_sbits(insn, 5, 19) << 2;
    bool is_vector = get_bits(insn, 26, 1);
    TCGv_i64 tcg_addr;
    bool is_signed;
    int size;

    tcg_addr = tcg_const_i64((s->pc - 4) + imm);

    switch (opc) {
    case 0:
        is_signed = false;
        size = 2;
        break;
    case 1:
        is_signed = false;
        size = 3;
        break;
    case 2:
	if (!is_vector) {
	    is_signed = true;
	    size = 2;
	} else
	  size = 4;
        break;
    case 3:
	if (is_vector) {
	    unallocated_encoding(s);
	    return;
	}
        /* prefetch */
        goto out;
    }

    ldst_do(s, dest, tcg_addr, size, false, is_signed, is_vector);

out:
    tcg_temp_free_i64(tcg_addr);
}

static void handle_ldst(DisasContext *s, uint32_t insn)
{
    int dest = get_reg(insn);
    int source = get_bits(insn, 5, 5);
    int offset;
    TCGv_i64 tcg_addr;
    bool is_store = !get_bits(insn, 22, 1);
    bool opc1 = get_bits(insn, 23, 1);
    bool is_signed = false;
    int size = get_bits(insn, 30, 2);
    int regsize = size;
    bool postindex = false;
    bool wback = get_bits(insn, 10, 1);
    int type = get_bits(insn, 10, 2);
    bool is_reg_offset = get_bits(insn, 21, 1);
    bool is_imm12 = get_bits(insn, 24, 1);
    bool is_vector = get_bits(insn, 26, 1);

    if (is_imm12) {
        /* wback, postindex and reg_offset bits are inside imm12 */
        postindex = false;
        wback = false;
        is_reg_offset = false;
    } else {
        /* These only apply to the IMM9 variant */
        if (is_reg_offset && type != 2) {
            unallocated_encoding(s);
            return;
        }
        /* LDR (post-index) */
        postindex = (type == 1);
    }

    if (is_vector) {
        size = (opc1 ? 0x4 : 0) | size;
        if (size > 4) {
            unallocated_encoding(s);
            return;
        }
    } else if (!opc1) {
        regsize = (size == 3) ? 3 : 2;
    } else {
        if (size == 3) {
            /* prefetch */
            if (!is_store) {
                unallocated_encoding(s);
            }
            return;
        }
        if (size == 2 && !is_store) {
            unallocated_encoding(s);
        }
        is_store = false;
        is_signed = true;
        regsize = is_store ? 3 : 2;
    }

    if (is_imm12) {
        /* UIMM12 version */
        offset = get_bits(insn, 10, 12) << size;
    } else {
        /* SIMM9 version */
        offset = get_sbits(insn, 12, 9);
    }

    tcg_addr = tcg_temp_new_i64();

    tcg_gen_mov_i64(tcg_addr, cpu_reg_sp(source));

    if (!postindex) {
        ldst_calc_index(s, tcg_addr, is_reg_offset, offset, size);
    }

    ldst_do(s, dest, tcg_addr, size, is_store, is_signed, is_vector);

    if (postindex) {
        ldst_calc_index(s, tcg_addr, is_reg_offset, offset, size);
    }

    if (wback) {
        tcg_gen_mov_i64(cpu_reg_sp(source), tcg_addr);
    }

    tcg_temp_free_i64(tcg_addr);
}

static void handle_shift(DisasContext *s, int rd, int rn, int rm,
			 int shift_type, bool is_32bit)
{
    TCGv_i64 tcg_shift;
    TCGv_i64 tcg_shifted;

    tcg_shift = tcg_temp_new_i64();
    tcg_gen_andi_i64(tcg_shift, cpu_reg(rm), is_32bit ? 31 : 63);
    tcg_shifted = get_shift(rn, shift_type, tcg_shift, is_32bit);
    tcg_gen_mov_i64(cpu_reg(rd), tcg_shifted);
    tcg_temp_free_i64(tcg_shift);
    tcg_temp_free_i64(tcg_shifted);
}

static void handle_mulh(DisasContext *s, uint32_t insn)
{
    int rd = get_reg(insn);
    int rn = get_bits(insn, 5, 5);
    int rm = get_bits(insn, 16, 5);
    bool is_signed = !get_bits(insn, 23, 1);

    if (is_signed) {
        gen_helper_smulh(cpu_reg(rd), cpu_reg(rn), cpu_reg(rm));
    } else {
        gen_helper_umulh(cpu_reg(rd), cpu_reg(rn), cpu_reg(rm));
    }
}

static void handle_udiv(DisasContext *s, int rd, int rn, int rm,
			bool is_signed, bool is_32bit)
{
    TCGv_i64 n = tcg_temp_new_i64();
    TCGv_i64 m = tcg_temp_new_i64();

    if (is_32bit) {
        if (is_signed) {
            tcg_gen_ext32s_i64(n, cpu_reg(rn));
            tcg_gen_ext32s_i64(m, cpu_reg(rm));
        } else {
            tcg_gen_ext32u_i64(n, cpu_reg(rn));
            tcg_gen_ext32u_i64(m, cpu_reg(rm));
        }
    } else {
        tcg_gen_mov_i64(n, cpu_reg(rn));
        tcg_gen_mov_i64(m, cpu_reg(rm));
    }

    if (is_signed) {
        gen_helper_sdiv64(cpu_reg(rd), n, m);
    } else {
        gen_helper_udiv64(cpu_reg(rd), n, m);
    }

    if (is_32bit) {
        if (is_signed) {
            tcg_gen_ext32s_i64(cpu_reg(rd), cpu_reg(rd));
        } else {
            tcg_gen_ext32u_i64(cpu_reg(rd), cpu_reg(rd));
        }
    }

    tcg_temp_free_i64(n);
    tcg_temp_free_i64(m);
}

static void handle_extr(DisasContext *s, uint32_t insn)
{
    int rd = get_reg(insn);
    int rn = get_bits(insn, 5, 5);
    int imms = get_bits(insn, 10, 6);
    int rm = get_bits(insn, 16, 5);
    bool is_32bit = !get_bits(insn, 31, 1);
    int bitsize = is_32bit ? 32 : 64;
    TCGv_i64 tcg_res = tcg_temp_new_i64();
    TCGv_i64 tcg_tmp = tcg_temp_new_i64();

    if (is_32bit) {
        tcg_gen_ext32u_i64(tcg_tmp, cpu_reg(rm));
    } else {
        tcg_gen_mov_i64(tcg_tmp, cpu_reg(rm));
    }
    tcg_gen_shri_i64(tcg_res, tcg_tmp, imms);
    tcg_gen_shli_i64(tcg_tmp, cpu_reg(rn), bitsize - imms);
    tcg_gen_or_i64(cpu_reg(rd), tcg_tmp, tcg_res);
    if (is_32bit) {
        tcg_gen_ext32u_i64(cpu_reg(rd), cpu_reg(rd));
    }

    tcg_temp_free_i64(tcg_tmp);
    tcg_temp_free_i64(tcg_res);
}

/* Verified against test case -agraf */
static void handle_bfm(DisasContext *s, uint32_t insn)
{
    bool is_32bit = !get_bits(insn, 31, 1);
    int opc = get_bits(insn, 29, 2);
    int dest = get_reg(insn);
    int source = get_bits(insn, 5, 5);
    int is_n = get_bits(insn, 22, 1);
    int imms = get_bits(insn, 10, 6);
    int immr = get_bits(insn, 16, 6);
    TCGv_i64 tcg_newmask;
    uint64_t mask, tmask, topmask;
    uint64_t signbit = 1;
    int bitsize = is_32bit ? 32 : 64;

    if (!is_32bit && !is_n) {
        /* reserved */
        unallocated_encoding(s);
    }

    if (is_32bit && (is_n || (immr & (1 << 5)) || imms & (1 << 5))) {
        /* reserved */
        unallocated_encoding(s);
    }

    tcg_newmask = tcg_temp_new_i64();

    if (imms == 0x3f) {
        tmask = mask = ~0ULL;
    } else {
        tmask = mask = ((1ULL << (imms + 1)) - 1);
    }

    tcg_gen_andi_i64(tcg_newmask, cpu_reg(source), mask);
    if (imms < immr) {
        tcg_gen_shli_i64(tcg_newmask, tcg_newmask, bitsize - immr);
        tmask <<= bitsize - immr;
        signbit <<= bitsize + imms - immr;
        if (signbit == 0x8000000000000000ULL) {
            /* Can't pad anymore - highest bit is already set */
            topmask = 0;
        } else {
            topmask = ~((1ULL << (bitsize + imms - immr + 1)) - 1);
        }
    } else {
        tcg_gen_shri_i64(tcg_newmask, tcg_newmask, immr);
        tmask >>= immr;
        signbit <<= imms - immr;
        topmask = ~tmask;
    }

    if (is_32bit) {
        tcg_gen_ext32u_i64(tcg_newmask, tcg_newmask);
    }

    switch (opc) {
    case 0: { /* SBFM */
        TCGv_i64 tcg_mask = tcg_const_i64(topmask);
        TCGv_i64 tcg_signbit = tcg_const_i64(signbit);
        gen_helper_sign_extend(cpu_reg(dest), tcg_newmask, tcg_signbit,
                               tcg_mask);
        tcg_temp_free_i64(tcg_mask);
        tcg_temp_free_i64(tcg_signbit);
        break;
    }
    case 1: /* BFM */
        /* replace the field inside dest */
        tcg_gen_andi_i64(cpu_reg(dest), cpu_reg(dest), ~tmask);
        tcg_gen_or_i64(cpu_reg(dest), cpu_reg(dest), tcg_newmask);
        break;
    case 2: /* UBFM */
        tcg_gen_mov_i64(cpu_reg(dest), tcg_newmask);
        break;
    default:
        unallocated_encoding(s);
        return;
    }

    if (is_32bit) {
        tcg_gen_ext32u_i64(cpu_reg(dest), cpu_reg(dest));
    }

    tcg_temp_free_i64(tcg_newmask);
}

static void handle_addi(DisasContext *s, uint32_t insn)
{
    TCGv_i64 tcg_result = tcg_temp_new_i64();
    TCGv_i64 tcg_imm;
    int dest = get_reg(insn);
    int source = get_bits(insn, 5, 5);
    int shift = get_bits(insn, 22, 2);
    uint64_t imm = get_bits(insn, 10, 12);
    bool is_32bit = !get_bits(insn, 31, 1);
    bool sub_op = get_bits(insn, 30, 1);
    bool setflags = get_bits(insn, 29, 1);

    switch (shift) {
    case 0x0:
        break;
    case 0x1:
        imm <<= 12;
        break;
    default:
        unallocated_encoding(s);
    }

    tcg_imm = tcg_const_i64(imm);

    if (sub_op) {
        tcg_gen_subi_i64(tcg_result, cpu_reg_sp(source), imm);
    } else {
        tcg_gen_addi_i64(tcg_result, cpu_reg_sp(source), imm);
    }

    if (setflags) {
        setflags_add(sub_op, is_32bit, cpu_reg_sp(source), tcg_imm, tcg_result);
        if (is_32bit) {
            tcg_gen_ext32u_i64(cpu_reg(dest), tcg_result);
        } else {
            tcg_gen_mov_i64(cpu_reg(dest), tcg_result);
        }
    } else {
        if (is_32bit) {
            tcg_gen_ext32u_i64(cpu_reg_sp(dest), tcg_result);
        } else {
            tcg_gen_mov_i64(cpu_reg_sp(dest), tcg_result);
        }
    }

}

static void handle_svc(DisasContext *s, uint32_t insn)
{
    gen_a64_set_pc_im(s->pc);
#define DISAS_SWI 5
    s->is_jmp = DISAS_SWI;
}

static void handle_ccmp(DisasContext *s, uint32_t insn)
{
    int rn = get_bits(insn, 5, 5);
    int rm = get_bits(insn, 16, 5);
    TCGv_i32 tcg_insn = tcg_const_i32(insn);

    if (get_bits(insn, 4, 1) || get_bits(insn, 10, 1) || !get_bits(insn, 29, 1)) {
	unallocated_encoding(s);
	return;
    }

    if (get_bits(insn, 11, 1)) {
	TCGv_i64 imm = tcg_const_i64(rm);
	gen_helper_ccmp(pstate, pstate, tcg_insn, cpu_reg(rn), imm);
	tcg_temp_free_i64(imm);
    } else {
	gen_helper_ccmp(pstate, pstate, tcg_insn, cpu_reg(rn), cpu_reg(rm));
    }

    tcg_temp_free_i32(tcg_insn);
}

/* Conditional select, CSEL, CS{INC,INV,NEG} */
static void handle_csel(DisasContext *s, uint32_t insn)
{
    int rd = get_reg(insn);
    int rn = get_bits(insn, 5, 5);
    int rm = get_bits(insn, 16, 5);
    TCGv_i32 tcg_insn = tcg_const_i32(insn);

    if (get_bits(insn, 29, 1) || get_bits(insn, 11, 1)) {
	unallocated_encoding(s);
	return;
    }

    gen_helper_csel(cpu_reg(rd), pstate, tcg_insn, cpu_reg(rn), cpu_reg(rm));

    tcg_temp_free_i32(tcg_insn);
}

/* Data processing single source */
static void handle_dp1s(DisasContext *s, uint32_t insn)
{
    int rd = get_reg(insn);
    int rn = get_bits(insn, 5, 5);
    int opcode = get_bits(insn, 10, 6);
    bool is_32bit = !get_bits(insn, 31, 1);

    if (get_bits(insn, 29, 1) || get_bits(insn, 16, 5) || opcode > 5) {
	unallocated_encoding(s);
	return;
    }

    switch (opcode) {
    case 0x0: /* RBIT */
        if (is_32bit) {
            TCGv_i64 tcg_tmp = tcg_temp_new_i32();
            tcg_gen_trunc_i64_i32(tcg_tmp, cpu_reg(rn));
            gen_helper_rbit(tcg_tmp, tcg_tmp);
            tcg_gen_extu_i32_i64(cpu_reg(rd), tcg_tmp);
            tcg_temp_free_i32(tcg_tmp);
        } else {
            gen_helper_rbit64(cpu_reg(rd), cpu_reg(rn));
        }
        break;
    case 0x1: /* REV16 */
        tcg_gen_bswap16_i64(cpu_reg(rd), cpu_reg(rn));
        break;
    case 0x2: /* REV32 */
        tcg_gen_bswap32_i64(cpu_reg(rd), cpu_reg(rn));
        break;
    case 0x3: /* REV64 */
	if (is_32bit) {
	    unallocated_encoding(s);
	    return;
	}
        tcg_gen_bswap64_i64(cpu_reg(rd), cpu_reg(rn));
        break;
    case 0x4: /* CLZ */
    case 0x5: /* CLS */
	{
	  TCGv_i64 tcg_val = tcg_temp_new_i64();
	  int reduce = 0;

	  if (is_32bit) {
	      if (opcode == 0x4)
		tcg_gen_ext32u_i64(tcg_val, cpu_reg(rn));
	      else
		tcg_gen_ext32s_i64(tcg_val, cpu_reg(rn));
	  } else {
	      tcg_gen_mov_i64(tcg_val, cpu_reg(rn));
	  }

	  if (opcode == 0x4) {
	      gen_helper_clz64(cpu_reg(rd), tcg_val);
	      if (is_32bit) {
		  reduce = 32;
	      }
	  } else {
	      /* cls == clz(V ^ V>>1)-1, V being signed */
	      tcg_gen_sari_i64 (cpu_reg(rd), tcg_val, 1);
	      tcg_gen_xor_i64 (tcg_val, cpu_reg(rd), tcg_val);
	      gen_helper_clz64 (cpu_reg(rd), tcg_val);
	      if (is_32bit)
		reduce = 33;
	      else
		reduce = 1;
	  }
	  tcg_temp_free_i64(tcg_val);

	  if (reduce)
	    tcg_gen_subi_i64(cpu_reg(rd), cpu_reg(rd), reduce);
	}
    }

    if (is_32bit) {
        tcg_gen_ext32u_i64(cpu_reg(rd), cpu_reg(rd));
    }
}

/* Data processing, two source */
static void handle_dp2s(DisasContext *s, uint32_t insn)
{
    int rd = get_reg(insn);
    int rn = get_bits(insn, 5, 5);
    int rm = get_bits(insn, 16, 5);
    int opcode = get_bits(insn, 10, 6);
    bool is_32bit = !get_bits(insn, 31, 1);

    switch (opcode) {
    case 2:  /* UDIV */
    case 3:  /* SDIV */
	handle_udiv(s, rd, rn, rm, get_bits(insn, 10, 1), is_32bit);
	break;
    case 8:  /* LSLV */
    case 9:  /* LSRV */
    case 10: /* ASRV */
    case 11: /* RORV */
	handle_shift(s, rd, rn, rm, get_bits(insn, 10, 2), is_32bit);
	break;
    default:
	unallocated_encoding(s);
	return;
    }
}

/* Data-processing (3 source) */
static void handle_dp3s(DisasContext *s, uint32_t insn)
{
    int rd = get_reg(insn);
    int rn = get_bits(insn, 5, 5);
    int ra = get_bits(insn, 10, 5);
    int rm = get_bits(insn, 16, 5);
    int op_id = (get_bits(insn, 29, 3) << 4) |
                (get_bits(insn, 21, 3) << 1) |
                get_bits(insn, 15, 1);
    bool is_32bit = !(op_id & 0x40);
    bool is_sub = op_id & 0x1;
    bool is_signed = (op_id >= 0x42) && (op_id <= 0x44);
    bool is_high = op_id & 0x4;
    TCGv_i64 tcg_op1;
    TCGv_i64 tcg_op2;
    TCGv_i64 tcg_tmp;

    switch (op_id) {
    case 0x0: /* MADD (32bit) */
    case 0x1: /* MSUB (32bit) */
    case 0x40: /* MADD (64bit) */
    case 0x41: /* MSUB (64bit) */
    case 0x42: /* SMADDL */
    case 0x43: /* SMSUBL */
    case 0x44: /* SMULH */
    case 0x4a: /* UMADDL */
    case 0x4b: /* UMSUBL */
    case 0x4c: /* UMULH */
        break;
    default:
        unallocated_encoding(s);
    }

    if (is_high) {
        handle_mulh(s, insn);
        return;
    }

    tcg_op1 = tcg_temp_new_i64();
    tcg_op2 = tcg_temp_new_i64();
    tcg_tmp = tcg_temp_new_i64();

    if (op_id < 0x42) {
        tcg_gen_mov_i64(tcg_op1, cpu_reg(rn));
        tcg_gen_mov_i64(tcg_op2, cpu_reg(rm));
    } else {
        if (is_signed) {
            tcg_gen_ext32s_i64(tcg_op1, cpu_reg(rn));
            tcg_gen_ext32s_i64(tcg_op2, cpu_reg(rm));
        } else {
            tcg_gen_ext32u_i64(tcg_op1, cpu_reg(rn));
            tcg_gen_ext32u_i64(tcg_op2, cpu_reg(rm));
        }
    }

    tcg_gen_mul_i64(tcg_tmp, tcg_op1, tcg_op2);

    if (is_sub) {
        tcg_gen_sub_i64(cpu_reg(rd), cpu_reg(ra), tcg_tmp);
    } else {
        tcg_gen_add_i64(cpu_reg(rd), cpu_reg(ra), tcg_tmp);
    }

    if (is_32bit) {
        tcg_gen_ext32u_i64(cpu_reg(rd), cpu_reg(rd));
    }

    tcg_temp_free_i64(tcg_op1);
    tcg_temp_free_i64(tcg_op2);
    tcg_temp_free_i64(tcg_tmp);
}

static void handle_fpfpcvt(DisasContext *s, uint32_t insn, bool direction,
                           int rmode)
{
    int rd = get_reg(insn);
    int rn = get_bits(insn, 5, 5);
    int scale = get_bits(insn, 10, 6);
    int opcode = get_bits(insn, 16, 3);
    int type = get_bits(insn, 22, 2);
    bool is_32bit = !get_bits(insn, 31, 1);
    bool is_double = get_bits(type, 0, 1);
    bool is_signed = !get_bits(opcode, 0, 1);
    int freg_offs;
    int fp_reg;
    TCGv_i64 tcg_int;
    TCGv_i64 tcg_single;
    TCGv_i64 tcg_double;
    TCGv_i64 tcg_fpstatus = get_fpstatus_ptr();
    TCGv_i32 tcg_shift = tcg_const_i32(scale);
    TCGv_i32 tcg_rmode = tcg_const_i32(rmode);
    TCGv_i64 tcg_tmp;

    if (direction) {
        fp_reg = rn;
        tcg_int = cpu_reg(rd);
    } else {
        fp_reg = rd;
        tcg_int = cpu_reg(rn);
    }
    freg_offs = offsetof(CPUARMState, vfp.regs[fp_reg * 2]);

    if (!direction) {
        clear_fpreg(fp_reg);
    }

    if (is_32bit && !direction) {
        tcg_tmp = tcg_temp_new_i64();
        if (is_signed) {
            tcg_gen_ext32s_i64(tcg_tmp, tcg_int);
        } else {
            tcg_gen_ext32u_i64(tcg_tmp, tcg_int);
        }
        tcg_int = tcg_tmp;
    }

    gen_helper_set_rmode(tcg_rmode, tcg_fpstatus);

    switch ((direction ? 0x10 : 0)|
            (is_double ? 0x1 : 0) |
            (is_signed ? 0x2 : 0)) {
    case 0x0: /* unsigned scalar->single */
        tcg_single = tcg_temp_new_i32();
        tcg_tmp = tcg_temp_new_i64();
        gen_helper_vfp_uqtos(tcg_single, tcg_int, tcg_shift, tcg_fpstatus);
        tcg_gen_extu_i32_i64(tcg_tmp, tcg_single);
        tcg_gen_st32_i64(tcg_tmp, cpu_env, freg_offs);
        tcg_temp_free_i32(tcg_single);
        tcg_temp_free_i64(tcg_tmp);
        break;
    case 0x1: /* unsigned scalar->double */
        tcg_double = tcg_temp_new_i64();
        gen_helper_vfp_uqtod(tcg_double, tcg_int, tcg_shift, tcg_fpstatus);
        tcg_gen_st_i64(tcg_double, cpu_env, freg_offs);
        tcg_temp_free_i64(tcg_double);
        break;
    case 0x2: /* signed scalar->single */
        tcg_single = tcg_temp_new_i32();
        tcg_tmp = tcg_temp_new_i64();
        gen_helper_vfp_sqtos(tcg_single, tcg_int, tcg_shift, tcg_fpstatus);
        tcg_gen_extu_i32_i64(tcg_tmp, tcg_single);
        tcg_gen_st32_i64(tcg_tmp, cpu_env, freg_offs);
        tcg_temp_free_i32(tcg_single);
        tcg_temp_free_i64(tcg_tmp);
        break;
    case 0x3: /* signed scalar->double */
        tcg_double = tcg_temp_new_i64();
        gen_helper_vfp_sqtod(tcg_double, tcg_int, tcg_shift, tcg_fpstatus);
        tcg_gen_st_i64(tcg_double, cpu_env, freg_offs);
        tcg_temp_free_i64(tcg_double);
        break;
    case 0x10: /* unsigned single->scalar */
        tcg_single = tcg_temp_new_i32();
        tcg_tmp = tcg_temp_new_i64();
        tcg_gen_ld32u_i64(tcg_tmp, cpu_env, freg_offs);
        tcg_gen_trunc_i64_i32(tcg_single, tcg_tmp);
        gen_helper_vfp_touqs(tcg_int, tcg_single, tcg_shift, tcg_fpstatus);
        tcg_temp_free_i32(tcg_single);
        tcg_temp_free_i64(tcg_tmp);
        break;
    case 0x11: /* unsigned single->double */
        tcg_double = tcg_temp_new_i64();
        tcg_gen_ld_i64(tcg_double, cpu_env, freg_offs);
        gen_helper_vfp_touqd(tcg_int, tcg_double, tcg_shift, tcg_fpstatus);
        tcg_temp_free_i64(tcg_double);
        break;
    case 0x12: /* signed single->scalar */
        tcg_single = tcg_temp_new_i32();
        tcg_tmp = tcg_temp_new_i64();
        tcg_gen_ld32u_i64(tcg_tmp, cpu_env, freg_offs);
        tcg_gen_trunc_i64_i32(tcg_single, tcg_tmp);
        gen_helper_vfp_tosqs(tcg_int, tcg_single, tcg_shift, tcg_fpstatus);
        tcg_temp_free_i32(tcg_single);
        tcg_temp_free_i64(tcg_tmp);
        break;
    case 0x13: /* signed single->double */
        tcg_double = tcg_temp_new_i64();
        tcg_gen_ld_i64(tcg_double, cpu_env, freg_offs);
        gen_helper_vfp_tosqd(tcg_int, tcg_double, tcg_shift, tcg_fpstatus);
        tcg_temp_free_i64(tcg_double);
        break;
    default:
        unallocated_encoding(s);
    }

    /* XXX use fpcr */
    tcg_gen_movi_i32(tcg_rmode, -1);
    gen_helper_set_rmode(tcg_rmode, tcg_fpstatus);

    if (is_32bit && direction) {
        tcg_gen_ext32u_i64(tcg_int, tcg_int);
    }

    tcg_temp_free_i64(tcg_fpstatus);
    tcg_temp_free_i32(tcg_shift);
    tcg_temp_free_i32(tcg_rmode);
}

/* fixed <-> floating conversion */
static void handle_fpfpconv(DisasContext *s, uint32_t insn)
{
    bool is_s = get_bits(insn, 29, 1);
    int type = get_bits(insn, 22, 2);
    int opcode = get_bits(insn, 16, 3);
    int rmode = get_bits(insn, 19, 2);
    bool direction;

    if (is_s || (type > 1) || (opcode > 3)) {
        unallocated_encoding(s);
        return;
    }

    switch (rmode) {
    case 0x0: /* [S|U]CVTF (scalar->float) */
	if (opcode < 2) {
	    unallocated_encoding(s);
	    return;
	}
        direction = false;
        break;
    case 0x3: /* FCVTZ[S|U] (float->scalar) */
	if (opcode > 1) {
	    unallocated_encoding(s);
	    return;
	}
        direction = true;
        break;
    default:
        unallocated_encoding(s);
        return;
    }

    handle_fpfpcvt(s, insn, direction, ROUND_MODE_ZERO);
}

/* fmov (immediate) */
static void handle_fmovi(DisasContext *s, uint32_t insn)
{
    int rd = get_reg(insn);
    int is_double = get_bits(insn, 22, 2);
    int imm8 = get_bits(insn, 13, 8);
    uint64_t imm;
    int freg_offs_d = offsetof(CPUARMState, vfp.regs[rd * 2]);
    TCGv_i64 tcg_res;

    if (is_double > 1) {
        unallocated_encoding(s);
        return;
    }

    if (is_double) {
        imm = (get_bits(imm8, 7, 1) ? 0x8000 : 0) |
              (get_bits(imm8, 6, 1) ? 0x3fc0 : 0x4000) |
              get_bits(imm8, 0, 6);
        imm <<= 48;
    } else {
        imm = (get_bits(imm8, 7, 1) ? 0x8000 : 0) |
              (get_bits(imm8, 6, 1) ? 0x3e00 : 0x4000) |
              (get_bits(imm8, 0, 6) << 3);
        imm <<= 16;
    }

    tcg_res = tcg_const_i64(imm);
    clear_fpreg(rd);
    tcg_gen_st_i64(tcg_res, cpu_env, freg_offs_d);
    tcg_temp_free_i64(tcg_res);
}

/* floating <-> integer conversion */
static void handle_fpintconv(DisasContext *s, uint32_t insn)
{
    int rd = get_reg(insn);
    int rn = get_bits(insn, 5, 5);
    int opcode = get_bits(insn, 16, 3);
    int rmode = get_bits(insn, 19, 2);
    int type = get_bits(insn, 22, 2);
    bool is_s = get_bits(insn, 29, 1);
    bool is_32bit = !get_bits(insn, 31, 1);

    if (!is_s && (rmode < 2) && (opcode > 5)) {
        /* FMOV */
        bool itof = opcode & 1;
        int dest = itof ? rd : rn;
        int freg_offs = offsetof(CPUARMState, vfp.regs[dest * 2]);

        if (rmode & 1) {
            freg_offs += sizeof(float64);
        }

        if (itof && (!(rmode & 1))) {
            clear_fpreg(dest);
        }

        switch (type |
                ((rmode & 1) ? 0x4 : 0) |
                (itof ? 0x8 : 0)) {
        case 0x0:
            tcg_gen_ld32u_i64(cpu_reg(rd), cpu_env, freg_offs);
            break;
        case 0x1:
        case 0x2 | 0x4:
            tcg_gen_ld_i64(cpu_reg(rd), cpu_env, freg_offs);
            break;
        case 0x8 | 0x0:
            tcg_gen_st32_i64(cpu_reg(rn), cpu_env, freg_offs);
            break;
        case 0x8 | 0x1:
        case 0x8 | 0x2 | 0x4:
            tcg_gen_st_i64(cpu_reg(rn), cpu_env, freg_offs);
            break;
        default:
            unallocated_encoding(s);
        }

        if (is_32bit && !itof) {
            tcg_gen_ext32u_i64(cpu_reg(rd), cpu_reg(rd));
        }
    } else if (!is_s && ((opcode & 0x6) < 5)) {
        /* [S|U]CVTF and FCVT[N|P|M|Z][S|U] */
        handle_fpfpcvt(s, insn, !(opcode & 0x6), rmode);
    } else {
        /* XXX */
        unallocated_encoding(s);
    }
}

/* floating point compare */
static void handle_fcmp(DisasContext *s, uint32_t insn)
{
    int opc = get_bits(insn, 3, 2);
    int rn = get_bits(insn, 5, 5);
    int rm = get_bits(insn, 16, 5);
    bool is_32bit = !get_bits(insn, 22, 1);
    bool is_cmp_with_zero = get_bits(opc, 0, 1);
    int freg_offs_n = offsetof(CPUARMState, vfp.regs[rn * 2]);
    int freg_offs_m = offsetof(CPUARMState, vfp.regs[rm * 2]);
    TCGv_i32 tcg_op1_32 = tcg_temp_new_i32();
    TCGv_i32 tcg_op1_64 = tcg_temp_new_i64();
    TCGv_i32 tcg_op2_32;
    TCGv_i32 tcg_op2_64;

    if (get_bits(insn, 23, 1)) {
        unallocated_encoding(s);
    }

    tcg_gen_ld_i64(tcg_op1_64, cpu_env, freg_offs_n);

    if (is_32bit) {
        tcg_gen_trunc_i64_i32(tcg_op1_32, tcg_op1_64);
    }

    if (is_cmp_with_zero) {
        tcg_op2_32 = tcg_const_i32(0);
        tcg_op2_64 = tcg_const_i64(0);
    } else {
        tcg_op2_32 = tcg_temp_new_i32();
        tcg_op2_64 = tcg_temp_new_i64();

        tcg_gen_ld_i64(tcg_op2_64, cpu_env, freg_offs_m);
        if (is_32bit) {
            tcg_gen_trunc_i64_i32(tcg_op2_32, tcg_op2_64);
        }
    }

    switch (opc | (is_32bit ? 0x0 : 0x4)) {
    case 0x0: /* FCMP single, 32bit */
    case 0x1: /* FCMPZ single, 32bit */
        gen_helper_vfp_cmps(pstate, tcg_op1_32, tcg_op2_32, cpu_env);
        break;
    case 0x2: /* FCMPE single, 32bit */
    case 0x3: /* FCMPEZ single, 32bit */
        gen_helper_vfp_cmpes(pstate, tcg_op1_32, tcg_op2_32, cpu_env);
        break;
    case 0x4: /* FCMP double, 64bit */
    case 0x5: /* FCMPZ double, 64bit */
        gen_helper_vfp_cmpd(pstate, tcg_op1_64, tcg_op2_64, cpu_env);
        break;
    case 0x6: /* FCMPE double, 64bit */
    case 0x7: /* FCMPEZ double, 64bit */
        gen_helper_vfp_cmped(pstate, tcg_op1_64, tcg_op2_64, cpu_env);
        break;
    }

    tcg_temp_free_i64(tcg_op1_64);
    tcg_temp_free_i64(tcg_op2_64);
    tcg_temp_free_i32(tcg_op1_32);
    tcg_temp_free_i32(tcg_op2_32);
}

/* floating point data processing (1 source) 64-bit */
static void handle_fpdp1s64(DisasContext *s, uint32_t insn)
{
    int rd = get_bits(insn, 0, 5);
    int rn = get_bits(insn, 5, 5);
    int opcode = get_bits(insn, 15, 6);
    int freg_offs_n = offsetof(CPUARMState, vfp.regs[rn * 2]);
    int freg_offs_d = offsetof(CPUARMState, vfp.regs[rd * 2]);
    TCGv_i64 tcg_op = tcg_temp_new_i64();
    TCGv_i64 tcg_res = tcg_temp_new_i64();
    TCGv_ptr fpst = get_fpstatus_ptr();

    tcg_gen_ld_i64(tcg_op, cpu_env, freg_offs_n);

    switch (opcode) {
    case 0x0: /* FMOV */
        tcg_gen_mov_i64(tcg_res, tcg_op);
        break;
    case 0x1: /* FABS */
        gen_helper_vfp_absd(tcg_res, tcg_op);
        break;
    case 0x2: /* FNEG */
        gen_helper_vfp_negd(tcg_res, tcg_op);
        break;
    case 0x3: /* FSQRT */
        gen_helper_vfp_sqrtd(tcg_res, tcg_op, cpu_env);
        break;
    case 0x4: /* FCVT (double to single) */
        gen_helper_vfp_fcvtsd(tcg_res, tcg_op, cpu_env);
        break;
#if 0 /* XXX */
    case 0x7: /* FCVT (double to half) */
        break;
#endif
    case 0x8: /* FRINTN XXX add rounding mode */
    case 0x9: /* FRINTP */
    case 0xa: /* FRINTM */
    case 0xb: /* FRINTZ */
    case 0xc: /* FRINTA */
    case 0xe: /* FRINTX */
    case 0xf: /* FRINTI */
    {
        TCGv_i32 tcg_rmode = tcg_const_i32(opcode & 7);

        gen_helper_set_rmode(tcg_rmode, fpst);
        gen_helper_rintd(tcg_res, tcg_op, fpst);

        /* XXX use fpcr */
        tcg_gen_movi_i32(tcg_rmode, -1);
        gen_helper_set_rmode(tcg_rmode, fpst);
        tcg_temp_free_i32(tcg_rmode);
        break;
    }
    default:
        unallocated_encoding(s);
        return;
    }

    clear_fpreg(rd);
    tcg_gen_st_i64(tcg_res, cpu_env, freg_offs_d);

    tcg_temp_free_ptr(fpst);
    tcg_temp_free_i64(tcg_op);
    tcg_temp_free_i64(tcg_res);
}

/* floating point data processing (1 source) 32-bit */
static void handle_fpdp1s32(DisasContext *s, uint32_t insn)
{
    int rd = get_bits(insn, 0, 5);
    int rn = get_bits(insn, 5, 5);
    int opcode = get_bits(insn, 15, 6);
    int freg_offs_n = offsetof(CPUARMState, vfp.regs[rn * 2]);
    int freg_offs_d = offsetof(CPUARMState, vfp.regs[rd * 2]);
    TCGv_i64 tcg_tmp = tcg_temp_new_i64();
    TCGv_i32 tcg_op = tcg_temp_new_i32();
    TCGv_i32 tcg_res = tcg_temp_new_i32();
    TCGv_ptr fpst = get_fpstatus_ptr();
    bool skip_write = false;

    tcg_gen_ld_i64(tcg_tmp, cpu_env, freg_offs_n);
    tcg_gen_trunc_i64_i32(tcg_op, tcg_tmp);

    switch (opcode) {
    case 0x0: /* FMOV */
        tcg_gen_mov_i32(tcg_res, tcg_op);
        break;
    case 0x1: /* FABS */
        gen_helper_vfp_abss(tcg_res, tcg_op);
        break;
    case 0x2: /* FNEG */
        gen_helper_vfp_negs(tcg_res, tcg_op);
        break;
    case 0x3: /* FSQRT */
        gen_helper_vfp_sqrts(tcg_res, tcg_op, cpu_env);
        break;
    case 0x5: /* FCVT (single to double) */
        skip_write = true;
        gen_helper_vfp_fcvtds(tcg_tmp, tcg_op, cpu_env);
        clear_fpreg(rd);
        tcg_gen_st_i64(tcg_tmp, cpu_env, freg_offs_d);
        break;
#if 0 /* XXX */
    case 0x7: /* FCVT (single to half) */
        break;
#endif
    case 0x8: /* FRINTN XXX add rounding mode */
    case 0x9: /* FRINTP */
    case 0xa: /* FRINTM */
    case 0xb: /* FRINTZ */
    case 0xc: /* FRINTA */
    case 0xe: /* FRINTX */
    case 0xf: /* FRINTI */
    {
        TCGv_i32 tcg_rmode = tcg_const_i32(opcode & 7);

        gen_helper_set_rmode(tcg_rmode, fpst);
        gen_helper_rints(tcg_res, tcg_op, fpst);

        /* XXX use fpcr */
        tcg_gen_movi_i32(tcg_rmode, -1);
        gen_helper_set_rmode(tcg_rmode, fpst);
        tcg_temp_free_i32(tcg_rmode);
        break;
    }
    default:
        unallocated_encoding(s);
        return;
    }

    if (!skip_write) {
        clear_fpreg(rd);
        tcg_gen_ext32u_i64(tcg_tmp, tcg_res);
        tcg_gen_st32_i64(tcg_tmp, cpu_env, freg_offs_d);
    }

    tcg_temp_free_ptr(fpst);
    tcg_temp_free_i32(tcg_op);
    tcg_temp_free_i32(tcg_res);
    tcg_temp_free_i64(tcg_tmp);
}

/* floating point data processing (2 source) 64-bit */
static void handle_fpdp2s64(DisasContext *s, uint32_t insn)
{
    int rd = get_bits(insn, 0, 5);
    int rn = get_bits(insn, 5, 5);
    int rm = get_bits(insn, 16, 5);
    int opcode = get_bits(insn, 12, 4);
    int freg_offs_n = offsetof(CPUARMState, vfp.regs[rn * 2]);
    int freg_offs_m = offsetof(CPUARMState, vfp.regs[rm * 2]);
    int freg_offs_d = offsetof(CPUARMState, vfp.regs[rd * 2]);
    TCGv_i64 tcg_op1 = tcg_temp_new_i64();
    TCGv_i64 tcg_op2 = tcg_temp_new_i64();
    TCGv_i64 tcg_res = tcg_temp_new_i64();
    TCGv_ptr fpst = get_fpstatus_ptr();

    tcg_gen_ld_i64(tcg_op1, cpu_env, freg_offs_n);
    tcg_gen_ld_i64(tcg_op2, cpu_env, freg_offs_m);

    switch (opcode) {
    case 0x0: /* FMUL */
        gen_helper_vfp_muld(tcg_res, tcg_op1, tcg_op2, fpst);
        break;
    case 0x8: /* FNMUL */
        gen_helper_vfp_muld(tcg_res, tcg_op1, tcg_op2, fpst);
        gen_helper_vfp_negd(tcg_res, tcg_res);
        break;
    case 0x1: /* FDIV */
        gen_helper_vfp_divd(tcg_res, tcg_op1, tcg_op2, fpst);
        break;
    case 0x2: /* FADD */
        gen_helper_vfp_addd(tcg_res, tcg_op1, tcg_op2, fpst);
        break;
    case 0x3: /* FSUB */
        gen_helper_vfp_subd(tcg_res, tcg_op1, tcg_op2, fpst);
        break;
    case 0x4: /* FMAX */
	gen_helper_vfp_maxd(tcg_res, tcg_op1, tcg_op2, fpst);
	break;
    case 0x5: /* FMIN */
	gen_helper_vfp_mind(tcg_res, tcg_op1, tcg_op2, fpst);
	break;
    case 0x6: /* FMAXNM */
	gen_helper_vfp_maxnmd(tcg_res, tcg_op1, tcg_op2, fpst);
	break;
    case 0x7: /* FMINNM */
	gen_helper_vfp_minnmd(tcg_res, tcg_op1, tcg_op2, fpst);
	break;
    default:
        unallocated_encoding(s);
        return;
    }

    clear_fpreg(rd);
    tcg_gen_st_i64(tcg_res, cpu_env, freg_offs_d);

    tcg_temp_free_ptr(fpst);
    tcg_temp_free_i64(tcg_op1);
    tcg_temp_free_i64(tcg_op2);
    tcg_temp_free_i64(tcg_res);
}

/* floating point data processing (2 source) 32-bit */
static void handle_fpdp2s32(DisasContext *s, uint32_t insn)
{
    int rd = get_bits(insn, 0, 5);
    int rn = get_bits(insn, 5, 5);
    int rm = get_bits(insn, 16, 5);
    int opcode = get_bits(insn, 12, 4);
    int freg_offs_n = offsetof(CPUARMState, vfp.regs[rn * 2]);
    int freg_offs_m = offsetof(CPUARMState, vfp.regs[rm * 2]);
    int freg_offs_d = offsetof(CPUARMState, vfp.regs[rd * 2]);
    TCGv_i64 tcg_tmp = tcg_temp_new_i64();
    TCGv_i32 tcg_op1 = tcg_temp_new_i32();
    TCGv_i32 tcg_op2 = tcg_temp_new_i32();
    TCGv_i32 tcg_res = tcg_temp_new_i32();
    TCGv_ptr fpst = get_fpstatus_ptr();

    tcg_gen_ld_i64(tcg_tmp, cpu_env, freg_offs_n);
    tcg_gen_trunc_i64_i32(tcg_op1, tcg_tmp);
    tcg_gen_ld_i64(tcg_tmp, cpu_env, freg_offs_m);
    tcg_gen_trunc_i64_i32(tcg_op2, tcg_tmp);

    switch (opcode) {
    case 0x0: /* FMUL */
        gen_helper_vfp_muls(tcg_res, tcg_op1, tcg_op2, fpst);
        break;
    case 0x8: /* FNMUL */
        gen_helper_vfp_muls(tcg_res, tcg_op1, tcg_op2, fpst);
        gen_helper_vfp_negs(tcg_res, tcg_res);
        break;
    case 0x1: /* FDIV */
        gen_helper_vfp_divs(tcg_res, tcg_op1, tcg_op2, fpst);
        break;
    case 0x2: /* FADD */
        gen_helper_vfp_adds(tcg_res, tcg_op1, tcg_op2, fpst);
        break;
    case 0x3: /* FSUB */
        gen_helper_vfp_subs(tcg_res, tcg_op1, tcg_op2, fpst);
        break;
    case 0x4: /* FMAX */
	gen_helper_vfp_maxs(tcg_res, tcg_op1, tcg_op2, fpst);
	break;
    case 0x5: /* FMIN */
	gen_helper_vfp_mins(tcg_res, tcg_op1, tcg_op2, fpst);
	break;
    case 0x6: /* FMAXNM */
	gen_helper_vfp_maxnms(tcg_res, tcg_op1, tcg_op2, fpst);
	break;
    case 0x7: /* FMINNM */
	gen_helper_vfp_minnms(tcg_res, tcg_op1, tcg_op2, fpst);
	break;
    default:
        unallocated_encoding(s);
        return;
    }

    clear_fpreg(rd);
    tcg_gen_ext32u_i64(tcg_tmp, tcg_res);
    tcg_gen_st32_i64(tcg_tmp, cpu_env, freg_offs_d);

    tcg_temp_free_ptr(fpst);
    tcg_temp_free_i32(tcg_op1);
    tcg_temp_free_i32(tcg_op2);
    tcg_temp_free_i32(tcg_res);
    tcg_temp_free_i64(tcg_tmp);
}

/* floating point data processing (3 source) double precision */
static void handle_fpdp3s64(DisasContext *s, uint32_t insn)
{
    int rd = get_bits(insn, 0, 5);
    int rn = get_bits(insn, 5, 5);
    int ra = get_bits(insn, 10, 5);
    int rm = get_bits(insn, 16, 5);
    bool is_neg = get_bits(insn, 21, 1);
    bool is_sub = get_bits(insn, 15, 1);
    int freg_offs_n = offsetof(CPUARMState, vfp.regs[rn * 2]);
    int freg_offs_a = offsetof(CPUARMState, vfp.regs[ra * 2]);
    int freg_offs_m = offsetof(CPUARMState, vfp.regs[rm * 2]);
    int freg_offs_d = offsetof(CPUARMState, vfp.regs[rd * 2]);
    TCGv_i64 tcg_op1 = tcg_temp_new_i64();
    TCGv_i64 tcg_op2 = tcg_temp_new_i64();
    TCGv_i64 tcg_op3 = tcg_temp_new_i64();
    TCGv_i64 tcg_res = tcg_temp_new_i64();
    TCGv_ptr fpst = get_fpstatus_ptr();

    tcg_gen_ld_i64(tcg_op1, cpu_env, freg_offs_n);
    tcg_gen_ld_i64(tcg_op2, cpu_env, freg_offs_m);
    tcg_gen_ld_i64(tcg_op3, cpu_env, freg_offs_a);

    if (is_neg) {
        gen_helper_vfp_negd(tcg_op1, tcg_op1);
        gen_helper_vfp_negd(tcg_op3, tcg_op3);
    }

    gen_helper_vfp_muld(tcg_res, tcg_op1, tcg_op2, fpst);
    if (is_sub) {
        gen_helper_vfp_subd(tcg_res, tcg_op3, tcg_res, fpst);
    } else {
        gen_helper_vfp_addd(tcg_res, tcg_op3, tcg_res, fpst);
    }

    clear_fpreg(rd);
    tcg_gen_st_i64(tcg_res, cpu_env, freg_offs_d);

    tcg_temp_free_ptr(fpst);
    tcg_temp_free_i64(tcg_op1);
    tcg_temp_free_i64(tcg_op2);
    tcg_temp_free_i64(tcg_op3);
    tcg_temp_free_i64(tcg_res);
}

/* floating point data processing (3 source) single precision */
static void handle_fpdp3s32(DisasContext *s, uint32_t insn)
{
    int rd = get_bits(insn, 0, 5);
    int rn = get_bits(insn, 5, 5);
    int ra = get_bits(insn, 10, 5);
    int rm = get_bits(insn, 16, 5);
    bool is_neg = get_bits(insn, 21, 1);
    bool is_sub = get_bits(insn, 15, 1);
    int freg_offs_n = offsetof(CPUARMState, vfp.regs[rn * 2]);
    int freg_offs_a = offsetof(CPUARMState, vfp.regs[ra * 2]);
    int freg_offs_m = offsetof(CPUARMState, vfp.regs[rm * 2]);
    int freg_offs_d = offsetof(CPUARMState, vfp.regs[rd * 2]);
    TCGv_i64 tcg_tmp = tcg_temp_new_i64();
    TCGv_i32 tcg_op1 = tcg_temp_new_i32();
    TCGv_i32 tcg_op2 = tcg_temp_new_i32();
    TCGv_i32 tcg_op3 = tcg_temp_new_i32();
    TCGv_i32 tcg_res = tcg_temp_new_i32();
    TCGv_ptr fpst = get_fpstatus_ptr();

    tcg_gen_ld_i64(tcg_tmp, cpu_env, freg_offs_n);
    tcg_gen_trunc_i64_i32(tcg_op1, tcg_tmp);
    tcg_gen_ld_i64(tcg_tmp, cpu_env, freg_offs_m);
    tcg_gen_trunc_i64_i32(tcg_op2, tcg_tmp);
    tcg_gen_ld_i64(tcg_tmp, cpu_env, freg_offs_a);
    tcg_gen_trunc_i64_i32(tcg_op3, tcg_tmp);

    if (is_neg) {
        gen_helper_vfp_negs(tcg_op1, tcg_op1);
        gen_helper_vfp_negs(tcg_op3, tcg_op3);
    }

    gen_helper_vfp_muls(tcg_res, tcg_op1, tcg_op2, fpst);
    if (is_sub) {
        gen_helper_vfp_subs(tcg_res, tcg_op3, tcg_res, fpst);
    } else {
        gen_helper_vfp_adds(tcg_res, tcg_op3, tcg_res, fpst);
    }

    clear_fpreg(rd);
    tcg_gen_ext32u_i64(tcg_tmp, tcg_res);
    tcg_gen_st32_i64(tcg_tmp, cpu_env, freg_offs_d);

    tcg_temp_free_ptr(fpst);
    tcg_temp_free_i32(tcg_op1);
    tcg_temp_free_i32(tcg_op2);
    tcg_temp_free_i32(tcg_op3);
    tcg_temp_free_i32(tcg_res);
    tcg_temp_free_i64(tcg_tmp);
}

static void simd_ld(TCGv_i64 tcg_reg, int freg_offs, int size, bool sext)
{
    switch ((size << 1) | (int)sext) {
    case 0:
        tcg_gen_ld8u_i64(tcg_reg, cpu_env, freg_offs);
        break;
    case 1:
        tcg_gen_ld8s_i64(tcg_reg, cpu_env, freg_offs);
        break;
    case 2:
        tcg_gen_ld16u_i64(tcg_reg, cpu_env, freg_offs);
        break;
    case 3:
        tcg_gen_ld16s_i64(tcg_reg, cpu_env, freg_offs);
        break;
    case 4:
        tcg_gen_ld32u_i64(tcg_reg, cpu_env, freg_offs);
        break;
    case 5:
        tcg_gen_ld32s_i64(tcg_reg, cpu_env, freg_offs);
        break;
    case 6: case 7:
        tcg_gen_ld_i64(tcg_reg, cpu_env, freg_offs);
        break;
    }
}

static void simd_st(TCGv_i64 tcg_reg, int freg_offs, int size)
{
    switch (size) {
    case 0:
        tcg_gen_st8_i64(tcg_reg, cpu_env, freg_offs);
        break;
    case 1:
        tcg_gen_st16_i64(tcg_reg, cpu_env, freg_offs);
        break;
    case 2:
        tcg_gen_st32_i64(tcg_reg, cpu_env, freg_offs);
        break;
    case 3:
        tcg_gen_st_i64(tcg_reg, cpu_env, freg_offs);
        break;
    }
}

static void handle_v3add(DisasContext *s, uint32_t insn)
{
    int rd = get_bits(insn, 0, 5);
    int rn = get_bits(insn, 5, 5);
    int rm = get_bits(insn, 16, 5);
    int size = get_bits(insn, 22, 2);
    bool is_sub = get_bits(insn, 29, 1);
    int freg_offs_d = offsetof(CPUARMState, vfp.regs[rd * 2]);
    int freg_offs_n = offsetof(CPUARMState, vfp.regs[rn * 2]);
    int freg_offs_m = offsetof(CPUARMState, vfp.regs[rm * 2]);
    TCGv_i64 tcg_op1 = tcg_temp_new_i64();
    TCGv_i64 tcg_op2 = tcg_temp_new_i64();
    TCGv_i64 tcg_res = tcg_temp_new_i64();

    simd_ld(tcg_op1, freg_offs_n, size, false);
    simd_ld(tcg_op2, freg_offs_m, size, false);

    if (is_sub) {
        tcg_gen_sub_i64(tcg_res, tcg_op1, tcg_op2);
    } else {
        tcg_gen_add_i64(tcg_res, tcg_op1, tcg_op2);
    }

    clear_fpreg(rd);
    simd_st(tcg_res, freg_offs_d, size);

    tcg_temp_free_i64(tcg_op1);
    tcg_temp_free_i64(tcg_op2);
    tcg_temp_free_i64(tcg_res);
}

/* SIMD logical ops (three same, opcode 3) */
static void handle_simdorr(DisasContext *s, uint32_t insn)
{
    int rd = get_bits(insn, 0, 5);
    int rn = get_bits(insn, 5, 5);
    int rm = get_bits(insn, 16, 5);
    int size = get_bits(insn, 22, 2);
    int opcode = get_bits(insn, 11, 5);
    bool is_u = get_bits(insn, 29, 1);
    bool is_q = get_bits(insn, 30, 1);
    int freg_offs_d = offsetof(CPUARMState, vfp.regs[rd * 2]);
    int freg_offs_n = offsetof(CPUARMState, vfp.regs[rn * 2]);
    int freg_offs_m = offsetof(CPUARMState, vfp.regs[rm * 2]);
    TCGv_i64 tcg_op1_1 = tcg_temp_new_i64();
    TCGv_i64 tcg_op1_2 = tcg_temp_new_i64();
    TCGv_i64 tcg_op2_1 = tcg_temp_new_i64();
    TCGv_i64 tcg_op2_2 = tcg_temp_new_i64();
    TCGv_i64 tcg_res_1 = tcg_temp_new_i64();
    TCGv_i64 tcg_res_2 = tcg_temp_new_i64();

    if (opcode != 0x3) {
	unallocated_encoding(s);
	return;
    }

    tcg_gen_ld_i64(tcg_op1_1, cpu_env, freg_offs_n);
    tcg_gen_ld_i64(tcg_op2_1, cpu_env, freg_offs_m);
    if (is_q) {
        tcg_gen_ld_i64(tcg_op1_2, cpu_env, freg_offs_n + sizeof(float64));
        tcg_gen_ld_i64(tcg_op2_2, cpu_env, freg_offs_m + sizeof(float64));
    }

    if (!is_u) {
	/* AND, BIC, ORR, ORN */
        if (size & 1) {
            tcg_gen_not_i64(tcg_op2_1, tcg_op2_1);
	    if (is_q)
	      tcg_gen_not_i64(tcg_op2_2, tcg_op2_2);
        }
        if (size & 2) {
            tcg_gen_or_i64(tcg_res_1, tcg_op1_1, tcg_op2_1);
	    if (is_q)
	      tcg_gen_or_i64(tcg_res_2, tcg_op1_2, tcg_op2_2);
        } else {
            tcg_gen_and_i64(tcg_res_1, tcg_op1_1, tcg_op2_1);
	    if (is_q)
	      tcg_gen_and_i64(tcg_res_2, tcg_op1_2, tcg_op2_2);
        }
    } else {
	/* EOR, BSL, BIT, BIF */
	switch (size) {
	case 0: /* EOR */
	    tcg_gen_xor_i64(tcg_res_1, tcg_op1_1, tcg_op2_1);
	    if (is_q)
	      tcg_gen_xor_i64(tcg_res_2, tcg_op1_2, tcg_op2_2);
	    break;
	case 1: /* BSL bitwise select */
	    tcg_gen_ld_i64(tcg_res_1, cpu_env, freg_offs_d);
	    tcg_gen_xor_i64(tcg_op1_1, tcg_op1_1, tcg_op2_1);
	    tcg_gen_and_i64(tcg_op1_1, tcg_op1_1, tcg_res_1);
	    tcg_gen_xor_i64(tcg_res_1, tcg_op2_1, tcg_op1_1);
	    if (is_q) {
		tcg_gen_ld_i64(tcg_res_2, cpu_env, freg_offs_d + sizeof(float64));
		tcg_gen_xor_i64(tcg_op1_2, tcg_op1_2, tcg_op2_2);
		tcg_gen_and_i64(tcg_op1_2, tcg_op1_2, tcg_res_2);
		tcg_gen_xor_i64(tcg_res_2, tcg_op2_2, tcg_op1_2);
	    }
	    break;
	case 2: /* BIT, bitwise insert if true */
	    tcg_gen_ld_i64(tcg_res_1, cpu_env, freg_offs_d);
	    tcg_gen_xor_i64(tcg_op1_1, tcg_op1_1, tcg_res_1);
	    tcg_gen_and_i64(tcg_op1_1, tcg_op1_1, tcg_op2_1);
	    tcg_gen_xor_i64(tcg_res_1, tcg_res_1, tcg_op1_1);
	    if (is_q) {
		tcg_gen_ld_i64(tcg_res_2, cpu_env, freg_offs_d + sizeof(float64));
		tcg_gen_xor_i64(tcg_op1_2, tcg_op1_2, tcg_res_2);
		tcg_gen_and_i64(tcg_op1_2, tcg_op1_2, tcg_op2_2);
		tcg_gen_xor_i64(tcg_res_2, tcg_res_2, tcg_op1_2);
	    }
	    break;
	case 3: /* BIF, bitwise insert if false */
	    tcg_gen_ld_i64(tcg_res_1, cpu_env, freg_offs_d);
	    tcg_gen_xor_i64(tcg_op1_1, tcg_op1_1, tcg_res_1);
	    tcg_gen_andc_i64(tcg_op1_1, tcg_op1_1, tcg_op2_1);
	    tcg_gen_xor_i64(tcg_res_1, tcg_res_1, tcg_op1_1);
	    if (is_q) {
		tcg_gen_ld_i64(tcg_res_2, cpu_env, freg_offs_d + sizeof(float64));
		tcg_gen_xor_i64(tcg_op1_2, tcg_op1_2, tcg_res_2);
		tcg_gen_andc_i64(tcg_op1_2, tcg_op1_2, tcg_op2_2);
		tcg_gen_xor_i64(tcg_res_2, tcg_res_2, tcg_op1_2);
	    }
	    break;
	}
    }

    tcg_gen_st_i64(tcg_res_1, cpu_env, freg_offs_d);
    if (!is_q) {
        tcg_gen_movi_i64(tcg_res_2, 0);
    }
    tcg_gen_st_i64(tcg_res_2, cpu_env, freg_offs_d + sizeof(float64));

    tcg_temp_free_i64(tcg_op1_1);
    tcg_temp_free_i64(tcg_op1_2);
    tcg_temp_free_i64(tcg_op2_1);
    tcg_temp_free_i64(tcg_op2_2);
    tcg_temp_free_i64(tcg_res_1);
    tcg_temp_free_i64(tcg_res_2);
}

/* SIMD 3 same (U=0) */
static void handle_simd3su0(DisasContext *s, uint32_t insn)
{
    int rd = get_bits(insn, 0, 5);
    int rn = get_bits(insn, 5, 5);
    int rm = get_bits(insn, 16, 5);
    int size = get_bits(insn, 22, 2);
    int opcode = get_bits(insn, 11, 5);
    bool is_q = get_bits(insn, 30, 1);
    bool is_u = get_bits(insn, 29, 1);
    bool is_pair = is_u;
    bool is_float = false;
    bool is_op2 = false;
    int freg_offs_d = offsetof(CPUARMState, vfp.regs[rd * 2]);
    int freg_offs_n = offsetof(CPUARMState, vfp.regs[rn * 2]);
    int freg_offs_m = offsetof(CPUARMState, vfp.regs[rm * 2]);
    TCGv_i64 tcg_op1 = tcg_temp_new_i64();
    TCGv_i64 tcg_op2 = tcg_temp_new_i64();
    TCGv_i64 tcg_res = tcg_temp_new_i64();
    int ebytes = (1 << size);
    int i;

    switch (opcode) {
    case 0x13: /* MUL / PMUL */
	if (is_u && size != 0) {
	    /* PMUL is only defined for bytes.  */
	    unallocated_encoding(s);
	    return;
	}
	break;
	       /* base / pair  / sz&2 / sz&2 && pair */
    case 0x1a: /* FADD / FADDP / FSUB / FABD */
	is_float = true;
	is_op2 = size & 2;
	size = (size & 1) + 2;
	ebytes = (1 << size);
	is_u = true;
	if (is_pair) {
	    /* XXX Can't yet handle.  */
	    unallocated_encoding(s);
	    return;
	}
	break;
    }

    /* XXX this is all suboptimal.  When we use the neon helpers
       they already work on a whole 32bit vector (e.g. 4 bytes at once),
       but we still do the loop element-wise, i.e. do four times or
       twice as much work as needed.  */
    for (i = 0; i < (is_q ? 16 : 8); i += ebytes) {
        simd_ld(tcg_op1, freg_offs_n + i, size, !is_u);
        simd_ld(tcg_op2, freg_offs_m + i, size, !is_u);

        switch (opcode) {
        case 0x10: /* ADD / SUB */
            if (is_u) {
                tcg_gen_sub_i64(tcg_res, tcg_op1, tcg_op2);
            } else {
                tcg_gen_add_i64(tcg_res, tcg_op1, tcg_op2);
            }
            break;
	case 0x13: /* MUL / PMUL */
	    if (is_u) {
		gen_helper_neon_mul_p8 (tcg_res, tcg_op1, tcg_op2);
	    } else {
		tcg_gen_mul_i64(tcg_res, tcg_op1, tcg_op2);
	    }
	    break;
	case 0x0c: /* SMAX / UMAX */
	    tcg_gen_movcond_i64 (is_u ? TCG_COND_GEU : TCG_COND_GE,
				 tcg_res,
				 tcg_op1, tcg_op2, tcg_op1, tcg_op2);
	    break;
	case 0x0d: /* SMIN / UMIN */
	    tcg_gen_movcond_i64 (is_u ? TCG_COND_LEU : TCG_COND_LE,
				 tcg_res,
				 tcg_op1, tcg_op2, tcg_op1, tcg_op2);
	    break;
	case 0x06: /* CMGT / CMHI */
	    /* setcond results in 1/0, which we transform to 0/all-ones,
	       the latter being true, so we need to inverse the condition
	       for the test.  GT->LE.  */
	    tcg_gen_setcond_i64 (is_u ? TCG_COND_LEU : TCG_COND_LE,
				 tcg_res, tcg_op1, tcg_op2);
	    goto expand_ones;
	case 0x07: /* CMGE / CMHS */
	    tcg_gen_setcond_i64 (is_u ? TCG_COND_LTU : TCG_COND_LT,
				 tcg_res, tcg_op1, tcg_op2);
	    goto expand_ones;
	case 0x11: /* CMTST / CMEQ */
	    if (is_u) { /* CMEQ */
		tcg_gen_setcond_i64 (TCG_COND_NE, tcg_res, tcg_op1, tcg_op2);
	    } else {
		/* We sign-extended the elements above, that's okay here.  */
		tcg_gen_and_i64 (tcg_res, tcg_op1, tcg_op2);
		tcg_gen_setcondi_i64 (TCG_COND_EQ, tcg_res, tcg_res, 0);
	    }
	    goto expand_ones;
        expand_ones:
	    /* Convert 1/0 into 0/all-ones.  */
	    tcg_gen_subi_i64 (tcg_res, tcg_res, 1);
	    break;

	case 0x1a: /* FADD / FADDP / FSUB / FABD */
	    {
	      TCGv_ptr fpst = get_fpstatus_ptr();

	      if (size == 2) {
		  tcg_gen_trunc_i64_i32(tcg_op1, tcg_op1);
		  tcg_gen_trunc_i64_i32(tcg_op2, tcg_op2);
	      }

	      switch (is_op2 << 8 | opcode) {
	      case 0x01a: /* FADD */
		  if (size == 2)
		    gen_helper_vfp_adds(tcg_res, tcg_op1, tcg_op2, fpst);
		  else
		    gen_helper_vfp_addd(tcg_res, tcg_op1, tcg_op2, fpst);
		  break;
	      case 0x11a: /* FSUB */
		  if (size == 2)
		    gen_helper_vfp_subs(tcg_res, tcg_op1, tcg_op2, fpst);
		  else
		    gen_helper_vfp_subd(tcg_res, tcg_op1, tcg_op2, fpst);
		  break;
	      }

	      tcg_temp_free_ptr(fpst);
	      break;
	    }

	case 0x08: /* SSHL / USHL */
	case 0x09: /* SQSHL / UQSHL (saturating) */
	case 0x0a: /* SRSHL / URSHL (rounding) */
	case 0x0b: /* SQRSHL / UQRSHL (sat + round) */
	    {
	      TCGv_i32 insncode = tcg_const_i32(insn);
	      /* The saturating ones might set the QC flag in fpcsr,
	         so need the environment.  */
	      if (opcode & 1)
		gen_helper_simd_op3s_env(tcg_res, cpu_env, tcg_op1,
					 tcg_op2, insncode);
	      else
		gen_helper_simd_op3s(tcg_res, tcg_op1, tcg_op2, insncode);
	    }
	    break;

        default:
            unallocated_encoding(s);
            return;
        }

        simd_st(tcg_res, freg_offs_d + i, size);
    }

    if (!is_q) {
        TCGv_i64 tcg_zero = tcg_const_i64(0);
        simd_st(tcg_zero, freg_offs_d + sizeof(float64), 3);
        tcg_temp_free_i64(tcg_zero);
    }

    tcg_temp_free_i64(tcg_op1);
    tcg_temp_free_i64(tcg_op2);
    tcg_temp_free_i64(tcg_res);
}

/* SIMD ZIP/UZP/TRN */
static void handle_simd_zip(DisasContext *s, uint32_t insn)
{
    int rd = get_bits(insn, 0, 5);
    int rn = get_bits(insn, 5, 5);
    int rm = get_bits(insn, 16, 5);
    int size = get_bits(insn, 22, 2);
    int opcode = get_bits(insn, 12, 3);
    bool is_q = get_bits(insn, 30, 1);
    int freg_offs_d = offsetof(CPUARMState, vfp.regs[rd * 2]);
    int freg_offs_n = offsetof(CPUARMState, vfp.regs[rn * 2]);
    int freg_offs_m = offsetof(CPUARMState, vfp.regs[rm * 2]);
    TCGv_i64 tcg_res = tcg_temp_new_i64();
    int ebytes = (1 << size);
    int i;
    int datasize = is_q ? 16 : 8;
    int elements = datasize / ebytes;
    bool part = opcode & 4;

    if (size == 3 && !is_q) {
	unallocated_encoding(s);
	return;
    }

    switch (opcode & 3) {
    case 1: /* UZP1/2 */
	for (i = 0; i < datasize; i += ebytes) {
	    if (i < (datasize / 2))
	      simd_ld(tcg_res,
		      freg_offs_n + 2 * i + part * ebytes,
		      size, false);
	    else
	      simd_ld(tcg_res,
		      freg_offs_m + 2 * (i - datasize / 2) + part * ebytes,
		      size, false);
	    simd_st(tcg_res, freg_offs_d + i, size);
	}
	break;
    case 2: /* TRN1/2 */
	for (i = 0; i < elements; i++) {
	    if (i & 1)
	      simd_ld(tcg_res, freg_offs_m + ((i & ~1) + part) * ebytes,
		      size, false);
	    else
	      simd_ld(tcg_res, freg_offs_n + ((i & ~1) + part) * ebytes,
		      size, false);
	    simd_st(tcg_res, freg_offs_d + i * ebytes, size);
	}
	break;
    case 3: /* ZIP1/2 */
	if (part) {
	    freg_offs_n += sizeof(float64);
	    freg_offs_m += sizeof(float64);
	}
	for (i = 0; i < elements; i++) {
	    if (i & 1)
	      simd_ld(tcg_res, freg_offs_m + (i >> 1) * ebytes, size, false);
	    else
	      simd_ld(tcg_res, freg_offs_n + (i >> 1) * ebytes, size, false);
	    simd_st(tcg_res, freg_offs_d + i * ebytes, size);
	}
	break;
    default:
	unallocated_encoding(s);
	return;
    }

    tcg_temp_free_i64(tcg_res);

    if (!is_q) {
        TCGv_i64 tcg_zero = tcg_const_i64(0);
        simd_st(tcg_zero, freg_offs_d + sizeof(float64), 3);
        tcg_temp_free_i64(tcg_zero);
    }
}

static void handle_simd_accross(DisasContext *s, uint32_t insn)
{
    int rd = get_bits(insn, 0, 5);
    int rn = get_bits(insn, 5, 5);
    int size = get_bits(insn, 22, 2);
    int opcode = get_bits(insn, 12, 5);
    bool is_q = get_bits(insn, 30, 1);
    bool is_u = get_bits(insn, 29, 1);
    int freg_offs_d = offsetof(CPUARMState, vfp.regs[rd * 2]);
    int freg_offs_n = offsetof(CPUARMState, vfp.regs[rn * 2]);
    TCGv_i64 tcg_op1 = tcg_temp_new_i64();
    TCGv_i64 tcg_res = tcg_temp_new_i64();
    int ebytes = (1 << size);
    int i;

    /* For the non-floats size can't be 3, and if it's 2 then
       Q is set (128 bits).  So for ADDLV which uses a 2*esize
       result, we still fit into 64 bit.
       The float min/max (and ADDV) are defined via
       the Reduce pseudoop that uses divide and conquer on the two halfs
       of the input.  The integer ones user linear iteration over the
       vec elements.  Both should be equivalent.  */
    simd_ld(tcg_res, freg_offs_n + 0, size, !is_u);
    for (i = ebytes; i < (is_q ? 16 : 8); i += ebytes) {
        simd_ld(tcg_op1, freg_offs_n + i, size, !is_u);

        switch (opcode) {
	case 0x03: /* SADDLV / UADDLV */
	    tcg_gen_add_i64(tcg_res, tcg_res, tcg_op1);
	    break;
	case 0x0a: /* SMAXV / UMAXV */
	    tcg_gen_movcond_i64 (is_u ? TCG_COND_GEU : TCG_COND_GE,
				 tcg_res,
				 tcg_res, tcg_op1, tcg_res, tcg_op1);
	    break;
	case 0x1a: /* SMINV / UMINV */
	    tcg_gen_movcond_i64 (is_u ? TCG_COND_LEU : TCG_COND_LE,
				 tcg_res,
				 tcg_res, tcg_op1, tcg_res, tcg_op1);
	    break;
	case 0x1b: /* ADDV */
	    if (is_u) {
	        unallocated_encoding(s);
		return;
	    }
	    /* ADDV actually is supposed to use esize width intermediate
	       results, due to 2-complement we can as well use a 64bit
	       intermediate and truncate later.  */
	    tcg_gen_add_i64(tcg_res, tcg_res, tcg_op1);
	    break;

	case 0x0c: /* FMAXNMV / FMINNMV */
	case 0x0f: /* FMAXV / FMINV*/
	    /* We don't yet implement these.  */
	default:
	    unallocated_encoding(s);
	    return;
        }

    }

    tcg_temp_free_i64(tcg_op1);
    
    if (opcode == 0x03) {
	/* ADDLV uses a 2*esize result, and esize can't be 64.  */
	switch (size) {
	case 0: tcg_gen_ext16u_i64(tcg_res, tcg_res); break;
	case 1: tcg_gen_ext32u_i64(tcg_res, tcg_res); break;
	}
	simd_st(tcg_res, freg_offs_d + 0, size + 1);
    } else {
	switch (size) {
	case 0: tcg_gen_ext8u_i64(tcg_res, tcg_res); break;
	case 1: tcg_gen_ext16u_i64(tcg_res, tcg_res); break;
	case 2: tcg_gen_ext32u_i64(tcg_res, tcg_res); break;
	}
	simd_st(tcg_res, freg_offs_d + 0, size);
    }

    tcg_temp_free_i64(tcg_res);

    /* The upper part of the vector here is always zero.  Irrespective
       of the Q bit.  */
    TCGv_i64 tcg_zero = tcg_const_i64(0);
    simd_st(tcg_zero, freg_offs_d + sizeof(float64), 3);
    tcg_temp_free_i64(tcg_zero);
}

/* AdvSIMD two-reg misc */
static void handle_simd_misc(DisasContext *s, uint32_t insn)
{
    int rd = get_bits(insn, 0, 5);
    int rn = get_bits(insn, 5, 5);
    int size = get_bits(insn, 22, 2);
    int opcode = get_bits(insn, 12, 5);
    bool is_u = get_bits(insn, 29, 1);
    bool is_q = get_bits(insn, 30, 1);
    int freg_offs_d = offsetof(CPUARMState, vfp.regs[rd * 2]);
    int freg_offs_n = offsetof(CPUARMState, vfp.regs[rn * 2]);
    TCGv_i64 tcg_op1 = tcg_temp_new_i64();
    TCGv_i64 tcg_res = tcg_temp_new_i64();
    int ebytes = (1 << size);
    int i;

    switch (opcode) {
    case 0x12: /* XTN, Extract narrow */
	if (size == 3) {
	    unallocated_encoding(s);
	    return;
	}
	/* We need to initialize tcg_res, even though we overwrite
	   all its bytes, otherwise TCG will generate wrong code.  */
	tcg_gen_movi_i64(tcg_res, 0);
	for (i = 0; i < 8; i += ebytes) {
	    /* XXX Should we be able to just generate a smaller
	       load?  */
	    simd_ld(tcg_op1, freg_offs_n + 2 * i, size + 1, false);
	    tcg_gen_deposit_i64(tcg_res, tcg_res, tcg_op1, 8 * i, 8 << size);
	}
	simd_st(tcg_res, freg_offs_d + (is_q ? sizeof(float64) : 0), 3);
	break;
    case 0x0f: /* FABS / FNEG */
	size = (size & 1) + 2;
	ebytes = 1 << size;
	for (i = 0; i < (is_q ? 16 : 8); i += ebytes) {
	    /* XXX For 32bit this creates a zero extended 64bit value,
	       i.e. the float in the low bits.  The helpers are supposed
	       to use TCGv_i32, and we give them this zero-extended TCGv_i64.
	       This seems to work just fine, but is it guaranteed?  */
	    simd_ld(tcg_op1, freg_offs_n + i, size, false);
	    if (is_u) {
		if (ebytes == 4)
		  gen_helper_vfp_negs(tcg_res, tcg_op1);
		else
		  gen_helper_vfp_negd(tcg_res, tcg_op1);
	    } else {
		if (ebytes == 4)
		  gen_helper_vfp_abss(tcg_res, tcg_op1);
		else
		  gen_helper_vfp_absd(tcg_res, tcg_op1);
	    }
	    simd_st(tcg_res, freg_offs_d + i, size);
	}
	break;

    default:
	unallocated_encoding(s);
	return;
    }

    tcg_temp_free_i64(tcg_op1);
    tcg_temp_free_i64(tcg_res);

    if (!is_q) {
        TCGv_i64 tcg_zero = tcg_const_i64(0);
        simd_st(tcg_zero, freg_offs_d + sizeof(float64), 3);
        tcg_temp_free_i64(tcg_zero);
    }
}

/* SIMD movi */
static void handle_simdmovi(DisasContext *s, uint32_t insn)
{
    int rd = get_bits(insn, 0, 5);
    int cmode = get_bits(insn, 12, 4);
    uint64_t abcdefgh = get_bits(insn, 5, 5) | (get_bits(insn, 16, 3) << 5);
    bool is_neg = get_bits(insn, 29, 1);
    bool is_q = get_bits(insn, 30, 1);
    int freg_offs_d = offsetof(CPUARMState, vfp.regs[rd * 2]);
    uint64_t imm = 0;
    int shift, i;
    TCGv_i64 tcg_op1_1 = tcg_temp_new_i64();
    TCGv_i64 tcg_op1_2 = tcg_temp_new_i64();
    TCGv_i64 tcg_res_1 = tcg_temp_new_i64();
    TCGv_i64 tcg_res_2 = tcg_temp_new_i64();
    TCGv_i64 tcg_imm;

    switch ((cmode >> 1) & 0x7) {
    case 0:
    case 1:
    case 2:
    case 3:
        shift = ((cmode >> 1) & 0x7) * 8;
        imm = (abcdefgh << shift) | (abcdefgh << (32 + shift));
        break;
    case 4:
    case 5:
        shift = ((cmode >> 1) & 0x1) * 8;
        imm = (abcdefgh << shift) |
              (abcdefgh << (16 + shift)) |
              (abcdefgh << (32 + shift)) |
              (abcdefgh << (48 + shift));
        break;
    case 6:
        if (cmode & 1) {
            imm = (abcdefgh << 8) |
                  (abcdefgh << 48) |
                  0x000000ff000000ffULL;
        } else {
            imm = (abcdefgh << 16) |
                  (abcdefgh << 56) |
                  0x0000ffff0000ffffULL;
        }
        break;
    case 7:
        if (!(cmode & 1) && !is_neg) {
            imm = abcdefgh |
                  (abcdefgh << 8) |
                  (abcdefgh << 16) |
                  (abcdefgh << 24) |
                  (abcdefgh << 32) |
                  (abcdefgh << 40) |
                  (abcdefgh << 48) |
                  (abcdefgh << 56);
        } else if (!(cmode & 1) && is_neg) {
            imm = 0;
            for (i = 0; i < 8; i++) {
                if ((abcdefgh) & (1 << (7 - i))) {
                    imm |= 0xffULL << (i * 8);
                }
            }
        } else if (cmode & 1) {
            shift = is_neg ? 48 : 19;
            imm = (abcdefgh & 0x1f) << 19;
            if (abcdefgh & 0x80) {
                imm |= 0x80000000;
            }
            if (!(abcdefgh & 0x40)) {
                imm |= 0x40000000;
            }
            if (abcdefgh & 0x20) {
                imm |= is_neg ? 0x3fc00000 : 0x3e000000;
            }
            imm |= (imm << 32);
        }
        shift = ((cmode >> 1) & 0x1) * 8;
        break;
    }

    if (((cmode >> 1) & 7) != 7 && is_neg) {
        imm = ~imm;
    }
    tcg_imm = tcg_const_i64(imm);

    tcg_gen_ld_i64(tcg_op1_1, cpu_env, freg_offs_d);
    if (is_q) {
        tcg_gen_ld_i64(tcg_op1_2, cpu_env, freg_offs_d + sizeof(float64));
    } else {
        tcg_gen_movi_i64(tcg_op1_2, 0);
    }

    if ((cmode == 0xf) && is_neg && !is_q) {
        unallocated_encoding(s);
        return;
    } else if ((cmode & 1) && is_neg) {
        /* AND (BIC) */
        tcg_gen_and_i64(tcg_res_1, tcg_op1_1, tcg_imm);
        tcg_gen_and_i64(tcg_res_2, tcg_op1_2, tcg_imm);
    } else if ((cmode & 1) && !is_neg) {
        /* ORR */
        tcg_gen_or_i64(tcg_res_1, tcg_op1_1, tcg_imm);
        tcg_gen_or_i64(tcg_res_2, tcg_op1_2, tcg_imm);
    } else {
        /* MOVI */
        tcg_gen_mov_i64(tcg_res_1, tcg_imm);
        tcg_gen_mov_i64(tcg_res_2, tcg_imm);
    }

    tcg_gen_st_i64(tcg_res_1, cpu_env, freg_offs_d);
    if (!is_q) {
        tcg_gen_movi_i64(tcg_res_2, 0);
    }
    tcg_gen_st_i64(tcg_res_2, cpu_env, freg_offs_d + sizeof(float64));

    tcg_temp_free_i64(tcg_op1_1);
    tcg_temp_free_i64(tcg_op1_2);
    tcg_temp_free_i64(tcg_res_1);
    tcg_temp_free_i64(tcg_res_2);
    tcg_temp_free_i64(tcg_imm);
}

/* AdvSIMD shift by immediate.  */
static void handle_simd_shifti(DisasContext *s, uint32_t insn)
{
    /* This handles the vector encoding, many of the shifts also
       have a scalar encoding (in page 0x1f), which we also could
       handle with some adjustments.  */
    int rd = get_bits(insn, 0, 5);
    int rn = get_bits(insn, 5, 5);
    int opcode = get_bits(insn, 11, 5);
    int immb = get_bits(insn, 16, 3);
    int immh = get_bits(insn, 19, 4);
    bool is_u = get_bits(insn, 29, 1);
    bool is_q = get_bits(insn, 30, 1);
    bool accumulate = get_bits(insn, 12, 1);
    bool round = get_bits(insn, 13, 1);
    int freg_offs_d = offsetof(CPUARMState, vfp.regs[rd * 2]);
    int freg_offs_n = offsetof(CPUARMState, vfp.regs[rn * 2]);
    TCGv_i64 tcg_tmp = tcg_temp_new_i64();
    TCGv_i64 tmp2;
    int i;
    int ebytes;
    int size;
    int shift = immh << 3 | immb;

    if (!immh) {
	/* Should have been handled by handle_simdmovi.  */
	unallocated_encoding(s);
	return;
    }
    size = 32 - clz32(immh) - 1;
    if (size > 3 && !is_q) {
        unallocated_encoding(s);
        return;
    }

    ebytes = 1 << size;

    switch (opcode) {
    case 0x00: /* SSHR / USHR */
    case 0x02: /* SSRA / USRA (accumulate) */
    case 0x04: /* SRSHR / URSHR (rounding) */
    case 0x06: /* SRSRA / URSRA (accum + rounding) */
	shift = 2 * (8 << size) - shift;
	accumulate = get_bits(insn, 12, 1);
	round = get_bits(insn, 13, 1);
	break;
    case 0x0a: /* SHL / SLI */
	if (is_u) {
	    /* SLI not implemented */
	    unallocated_encoding(s);
	    return;
	}
	accumulate = round = false;
	shift = shift - (8 << size);
	break;
    case 0x14: /* SSHLL / USHLL */
	accumulate = round = false;
	shift = shift - (8 << size);
	if (size >= 3) {
	    unallocated_encoding(s);
	    return;
	}
	/* Do as if datasize is 64 always.  */
	if (is_q)
	  freg_offs_n += sizeof(float64);
	is_q = false;
	/* For the LL variants the store is larger than the load,
	   so if rd == rn we would overwrite parts of our input.
	   So load everything right now and use shifts in the main 
	   loop.  */
	tmp2 = tcg_temp_new_i64();
	simd_ld(tmp2, freg_offs_n, 3, false);
	break;
    default:
	/* So we don't implement any of the Narrow or saturating shifts,
	   neither do we implement the fixed-point conversions in this
	   page (SCVTF, FCVTZS, UCVTF, FCVTZU).  */
	unallocated_encoding(s);
	return;
    }

    if (accumulate)
      tmp2 = tcg_temp_new_i64();

    for (i = 0; i < (is_q ? 16 : 8); i += ebytes) {
	if (opcode != 0x14)
	  simd_ld(tcg_tmp, freg_offs_n + i, size, !is_u);
	else {
	    tcg_gen_shri_i64(tcg_tmp, tmp2, i*8);
	    switch (size << 1 | is_u) {
	    case 0: tcg_gen_ext8s_i64 (tcg_tmp, tcg_tmp); break;
	    case 1: tcg_gen_ext8u_i64 (tcg_tmp, tcg_tmp); break;
	    case 2: tcg_gen_ext16s_i64 (tcg_tmp, tcg_tmp); break;
	    case 3: tcg_gen_ext16u_i64 (tcg_tmp, tcg_tmp); break;
	    case 4: tcg_gen_ext32s_i64 (tcg_tmp, tcg_tmp); break;
	    case 5: tcg_gen_ext32u_i64 (tcg_tmp, tcg_tmp); break;
	    }
	}
	if (round)
	  tcg_gen_addi_i64(tcg_tmp, tcg_tmp, 1 << (shift - 1));
	switch (opcode) {
	case 0x00: /* SSHR / USHR */
	case 0x02: /* SSRA / USRA (accumulate) */
	case 0x04: /* SRSHR / URSHR (rounding) */
	case 0x06: /* SRSRA / URSRA (accum + rounding) */
	    if (is_u)
	      tcg_gen_shri_i64(tcg_tmp, tcg_tmp, shift);
	    else
	      tcg_gen_sari_i64(tcg_tmp, tcg_tmp, shift);
	    break;
	case 0x0a: /* SHL / SLI */
	case 0x14: /* SSHLL / USHLL */
	    tcg_gen_shli_i64(tcg_tmp, tcg_tmp, shift);
	    break;
	}
	if (accumulate) {
	    simd_ld(tmp2, freg_offs_d + i, size, !is_u);
	    tcg_gen_add_i64(tcg_tmp, tcg_tmp, tmp2);
	}
	if (opcode != 0x14)
	  simd_st(tcg_tmp, freg_offs_d + i, size);
	else
	  simd_st(tcg_tmp, freg_offs_d + 2 * i, size + 1);
    }

    if (accumulate)
      tcg_temp_free_i64(tmp2);
    tcg_temp_free_i64(tcg_tmp);

    if (opcode != 0x14 && !is_q) {
        TCGv_i64 tcg_zero = tcg_const_i64(0);
        simd_st(tcg_zero, freg_offs_d + sizeof(float64), 3);
        tcg_temp_free_i64(tcg_zero);
    }
}

/* SIMD load/store multiple (post-indexed) */
static void handle_simdldstm(DisasContext *s, uint32_t insn, bool is_wback)
{
    int rd = get_bits(insn, 0, 5);
    int rn = get_bits(insn, 5, 5);
    int rm = get_bits(insn, 16, 5);
    int size = get_bits(insn, 10, 2);
    int opcode = get_bits(insn, 12, 4);
    bool is_load = get_bits(insn, 22, 1);
    bool is_q = get_bits(insn, 30, 1);
    TCGv_i64 tcg_tmp = tcg_temp_new_i64();
    TCGv_i64 tcg_addr = tcg_temp_new_i64();
    int r, e, xs, tt, rpt, selem;
    int ebytes = 1 << size;
    int elements = (is_q ? 128 : 64) / (8 << size);

    tcg_gen_mov_i64(tcg_addr, cpu_reg_sp(rn));

    switch (opcode) {
    case 0x0:
        rpt = 1;
        selem = 4;
        break;
    case 0x2:
        rpt = 4;
        selem = 1;
        break;
    case 0x4:
        rpt = 1;
        selem = 3;
        break;
    case 0x6:
        rpt = 3;
        selem = 1;
        break;
    case 0x7:
        rpt = 1;
        selem = 1;
        break;
    case 0x8:
        rpt = 1;
        selem = 2;
        break;
    case 0xa:
        rpt = 2;
        selem = 1;
        break;
    default:
        unallocated_encoding(s);
        return;
    }

    if (size == 3 && !is_q && selem != 1) {
        /* reserved */
        unallocated_encoding(s);
    }

    /* XXX check SP alignment on Rn */

    for (r = 0; r < rpt; r++) {
        for (e = 0; e < elements; e++) {
            tt = (rd + r) % 32;
            for (xs = 0; xs < selem; xs++) {
                int freg_offs = offsetof(CPUARMState, vfp.regs[tt * 2]) +
                                  (e * ebytes);

                /* XXX merge with ldst_do_vec() */
                switch (size | (is_load << 4)) {
                case 0x0:
                    tcg_gen_ld8u_i64(tcg_tmp, cpu_env, freg_offs);
                    tcg_gen_qemu_st8(tcg_tmp, tcg_addr, get_mem_index(s));
                    break;
                case 0x1:
                    tcg_gen_ld16u_i64(tcg_tmp, cpu_env, freg_offs);
                    tcg_gen_qemu_st16(tcg_tmp, tcg_addr, get_mem_index(s));
                    break;
                case 0x2:
                    tcg_gen_ld32u_i64(tcg_tmp, cpu_env, freg_offs);
                    tcg_gen_qemu_st32(tcg_tmp, tcg_addr, get_mem_index(s));
                    break;
                case 0x4:
                    tcg_gen_ld_i64(tcg_tmp, cpu_env, freg_offs);
                    tcg_gen_qemu_st64(tcg_tmp, tcg_addr, get_mem_index(s));
                    freg_offs += sizeof(uint64_t);
                    tcg_gen_addi_i64(tcg_addr, tcg_addr, sizeof(uint64_t));
                    /* fall through */
                case 0x3:
                    tcg_gen_ld_i64(tcg_tmp, cpu_env, freg_offs);
                    tcg_gen_qemu_st64(tcg_tmp, tcg_addr, get_mem_index(s));
                    break;
                case 0x10:
                    tcg_gen_qemu_ld8u(tcg_tmp, tcg_addr, get_mem_index(s));
                    tcg_gen_st8_i64(tcg_tmp, cpu_env, freg_offs);
                    break;
                case 0x11:
                    tcg_gen_qemu_ld16u(tcg_tmp, tcg_addr, get_mem_index(s));
                    tcg_gen_st16_i64(tcg_tmp, cpu_env, freg_offs);
                    break;
                case 0x12:
                    tcg_gen_qemu_ld32u(tcg_tmp, tcg_addr, get_mem_index(s));
                    tcg_gen_st32_i64(tcg_tmp, cpu_env, freg_offs);
                    break;
                case 0x14:
                    tcg_gen_qemu_ld64(tcg_tmp, tcg_addr, get_mem_index(s));
                    tcg_gen_st_i64(tcg_tmp, cpu_env, freg_offs);
                    freg_offs += sizeof(uint64_t);
                    tcg_gen_addi_i64(tcg_addr, tcg_addr, sizeof(uint64_t));
                    /* fall through */
                case 0x13:
                    tcg_gen_qemu_ld64(tcg_tmp, tcg_addr, get_mem_index(s));
                    tcg_gen_st_i64(tcg_tmp, cpu_env, freg_offs);
                    break;
                }

                tcg_gen_addi_i64(tcg_addr, tcg_addr, ebytes);
                tt = (tt + 1) % 32;
            }
        }
    }

    if (is_wback) {
        if (rm == 31) {
            tcg_gen_mov_i64(cpu_reg_sp(rn), tcg_addr);
        } else {
            tcg_gen_add_i64(cpu_reg_sp(rn), cpu_reg(rn), cpu_reg(rm));
        }
    }

    tcg_temp_free_i64(tcg_tmp);
    tcg_temp_free_i64(tcg_addr);
}

static void handle_dupg(DisasContext *s, uint32_t insn)
{
    int rd = get_bits(insn, 0, 5);
    int rn = get_bits(insn, 5, 5);
    int imm5 = get_bits(insn, 16, 5);
    int q = get_bits(insn, 30, 1);
    int freg_offs_d = offsetof(CPUARMState, vfp.regs[rd * 2]);
    int size;
    int i;

    for (size = 0; !(imm5 & (1 << size)); size++) {
        if (size > 3) {
            unallocated_encoding(s);
            return;
        }
    }

    if ((size == 3) && !q) {
        /* XXX reserved value */
        unallocated_encoding(s);
    }

    clear_fpreg(rd);
    switch (size) {
    case 0:
        for (i = 0; i < (q ? 16 : 8); i++) {
            tcg_gen_st8_i64(cpu_reg(rn), cpu_env, freg_offs_d + i);
        }
        break;
    case 1:
        for (i = 0; i < (q ? 16 : 8); i+=2) {
            tcg_gen_st16_i64(cpu_reg(rn), cpu_env, freg_offs_d + i);
        }
        break;
    case 2:
        for (i = 0; i < (q ? 16 : 8); i+=4) {
            tcg_gen_st32_i64(cpu_reg(rn), cpu_env, freg_offs_d + i);
        }
        break;
    case 3:
        for (i = 0; i < (q ? 16 : 8); i+=8) {
            tcg_gen_st_i64(cpu_reg(rn), cpu_env, freg_offs_d + i);
        }
        break;
    }
}

static void handle_umov_smov(DisasContext *s, uint32_t insn)
{
    int rd = get_bits(insn, 0, 5);
    int rn = get_bits(insn, 5, 5);
    int imm5 = get_bits(insn, 16, 5);
    bool is_signed = get_bits(insn, 11, 4) == 0x5;
    int q = get_bits(insn, 30, 1);
    int freg_offs_n = offsetof(CPUARMState, vfp.regs[rn * 2]);
    int size;

    size = ctz32(imm5);
    if (size > 3 || (size > 2 && is_signed)) {
	unallocated_encoding(s);
	return;
    }
    if ((size == 3) && !q) {
        /* XXX reserved value */
        unallocated_encoding(s);
    }

    simd_ld(cpu_reg(rd),
	    freg_offs_n + (get_bits(imm5, 1+size, 4-size) << size),
	    size, is_signed);
}

static void handle_ins_elem(DisasContext *s, uint32_t insn)
{
    int rd = get_bits(insn, 0, 5);
    int rn = get_bits(insn, 5, 5);
    int imm4 = get_bits(insn, 11, 4);
    int imm5 = get_bits(insn, 16, 5);
    int freg_offs_n = offsetof(CPUARMState, vfp.regs[rn * 2]);
    int freg_offs_d = offsetof(CPUARMState, vfp.regs[rd * 2]);
    int size;
    int src_index, dst_index;
    TCGv_i64 tmp;

    size = ctz32(imm5);
    if (size > 3) {
	unallocated_encoding(s);
	return;
    }
    dst_index = get_bits(imm5, 1+size, 4-size);
    src_index = get_bits(imm4, size, 4-size);

    tmp = tcg_temp_new_i64();

    simd_ld(tmp, freg_offs_n + (src_index << size), size, false);
    simd_st(tmp, freg_offs_d + (dst_index << size), size);

    tcg_temp_free_i64(tmp);
}

static void handle_insg(DisasContext *s, uint32_t insn)
{
    int rd = get_bits(insn, 0, 5);
    int rn = get_bits(insn, 5, 5);
    int imm5 = get_bits(insn, 16, 5);
    int freg_offs_d = offsetof(CPUARMState, vfp.regs[rd * 2]);
    int size;

    size = ctz32(imm5);
    if (size > 3) {
	unallocated_encoding(s);
	return;
    }

    simd_st(cpu_reg(rn),
	    freg_offs_d + (get_bits(imm5, 1+size, 4-size) << size),
	    size);
}

void disas_a64_insn(CPUARMState *env, DisasContext *s)
{
    uint32_t insn;

    insn = arm_ldl_code(env, s->pc, s->bswap_code);
    s->pc += 4;

#ifdef DEBUG_SINGLESTEP
    // XXX singlestep hack
    gen_a64_set_pc_im(s->pc);
    s->is_jmp = DISAS_JUMP;
    //////////////////////
#endif

//    fprintf(stderr, "insn: %08x\n", insn);

    /* One-off branch instruction layout */
    switch ((insn & 0xfc000000) >> 26) {
    case 0x25:
    case 0x5:
        handle_b(s, insn);
        return;
    case 0x35:
        if ((insn & 0xff9ffc1f) == 0xd61f0000) {
            handle_br(s, insn);
            return;
        }
        break;
    }

    /* Typical major opcode encoding */
    switch ((insn >> 24) & 0x1f) {
    case 0x0a:
        handle_orr(s, insn);
        break;
    case 0x0b:
        handle_add(s, insn);
        break;
    case 0x08:
    case 0x09:
        if (get_bits(insn, 29, 1)) {
            handle_stp(s, insn);
        } else {
            handle_ldarx(s, insn);
        }
        break;
    case 0x0c:
        if (get_bits(insn, 29, 1)) {
            handle_stp(s, insn);
        } else if (!get_bits(insn, 31, 1) && get_bits(insn, 23, 1) &&
                   !get_bits(insn, 21, 1)) {
            handle_simdldstm(s, insn, true);
        } else if (!get_bits(insn, 31, 1) && !get_bits(insn, 29, 1) &&
                   !get_bits(insn, 23, 1) && !get_bits(insn, 16, 6)) {
            handle_simdldstm(s, insn, false);
        } else {
            goto unknown_insn;
        }
        break;
    case 0x0d:
        if (get_bits(insn, 29, 1)) {
            handle_stp(s, insn);
        } else {
            goto unknown_insn;
        }
        break;
    case 0x0e:
	if (get_bits(insn, 31, 1))
	  goto unknown_insn;
	if (!get_bits(insn, 21, 3) && !get_bits(insn, 15, 1) &&
	    get_bits(insn, 10, 1)) {
	    /* AdvSIMD copy.  */
	    if (get_bits(insn, 29, 2) == 3) /* INS element */
	      handle_ins_elem(s, insn);
	    else if (get_bits(insn, 29, 2) == 1) { /* unalloc */
		goto unknown_insn;
	    } else
	      switch (get_bits(insn, 11, 4)) {
	      case 0x1: /* DUP general reg */
		  handle_dupg(s, insn);
		  break;
	      case 0x7: /* UMOV */
	      case 0x5: /* SMOV */
		  handle_umov_smov(s, insn);
		  break;
	      case 0x3: /* INS general reg */
		  handle_insg(s, insn);
		  break;
	      case 0x0: /* DUP element */
	      default:
		  goto unknown_insn;
	      }
        } else if (get_bits(insn, 21, 1) &&
                   get_bits(insn, 10, 1)) {
	    if (get_bits(insn, 11, 5) == 0x3)
	      handle_simdorr(s, insn);
	    else
	      handle_simd3su0(s, insn);
	} else if (get_bits(insn, 17, 5) == 0x18 &&
		   get_bits(insn, 11, 1) && !get_bits(insn, 10, 1)) {
	    handle_simd_accross(s, insn);
	} else if (get_bits(insn, 17, 5) == 0x10 &&
		   get_bits(insn, 11, 1) && !get_bits(insn, 10, 1)) {
	    handle_simd_misc(s, insn);
	} else if (!get_bits(insn, 21, 1) && !get_bits(insn, 15, 1) &&
		   !get_bits(insn, 10, 1)) {
	    if (get_bits(insn, 29, 1)) {
		/* AdvSIMD EXT */
		goto unknown_insn;
	    } else if (get_bits (insn, 11, 1)) {
		/* AdvSIMD ZIP/UZP/TRN */
		handle_simd_zip(s, insn);
	    } else {
		/* AdvSIMD TBL/TBX */
		if (get_bits(insn, 22, 2))
		  goto unknown_insn;
		else {
		    TCGv_i32 tcginsn = tcg_const_i32(insn);
		    gen_helper_simd_tbl(cpu_env, tcginsn);
		    tcg_temp_free_i32(tcginsn);
		}
	    }
        } else {
            goto unknown_insn;
        }
        break;
    case 0x0f:
        if (!get_bits(insn, 31, 1) && !get_bits(insn, 19, 5) &&
            (get_bits(insn, 10, 2) == 1)) {
            handle_simdmovi(s, insn);
	} else if (!get_bits(insn, 31, 1) && !get_bits(insn, 23, 1) &&
		   get_bits(insn, 10, 1)) {
	    handle_simd_shifti(s, insn);
        } else {
            goto unknown_insn;
        }
        break;
    case 0x10:
        handle_adr(s, insn);
        break;
    case 0x11:
        handle_addi(s, insn);
        break;
    case 0x12:
        if (get_bits(insn, 23, 1)) {
            handle_movi(s, insn);
        } else {
            handle_orri(s, insn);
        }
        break;
    case 0x13:
        if (get_bits(insn, 23, 1) && !get_bits(insn, 21, 1) &&
            !get_bits(insn, 29, 2)) {
            handle_extr(s, insn);
        } else {
            handle_bfm(s, insn);
        }
        break;
    case 0x14:
        if (get_bits(insn, 29, 3) == 0x6 && !get_bits(insn, 2, 3)) {
            handle_svc(s, insn);
            break;
        }
        if (get_bits(insn, 29, 2) == 0x1) {
            handle_cb(s, insn);
            break;
        }
        if (get_bits(insn, 29, 3) == 0x2) {
            handle_condb(s, insn);
            break;
        }
        goto unknown_insn;
    case 0x15:
        if (get_bits(insn, 29, 2) == 0x1) {
            handle_cb(s, insn);
            break;
        } else if (get_bits(insn, 20, 12) == 0xd53) {
            handle_mrs(s, insn);
            break;
        } else if (get_bits(insn, 20, 12) == 0xd51) {
            handle_msr(s, insn);
            break;
        } else if ((insn & 0xfffff01f) == 0xd503201f) {
            /* HINT instructions */
            break;
        } else if ((insn & 0xfffff09f) == 0xd503309f) {
            /* barrier instructions */
            break;
        } else if (get_bits(insn, 19, 13) == 0x1aa1) {
            handle_sys(s, insn);
            break;
        } else {
            goto unknown_insn;
        }
        break;
    case 0x16:
    case 0x17:
        if (get_bits(insn, 29, 2) == 0x1) {
            handle_tbz(s, insn);
            break;
        }
        goto unknown_insn;
    case 0x18:
    case 0x19:
    case 0x1c:
    case 0x1d:
        if (get_bits(insn, 29, 1)) {
            handle_ldst(s, insn);
        } else {
            handle_literal(s, insn);
        }
        break;
    case 0x1a:
	if (get_bits(insn, 21, 1))
	  goto unknown_insn;
	else switch (get_bits(insn, 22, 2)) {
	  case 0:
	      if (!get_bits(insn, 10, 6))
		handle_add(s, insn);
	      else
		goto unknown_insn;
	      break;
	  case 1:
	      handle_ccmp(s, insn);
	      break;
	  case 2:
	      handle_csel(s, insn);
	      break;
	  case 3:
	      if (get_bits(insn, 30, 1))
		handle_dp1s(s, insn);
	      else
		handle_dp2s(s, insn);
	      break;
	}
	break;
    case 0x1b:
        handle_dp3s(s, insn);
        break;
    case 0x1e:
        if (!get_bits(insn, 21, 1) && !get_bits(insn, 30, 1)) {
            handle_fpfpconv(s, insn);
        } else if (!get_bits(insn, 29, 3) && get_bits(insn, 21, 1) &&
                   (get_bits(insn, 5, 8) == 0x80)) {
            handle_fmovi(s, insn);
        } else if (get_bits(insn, 21, 1) && !get_bits(insn, 30, 1) &&
                   !get_bits(insn, 10, 6)) {
            handle_fpintconv(s, insn);
        } else if (!get_bits(insn, 29, 3) && get_bits(insn, 21, 1) &&
                   (get_bits(insn, 10, 6) == 0x8) && !get_bits(insn, 0, 3)) {
            handle_fcmp(s, insn);
        } else if (!get_bits(insn, 29, 3) && !get_bits(insn, 22, 2) &&
                   get_bits(insn, 21, 1) && (get_bits(insn, 10, 5) == 0x10)) {
            handle_fpdp1s32(s, insn);
        } else if (!get_bits(insn, 29, 3) && (get_bits(insn, 22, 2) == 0x1) &&
                   get_bits(insn, 21, 1) && (get_bits(insn, 10, 5) == 0x10)) {
            handle_fpdp1s64(s, insn);
        } else if (!get_bits(insn, 29, 3) && !get_bits(insn, 22, 2) &&
                   get_bits(insn, 21, 1) && (get_bits(insn, 10, 2) == 0x2)) {
            handle_fpdp2s32(s, insn);
        } else if (!get_bits(insn, 29, 3) && (get_bits(insn, 22, 2) == 0x1) &&
                   get_bits(insn, 21, 1) && (get_bits(insn, 10, 2) == 0x2)) {
            handle_fpdp2s64(s, insn);
        } else if ((get_bits(insn, 30, 2) == 0x1) && get_bits(insn, 21, 1) &&
                   (get_bits(insn, 10, 6) == 0x21)) {
            handle_v3add(s, insn);
        } else {
            goto unknown_insn;
        }
        break;
    case 0x1f:
        if (!get_bits(insn, 29, 3) && !get_bits(insn, 22, 2)) {
            handle_fpdp3s32(s, insn);
        } else if (!get_bits(insn, 29, 3) && (get_bits(insn, 22, 2) == 0x1)) {
            handle_fpdp3s64(s, insn);
        } else {
            goto unknown_insn;
        }
        break;
    default:
unknown_insn:
        fprintf(stderr, "unknown insn: %08x\n", insn);
        unallocated_encoding(s);
        break;
    }

    if (unlikely(s->singlestep_enabled) && (s->is_jmp == DISAS_TB_JUMP)) {
        /* go through the main loop for single step */
        s->is_jmp = DISAS_JUMP;
    }

#ifdef DEBUG_FLUSH
    if (s->is_jmp)
        gen_helper_tb_flush(cpu_env);
#endif
}
