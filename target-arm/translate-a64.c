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

#include "helper.h"
#define GEN_HELPER 1
#include "helper.h"

static TCGv_i64 cpu_X[32];
static TCGv_i64 cpu_pc;
static TCGv_i64 cpu_sp;
static TCGv_i32 pstate;

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
    TCGv tmp = tcg_temp_new_i32();
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
    if (unlikely(s->singlestep_enabled)) {
        s->is_jmp = DISAS_JUMP;
    } else {
        s->is_jmp = DISAS_TB_JUMP;
    }
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
        source = 30; /* XXX spec says "if absent"? */
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
    uint64_t addr = s->pc - 4 + (get_sbits(insn, 0, 19) << 2);
    int cond = get_bits(insn, 0, 4);
    int no_match;
    TCGv_i32 tcg_zero = tcg_const_i32(0);
    TCGv_i32 tcg_cond = tcg_const_i32(cond);
    TCGv_i32 tcg_condmatch = tcg_temp_new_i32();

    no_match = gen_new_label();

    gen_helper_cond(tcg_condmatch, pstate, tcg_cond);
    tcg_gen_brcond_i64(TCG_COND_NE, tcg_condmatch, tcg_zero, no_match);

    gen_a64_set_pc_im(addr);
    tcg_gen_exit_tb(0);

    gen_set_label(no_match);
    gen_a64_set_pc_im(s->pc);

    tcg_temp_free_i32(tcg_zero);
    tcg_temp_free_i32(tcg_cond);
    tcg_temp_free_i32(tcg_condmatch);

    s->is_jmp = DISAS_JUMP;
}

static void handle_cb(DisasContext *s, uint32_t insn)
{
    uint64_t addr = s->pc - 4 + (get_sbits(insn, 5, 19) << 2);
    bool is_zero = get_bits(insn, 24, 1);
    int dest = get_reg(insn);
    int no_match;

    no_match = gen_new_label();
    if (is_zero) {
        tcg_gen_brcond_i64(TCG_COND_NE, cpu_reg(dest), cpu_reg(31), no_match);
    } else {
        tcg_gen_brcond_i64(TCG_COND_EQ, cpu_reg(dest), cpu_reg(31), no_match);
    }
    gen_a64_set_pc_im(addr);
    tcg_gen_exit_tb(0);

    gen_set_label(no_match);
    gen_a64_set_pc_im(s->pc);

    s->is_jmp = DISAS_JUMP;
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
        tcg_gen_movi_i64(cpu_reg(reg), imm);
    }

    if (is_n) {
        tcg_gen_not_i32(cpu_reg(reg), cpu_reg(reg));
    }

    if (is_32bit) {
        tcg_gen_andi_i64(cpu_reg(reg), cpu_reg(reg), 0xffffffff);
    }
}

static void handle_orri(DisasContext *s, uint32_t insn)
{
    fprintf(stderr, "XXX orri\n");
    unallocated_encoding(s);
}

static TCGv_i64 get_shift(int reg, int shift_type, int shift)
{
    TCGv_i64 r;

    r = tcg_temp_new_i64();

    if (!shift) {
        tcg_gen_mov_i64(r, cpu_reg(reg));
        return r;
    }

    /* XXX carry_out */
    switch (shift_type) {
    case 0: /* LSL */
        tcg_gen_shli_i64(r, cpu_reg(reg), shift);
        break;
    case 1: /* LSR */
        tcg_gen_shri_i64(r, cpu_reg(reg), shift);
        break;
    case 2: /* ASR */
        tcg_gen_sari_i64(r, cpu_reg(reg), shift);
        break;
    case 3:
        tcg_gen_rotri_i64(r, cpu_reg(reg), shift);
        break;
    }

    return r;
}

static void handle_orr(DisasContext *s, uint32_t insn)
{
    int is_32bit = !get_bits(insn, 31, 1);
    int dest = get_reg(insn);
    int source = get_bits(insn, 5, 5);
    int rm = get_bits(insn, 16, 5);
    int shift_amount = get_sbits(insn, 10, 6);
    int is_n = !get_bits(insn, 21, 1);
    int shift_type = !get_bits(insn, 22, 2);
    int opc = !get_bits(insn, 22, 29);
    bool setflags = (opc == 0x3);
    TCGv_i64 tcg_op2;

    if (is_32bit && (shift_amount < 0)) {
        /* reserved value */
        tcg_abort();
    }

    if (is_32bit) {
        /* XXX handle 32 bit case */
        tcg_abort();
    }

    /* MOV is dest = xzr & (source & ~0) */
    if (!shift_amount && is_n && opc == 0x0 && source == 0x1f) {
        tcg_gen_mov_i64(cpu_reg_sp(dest), cpu_reg(rm));
        return;
    }

    tcg_op2 = get_shift(rm, shift_type, shift_amount);
    if (is_n) {
        tcg_gen_neg_i64(tcg_op2, tcg_op2);
    }

    switch (opc) {
    case 0x0:
    case 0x3:
        tcg_gen_and_i64(cpu_reg_sp(dest), cpu_reg(source), tcg_op2);
        break;
    case 0x1:
        tcg_gen_or_i64(cpu_reg_sp(dest), cpu_reg(source), tcg_op2);
        break;
    case 0x2:
        tcg_gen_xor_i64(cpu_reg_sp(dest), cpu_reg(source), tcg_op2);
        break;
    }

    if (setflags) {
        /* XXX set PSTATE.N,Z,C,V */
        tcg_abort();
    }

    tcg_temp_free_i64(tcg_op2);
}

static void handle_stp(DisasContext *s, uint32_t insn)
{
    int rt = get_reg(insn);
    int rn = get_bits(insn, 5, 5);
    int rt2 = get_bits(insn, 10, 5);
    int offset = get_sbits(insn, 15, 7) << 2;
    int is_load = get_bits(insn, 22, 1);
    int is_vector = get_bits(insn, 26, 1);
    int is_signed = get_bits(insn, 30, 1);
    int is_32bit = !get_bits(insn, 31, 1);
    int type = !get_bits(insn, 23, 2);
    TCGv_i64 tcg_addr;
    bool postindex;
    bool wback;

    switch (type) {
    case 0:
        postindex = false;
        wback = false;
        break;
    case 1:
        postindex = true;
        wback = true;
        break;
    case 2:
        unallocated_encoding(s);
        return;
    case 3:
        postindex = false;
        wback = true;
        break;
    }

    if (is_signed && !is_32bit) {
        unallocated_encoding(s);
        return;
    }

    switch (get_bits(insn, 23, 2)) {
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
    default:
        unallocated_encoding(s);
        return;
    }

    if (is_vector) {
        tcg_abort();
    }

    if (!is_32bit) {
        offset <<= 1;
    }

    tcg_addr = tcg_temp_new_i64();
    if (rn == 31) {
        /* XXX check SP alignment */
    }
    tcg_gen_mov_i64(tcg_addr, cpu_reg_sp(rn));

    if (!postindex) {
        tcg_gen_addi_i64(tcg_addr, tcg_addr, offset);
    }

    if (is_load) {
        if (is_32bit) {
            tcg_gen_qemu_ld32u(cpu_reg(rt), tcg_addr, get_mem_index(s));
            tcg_gen_addi_i64(tcg_addr, tcg_addr, 4);
            tcg_gen_qemu_ld32u(cpu_reg(rt2), tcg_addr, get_mem_index(s));
            tcg_gen_addi_i64(tcg_addr, tcg_addr, -4);
        } else {
            tcg_gen_qemu_ld64(cpu_reg(rt), tcg_addr, get_mem_index(s));
            tcg_gen_addi_i64(tcg_addr, tcg_addr, 8);
            tcg_gen_qemu_ld64(cpu_reg(rt2), tcg_addr, get_mem_index(s));
            tcg_gen_addi_i64(tcg_addr, tcg_addr, -8);
        }
    } else {
        if (is_32bit) {
            tcg_gen_qemu_st32(cpu_reg(rt), tcg_addr, get_mem_index(s));
            tcg_gen_addi_i64(tcg_addr, tcg_addr, 4);
            tcg_gen_qemu_st32(cpu_reg(rt2), tcg_addr, get_mem_index(s));
            tcg_gen_addi_i64(tcg_addr, tcg_addr, -4);
        } else {
            tcg_gen_qemu_st64(cpu_reg(rt), tcg_addr, get_mem_index(s));
            tcg_gen_addi_i64(tcg_addr, tcg_addr, 8);
            tcg_gen_qemu_st64(cpu_reg(rt2), tcg_addr, get_mem_index(s));
            tcg_gen_addi_i64(tcg_addr, tcg_addr, -8);
        }
    }

    if (wback) {
        if (postindex) {
            tcg_gen_addi_i64(tcg_addr, tcg_addr, offset);
        }
        tcg_gen_mov_i64(cpu_reg_sp(rn), tcg_addr);
    }

    tcg_temp_free_i64(tcg_addr);
}

static void ldst_calc_index(DisasContext *s, TCGv_i64 tcg_addr,
                            bool is_reg_offset, int offset, int size)
{
    bool is_shift = get_bits(offset, 0, 1);
    int shift = size;
    int extsize = get_bits(offset, 1, 2);
    int rm = get_bits(offset, 4, 5);
    bool is_signed = get_bits(offset, 3, 1);

    if (!is_reg_offset) {
        tcg_gen_addi_i64(tcg_addr, tcg_addr, offset);
        return;
    }

    /* offset in register */
    if (!(extsize & 1)) {
        unallocated_encoding(s);
        return;
    }

    if (!is_shift) {
        shift = 0;
    }

    if (is_signed) {
        switch (extsize) {
        case 0:
            tcg_gen_ext8s_i64(tcg_addr, cpu_reg(rm));
            break;
        case 1:
            tcg_gen_ext16s_i64(tcg_addr, cpu_reg(rm));
            break;
        case 2:
            tcg_gen_ext32s_i64(tcg_addr, cpu_reg(rm));
            break;
        }
    } else {
        switch (extsize) {
        case 0:
            tcg_gen_ext8u_i64(tcg_addr, cpu_reg(rm));
            break;
        case 1:
            tcg_gen_ext16u_i64(tcg_addr, cpu_reg(rm));
            break;
        case 2:
            tcg_gen_ext32u_i64(tcg_addr, cpu_reg(rm));
            break;
        }
    }

    if (shift) {
        tcg_gen_shli_i64(tcg_addr, tcg_addr, shift);
    }
}

static void handle_literal(DisasContext *s, uint32_t insn)
{
    int dest = get_reg(insn);
    int opc = get_bits(insn, 30, 2);
    int64_t imm = get_sbits(insn, 5, 19) << 2;
    TCGv_i64 tcg_addr;

    tcg_addr = tcg_const_i64((s->pc - 4) + imm);

    switch (opc) {
    case 0:
        tcg_gen_qemu_ld32u(cpu_reg(dest), tcg_addr, get_mem_index(s));
        break;
    case 1:
        tcg_gen_qemu_ld64(cpu_reg(dest), tcg_addr, get_mem_index(s));
        break;
    case 2:
        tcg_gen_qemu_ld32s(cpu_reg(dest), tcg_addr, get_mem_index(s));
        break;
    case 3:
        /* prefetch */
        break;
    }

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

    if (!opc1) {
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

    if (postindex) {
        ldst_calc_index(s, tcg_addr, is_reg_offset, offset, size);
    }

    if (wback) {
        tcg_gen_mov_i64(cpu_reg_sp(source), tcg_addr);
    }

    tcg_temp_free_i64(tcg_addr);
}

static void handle_extr(DisasContext *s, uint32_t insn)
{
    unallocated_encoding(s);
}

static void handle_bfm(DisasContext *s, uint32_t insn)
{
    unallocated_encoding(s);
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

    if (sub_op) {
        imm = -imm;
    }

    tcg_imm = tcg_const_i64(imm);
    tcg_gen_add_i64(tcg_result, cpu_reg_sp(source), tcg_imm);

    if (setflags) {
        if (sub_op) {
            if (is_32bit) {
                gen_helper_pstate_addc32(pstate, pstate, cpu_reg_sp(source),
                                         tcg_imm, tcg_result);
            } else {
                gen_helper_pstate_addc(pstate, pstate, cpu_reg_sp(source),
                                       tcg_imm, tcg_result);
            }
        } else {
            if (is_32bit) {
                gen_helper_pstate_add32(pstate, pstate, cpu_reg_sp(source),
                                        tcg_imm, tcg_result);
            } else {
                gen_helper_pstate_add(pstate, pstate, cpu_reg_sp(source),
                                      tcg_imm, tcg_result);
            }
        }
    }

    if (is_32bit) {
        tcg_gen_ext32u_i64(cpu_reg_sp(dest), tcg_result);
    } else {
        tcg_gen_mov_i64(cpu_reg_sp(dest), tcg_result);
    }
}

static void handle_svc(DisasContext *s, uint32_t insn)
{
    gen_a64_set_pc_im(s->pc);
#define DISAS_SWI 5
    s->is_jmp = DISAS_SWI;
}

void disas_a64_insn(CPUARMState *env, DisasContext *s)
{
    uint32_t insn;

    insn = arm_ldl_code(env, s->pc, s->bswap_code);
    s->pc += 4;

    // XXX singlestep hack
    gen_a64_set_pc_im(s->pc);
    s->is_jmp = DISAS_JUMP;
    //////////////////////

    printf("insn: %08x\n", insn);

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
    case 0x08:
    case 0x09:
        reserved(s, insn, 29, 1, 1);
        handle_stp(s, insn);
        break;
    case 0x0D:
        reserved(s, insn, 29, 1, 1);
        handle_stp(s, insn);
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
        if (get_bits(insn, 23, 1)) {
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
        }
        goto unknown_insn;
    case 0x18:
    case 0x19:
        if (get_bits(insn, 29, 1)) {
            handle_ldst(s, insn);
        } else {
            handle_literal(s, insn);
        }
        break;
    default:
unknown_insn:
        printf("unknown insn: %08x\n", insn);
        unallocated_encoding(s);
        break;
    }
}
