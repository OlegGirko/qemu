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

static const char *regnames[] =
    { "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7",
      "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15",
      "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23",
      "x24", "x25", "x26", "x27", "x28", "x29", "lr", "sp" };

/* initialize TCG globals.  */
void a64_translate_init(void)
{
    int i;

    cpu_pc = tcg_global_mem_new_i64(TCG_AREG0,
                                    offsetof(CPUARMState, pc),
                                    "pc");
    for (i = 0; i < 32; i++) {
        cpu_X[i] = tcg_global_mem_new_i64(TCG_AREG0,
                                          offsetof(CPUARMState, xregs[i]),
                                          regnames[i]);
    }
}

void cpu_dump_state_a64(CPUARMState *env, FILE *f, fprintf_function cpu_fprintf,
                        int flags)
{
    int i;

    cpu_fprintf(f, "PC=%016"PRIx64"\n", env->pc);
    for(i = 0; i < 32; i++) {
        cpu_fprintf(f, "X%02d=%016"PRIx64, i, env->xregs[i]);
        if ((i % 4) == 3)
            cpu_fprintf(f, "\n");
        else
            cpu_fprintf(f, " ");
    }
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
        r |= ((1ULL << (32 - len)) - 1) << len;
    }
    return r;
}

static int get_reg(uint32_t inst)
{
    return get_bits(inst, 0, 5);
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

static void handle_b(CPUARMState *env, DisasContext *s, uint32_t insn)
{
    uint64_t addr = s->pc + (get_bits(insn, 0, 26) << 2);

    gen_a64_set_pc_im(addr - 4);
    s->is_jmp = DISAS_JUMP;
}

static void handle_bl(CPUARMState *env, DisasContext *s, uint32_t insn)
{
    tcg_gen_movi_i64(cpu_X[30], s->pc);
    handle_b(env, s, insn);
}

/* PC relative address calculation */
static void handle_adr(CPUARMState *env, DisasContext *s, uint32_t insn)
{
    int reg = get_reg(insn);
    uint64_t imm;

    imm = get_sbits(insn, 5, 19) << 2;
    imm |= get_bits(insn, 29, 2);

    if (insn & 0x80000000) {
        /* ADRP (page based) */
        tcg_gen_movi_i64(cpu_X[reg], s->pc & ~0xfffULL);
        imm <<= 12;
    } else {
        tcg_gen_movi_i64(cpu_X[reg], s->pc);
    }

    tcg_gen_addi_i64(cpu_X[reg], cpu_X[reg], imm);

    /*
      [0..4] = target reg
      [5..23] = imm high
      [29..30] = imm low
    
      addr = sign_extend([imm high] [imm low])
      reg = pc + addr
      if (page)
          reg &= ~0xfff
    */

}

static void handle_movi(CPUARMState *env, DisasContext *s, uint32_t insn)
{
    int reg = get_reg(insn);
    uint64_t imm;

    /* XXX reserved bits */

    imm = get_bits(insn, 5, 16);
    /* XXX multiply logic */
    /* XXX flavors (movz, mov, ...) */

    tcg_gen_movi_i64(cpu_X[reg], imm);
}

static void handle_mov(CPUARMState *env, DisasContext *s, uint32_t insn)
{
    int dest = get_reg(insn);
    int source = get_bits(insn, 10, 5);

    /* XXX reserved bits */
    /* XXX flavors */

    tcg_gen_mov_i64(cpu_X[dest], cpu_X[source]);
}

static void handle_stp(CPUARMState *env, DisasContext *s, uint32_t insn)
{
    int x1 = get_reg(insn);
    int dest = get_bits(insn, 5, 5);
    int x2 = get_bits(insn, 10, 5);
    int offset = get_sbits(insn, 15, 7) << 3;
    TCGv_i64 tcg_addr;

    /* XXX reserved bits */
    /* XXX flavors */

    tcg_addr = tcg_temp_new_i64();
    tcg_gen_addi_i64(tcg_addr, cpu_X[dest], offset);
    tcg_gen_qemu_st64(tcg_addr, cpu_X[x1], 1);
    tcg_gen_addi_i64(tcg_addr, tcg_addr, 8);
    tcg_gen_qemu_st64(tcg_addr, cpu_X[x2], 1);
    tcg_temp_free_i64(tcg_addr);
}

static void handle_ldst(CPUARMState *env, DisasContext *s, uint32_t insn)
{
    int dest = get_reg(insn);
    int source = get_bits(insn, 5, 5); /* XXX probably more bits */
    int index = get_bits(insn, 12, 9);
    TCGv_i64 tcg_addr;
    bool is_relative = !((insn >> 24) & 0x1);
    bool is_store = !(insn & 0x00400000);

    /* XXX reserved bits */
    /* XXX flavors */
    /* XXX different bit width */

    if ((insn & 0xbf000000) == 0x18000000) {
        // XXX find pattern
        is_store = 0;
    }

    if ((insn & 0xbf800000) == 0xb9000000) {
        /* UIMM12 version */
        index = get_bits(insn, 10, 12);
    } else {
        /* SIMM9 version */
        index = get_sbits(insn, 12, 9);
    }

    if (is_relative) {
        int rel = get_bits(insn, 5, 19);
        tcg_addr = tcg_const_i64((rel << 2) + s->pc - 4);
    } else {
        tcg_addr = tcg_temp_new_i64();
        tcg_gen_mov_i64(tcg_addr, cpu_X[source]);
    }

    if (index) {
        tcg_gen_addi_i64(tcg_addr, tcg_addr, index);
    }

    if (is_store) {
        tcg_gen_qemu_st64(cpu_X[dest], tcg_addr, 1);
    } else {
        tcg_gen_qemu_ld64(cpu_X[dest], tcg_addr, 1);
    }

    tcg_temp_free_i64(tcg_addr);
}

static void handle_add(CPUARMState *env, DisasContext *s, uint32_t insn)
{
    int dest = get_reg(insn);
    int source = get_bits(insn, 5, 5);
    int is_64bit = get_bits(insn, 30, 1);
    int shift = get_bits(insn, 22, 2);
    uint64_t imm;

    /* XXX check reserved bits */

    imm = get_bits(insn, 10, 12);
    switch (shift) {
    case 0x0:
        break;
    case 0x1:
        imm <<= 12;
        break;
    default:
        tcg_abort();
    }

    /* XXX check is_64bit */
    if (is_64bit) {
    }

    tcg_gen_addi_i64(cpu_X[dest], cpu_X[source], imm);
}

static void handle_svc(CPUARMState *env, DisasContext *s, uint32_t insn)
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

    printf("insn: %08x\n", insn);

    /* One-off branch instruction layout */
    switch ((insn & 0xfc000000) >> 26) {
    case 0x5:
        handle_b(env, s, insn);
        return;
    case 0x25:
        handle_bl(env, s, insn);
        return;
    }

    /* Typical major opcode encoding */
    switch ((insn >> 24) & 0x1f) {
    case 0x0a:
        handle_mov(env, s, insn);
        break;
    case 0x09:
        handle_stp(env, s, insn);
        break;
    case 0x10:
        handle_adr(env, s, insn);
        break;
    case 0x11:
        handle_add(env, s, insn);
        break;
    case 0x12:
        handle_movi(env, s, insn);
        break;
    case 0x14:
        handle_svc(env, s, insn);
        break;
    case 0x18:
    case 0x19:
        handle_ldst(env, s, insn);
        break;
    default:
        printf("unknown insn: %08x\n", insn);
        gen_exception_insn(s, 4, EXCP_UDEF);
        break;
    }
}
