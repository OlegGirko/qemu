#include "disas/bfd.h"

#define INSNLEN 4

/* Stub disassembler for aarch64.  */

int
print_insn_aarch64 (bfd_vma pc,
		    struct disassemble_info *info)
{
  bfd_byte	buffer[INSNLEN];
  int		status;
  unsigned int	size = 4;
  uint32_t	data;

  /* Aarch64 instructions are always little-endian */
  info->endian = BFD_ENDIAN_LITTLE;
  info->bytes_per_chunk = size = INSNLEN;
  info->display_endian = info->endian;

  status = (*info->read_memory_func) (pc, buffer, size, info);
  if (status != 0)
    {
      (*info->memory_error_func) (status, pc, info);
      return -1;
    }

  data = ldl_p(buffer);

  (*info->fprintf_func) (info->stream,"\t[0x%08x] (%02x)\t.unknown", data, (data >> 24) & 0x1f);

  return size;
}
