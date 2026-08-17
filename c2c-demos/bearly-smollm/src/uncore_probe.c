/*
 * What can still be changed on taped-out silicon: the memory system's CONFIGURATION registers.
 *
 * Decode is pinned at ~55 cycles per weight byte (~13.7 MB/s), which is ~100x worse than any real
 * DRAM should manage, and the kernels are already at 91% of the cost of merely touching the bytes.
 * No software rewrite fixes that. But several of the things that could cause it are registers that
 * boot firmware is normally expected to program and that nothing in this repo writes:
 *
 *   - the SiFive inclusive-cache controller's WayEnable (0x02010008) resets to 0, which enables ONE
 *     way. Firmware is supposed to write ways-1. If that never happened, most of the L2 is dark.
 *   - SiFive cores carry a "feature disable" CSR (0x7C1) whose reset value turns OFF things like L1
 *     prefetching; firmware clears it.
 *
 * Both are single writes, on real silicon, today.
 *
 * DANGER, and why this file prints before it pokes: on this platform a read of an unmapped address
 * HANGS THE CORE SILENTLY rather than trapping (see .claude/plans/009-silicon-rtl-bug-list.md #4),
 * and an unimplemented CSR raises an illegal instruction. Every access below is announced first, so
 * if the console stops the last line printed names the register that killed it. Each group is
 * separately compile-gated so you can advance past a hang without guessing.
 */

#include <stdio.h>
#include <stdint.h>

#include "chip_config.h"
#include "smollm_config.h"

#if SMOLLM_PROBE_UNCORE

#ifndef SMOLLM_PROBE_CCACHE
#define SMOLLM_PROBE_CCACHE 1
#endif
#ifndef SMOLLM_PROBE_FEATURE_CSR
#define SMOLLM_PROBE_FEATURE_CSR 0    /* off by default: an unimplemented CSR traps */
#endif
#ifndef SMOLLM_CCACHE_BASE
#define SMOLLM_CCACHE_BASE 0x02010000UL
#endif

void smollm_probe_uncore(void) {
  SMOLLM_LOG("[probe] --- uncore configuration (each access is announced BEFORE it happens) ---\r\n");

  /* Core identity first: these CSRs are mandatory, so they cannot hang. */
  uint64_t misa = 0, marchid = 0, mimpid = 0, mvendorid = 0;
  __asm__ volatile("csrr %0, misa" : "=r"(misa));
  __asm__ volatile("csrr %0, marchid" : "=r"(marchid));
  __asm__ volatile("csrr %0, mimpid" : "=r"(mimpid));
  __asm__ volatile("csrr %0, mvendorid" : "=r"(mvendorid));
  SMOLLM_LOG("[probe] misa=0x%llx marchid=0x%llx mimpid=0x%llx mvendorid=0x%llx\r\n",
             (unsigned long long)misa, (unsigned long long)marchid,
             (unsigned long long)mimpid, (unsigned long long)mvendorid);

#if SMOLLM_PROBE_CCACHE
  /* SiFive inclusive cache: Config at +0x000, WayEnable at +0x008.
   * Config packs [7:0] banks, [15:8] ways, [23:16] lgSets, [31:24] lgBlockBytes. */
  SMOLLM_LOG("[probe] about to READ ccache Config at 0x%lx ...\r\n", (unsigned long)SMOLLM_CCACHE_BASE);
  const uint32_t cfg = *(volatile uint32_t *)(uintptr_t)SMOLLM_CCACHE_BASE;
  const unsigned banks = cfg & 0xFF, ways = (cfg >> 8) & 0xFF;
  const unsigned lg_sets = (cfg >> 16) & 0xFF, lg_block = (cfg >> 24) & 0xFF;
  SMOLLM_LOG("[probe] ccache Config=0x%08lx -> banks=%u ways=%u sets=%u block=%u B => %lu KB total\r\n",
             (unsigned long)cfg, banks, ways, 1u << lg_sets, 1u << lg_block,
             (unsigned long)(((uint64_t)banks * ways * (1u << lg_sets) * (1u << lg_block)) >> 10));

  SMOLLM_LOG("[probe] about to READ ccache WayEnable at 0x%lx ...\r\n",
             (unsigned long)(SMOLLM_CCACHE_BASE + 8));
  const uint32_t we = *(volatile uint32_t *)(uintptr_t)(SMOLLM_CCACHE_BASE + 8);
  /* WayEnable reads 0 on this part and ignores writes. That is NOT a disabled cache: this is the
   * OPEN-SOURCE SiFive cache, where all ways are always active and WayEnable is not the gating
   * control it is on the commercial IP. The full 256 KB is in use. */
  SMOLLM_LOG("[probe] ccache WayEnable=%lu (open-source ccache: all %u ways are active regardless; "
             "this register is not the gate)\r\n", (unsigned long)we, ways);

#if SMOLLM_PROBE_CCACHE_ENABLE_WAYS
  /* Enabling ways is a one-way door on SiFive parts (ways cannot be disabled again without reset),
   * hence its own switch. Write ways-1, then read it back. */
  if (ways > 1) {
    SMOLLM_LOG("[probe] about to WRITE WayEnable = %u (expected to be ignored) ...\r\n", ways - 1);
    *(volatile uint32_t *)(uintptr_t)(SMOLLM_CCACHE_BASE + 8) = ways - 1;
    const uint32_t we2 = *(volatile uint32_t *)(uintptr_t)(SMOLLM_CCACHE_BASE + 8);
    SMOLLM_LOG("[probe] WayEnable now %lu — re-run the bench to see whether it moved\r\n",
               (unsigned long)we2);
  }
#endif
#endif /* SMOLLM_PROBE_CCACHE */

#if SMOLLM_PROBE_FEATURE_CSR
  /* SiFive "Feature Disable" CSR. Unimplemented on non-SiFive cores => illegal instruction, which
   * is why this is opt-in. Bits set here mean features are OFF. */
  SMOLLM_LOG("[probe] about to READ CSR 0x7c1 (SiFive feature-disable) ...\r\n");
  uint64_t fd = 0;
  __asm__ volatile("csrr %0, 0x7c1" : "=r"(fd));
  SMOLLM_LOG("[probe] CSR 0x7c1 = 0x%llx %s\r\n", (unsigned long long)fd,
             fd ? "   <-- non-zero: features are disabled, firmware normally clears this" : "");
#endif

  SMOLLM_LOG("[probe] --- uncore probe done ---\r\n");
}

#else
void smollm_probe_uncore(void) { }
#endif
