/*
 * membw — memory latency, memory-level parallelism and bandwidth sweep for DSP 25 / Bearly ML 25.
 *
 * The question this is built to answer: CAN a faster off-chip link ever be visible from the core,
 * and if so at what operating point? Little's Law fixes the shape — BW = lines_in_flight * 64 B /
 * RTT — so the answer turns entirely on two measured numbers and nothing else:
 *
 *   RTT              the flat tail of the pointer-chase curve (section 3)
 *   lines in flight  how many misses the core actually keeps outstanding (section 4) — THE GATE
 *
 * If the core sustains one miss at a time, no wire clock and no software change can saturate the
 * link from the core, and an ILA on the serial TL port counting `out.valid && out.ready` is the only
 * way to see a difference. Section 6 prints both numbers and the bar to clear, so the run itself
 * says which world you are in.
 *
 * Three deliberate design choices, each because the obvious version misleads:
 *
 *   - MLP is measured with N INDEPENDENT pointer chases, not N interleaved sequential streams. A
 *     sequential stream mixes in the prefetcher, the row buffer and the loop's own scheduling; N
 *     chains offer the memory system exactly N independent misses and nothing else.
 *   - The interleaved-stream test is kept but reported in TWO forms, dependent and independent. The
 *     dependent form consumes each load with the very next instruction, so an in-order core cannot
 *     overlap the misses no matter how many MSHRs the D$ has — that alone produces the "parallelism
 *     makes it worse" shape, with no hardware defect required. Printing both separates the loop from
 *     the hardware.
 *   - Every parallelism figure is a RATIO OF CYCLE COUNTS, so it stays valid while the core clock is
 *     in doubt. Only the ns and MB/s columns need the clock, and section 1 establishes that
 *     separately (see c2c-demos/common/clock_probe.h — the naive rdcycle-vs-mtime measurement is
 *     WRONG on this silicon, and it is wrong by 15x in the direction that changes the conclusion).
 *
 * There are no page tables here: this is bare M-mode and nothing in the runtime ever writes `satp`
 * (checked across glossy/, platform/ and bmark-lib/). So no knee in the latency curve can be a TLB
 * or page-walk effect, and huge pages are not available to try — the 256 KB -> 1 MB cliff is the
 * last-level cache boundary.
 *
 * Build (either chip — it touches no accelerator, only DRAM):
 *   make build CHIP=dsp25    PLATFORM=CHIP TARGET=membw EXTRA_CMAKE_ARGS="-DLINKER=chip"
 *   make build CHIP=bearly25 PLATFORM=CHIP TARGET=membw EXTRA_CMAKE_ARGS="-DLINKER=chip"
 *   make tsi-run TTY=<tty> BINARY=build/c2c-demos/membw/membw.elf
 */

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#include "chip_config.h"
#include "clint.h"
#include "uart.h"
#include "simple_setup.h"   /* init_test(): portable PLL + UART bring-up */

#include "membw_config.h"
#include "mem_probe.h"
#include "clock_probe.h"

uint64_t target_frequency = MEMBW_TARGET_FREQ_HZ;

static clock_probe_t g_clk;
static uint64_t g_core_hz;          /* resolved by clock_probe, NOT the naive mtime measurement */
static uint64_t g_sink;             /* keeps every load live */
static uint32_t g_isink;

/* Headline numbers, captured on the last pass and reported together in section 6. */
static uint64_t g_rtt_cyc;          /* dependent-load latency at the largest span */
static uint64_t g_read_cpb_x100;    /* best sustained streaming read cost */
static uint64_t g_mlp_x100;         /* overlapping misses the core actually achieved */

static uint8_t *const g_buf = (uint8_t *)MEMBW_BUF_ADDR;
static const size_t g_half = MEMBW_BUF_BYTES / 2u;

_Static_assert(MEMBW_MAX_SPAN <= MEMBW_BUF_BYTES / 2u,
               "MEMBW_MAX_SPAN must fit in half the window (copy needs a distinct src and dst)");
_Static_assert(MEMBW_STREAM_SPAN <= MEMBW_MAX_SPAN, "MEMBW_STREAM_SPAN must fit in MEMBW_MAX_SPAN");
_Static_assert(MEMBW_MLP_SUB_SPAN * 8u <= MEMBW_MAX_SPAN,
               "8 independent chase chains must fit in MEMBW_MAX_SPAN");
_Static_assert((MEMBW_BUF_BYTES % (2u * MEM_PROBE_LINE_BYTES)) == 0u,
               "window must be a whole number of cache lines in each half");

static uint64_t mtime_now(void) { return clint_get_time(CLINT); }

/* ------------------------------------------------------------------------------------------------
 * 2. Boot check
 *
 * A load or store to an address this SoC does not back HANGS the core silently — no trap, no print
 * (CLAUDE.md). Before writing 128 MB, touch four words and announce each one first, so if the window
 * is not backed on this board the console says which access killed it instead of just stopping.
 * ---------------------------------------------------------------------------------------------- */
static int window_check(void) {
  volatile uint32_t *lo = (volatile uint32_t *)g_buf;
  volatile uint32_t *hi = (volatile uint32_t *)(g_buf + MEMBW_BUF_BYTES - 4u);

  MEMBW_LOG("[membw] window [0x%08lx, 0x%08lx) = %lu MB\r\n",
            (unsigned long)MEMBW_BUF_ADDR, (unsigned long)(MEMBW_BUF_ADDR + MEMBW_BUF_BYTES),
            (unsigned long)(MEMBW_BUF_BYTES >> 20));

  MEMBW_LOG("[membw] probing first word 0x%08lx ... ", (unsigned long)MEMBW_BUF_ADDR);
  fflush(stdout);
  *lo = 0xA5A5F00Du;
  const uint32_t glo = *lo;
  MEMBW_LOG("%s\r\n", (glo == 0xA5A5F00Du) ? "ok" : "READ BACK WRONG");

  MEMBW_LOG("[membw] probing last  word 0x%08lx ... ",
            (unsigned long)(MEMBW_BUF_ADDR + MEMBW_BUF_BYTES - 4u));
  fflush(stdout);
  *hi = 0x5A5A0FF0u;
  const uint32_t ghi = *hi;
  MEMBW_LOG("%s\r\n", (ghi == 0x5A5A0FF0u) ? "ok" : "READ BACK WRONG");

  /* Aliasing check: if the window wraps onto itself (a smaller DRAM than declared), writing the top
   * word will have clobbered the bottom one. That is exactly how a too-large window fails on a
   * board with less memory than assumed, and it would otherwise look like a bandwidth anomaly. */
  const uint32_t again = *lo;
  if (again != 0xA5A5F00Du) {
    MEMBW_LOG("[membw] FATAL: first word changed to 0x%08lx after writing the last word — the "
              "window ALIASES (real DRAM is smaller than %lu MB). Lower MEMBW_BUF_BYTES.\r\n",
              (unsigned long)again, (unsigned long)(MEMBW_BUF_BYTES >> 20));
    return -1;
  }
  if (glo != 0xA5A5F00Du || ghi != 0x5A5A0FF0u) {
    MEMBW_LOG("[membw] FATAL: window does not hold data — not backed DRAM.\r\n");
    return -1;
  }
  MEMBW_LOG("[membw] window ok (no aliasing)\r\n");
  return 0;
}

/* ------------------------------------------------------------------------------------------------
 * 3. Latency curve
 *
 * Finer than a power-of-four sweep so the knees are actually locatable: L1, whatever sits between
 * L1 and the last-level cache, the LLC boundary, and the flat off-chip tail. The tail is the RTT
 * that Little's Law needs.
 * ---------------------------------------------------------------------------------------------- */
static void sweep_latency(void) {
  static const size_t spans[] = {
    4u << 10, 8u << 10, 16u << 10, 32u << 10, 64u << 10, 128u << 10, 256u << 10, 512u << 10,
    1u << 20, 2u << 20, 4u << 20, 16u << 20, 64u << 20,
  };
  MEMBW_LOG("[membw] --- 3. latency: random dependent loads, one per %u B line ------------------\r\n",
            (unsigned)MEM_PROBE_LINE_BYTES);
  for (size_t k = 0; k < sizeof(spans) / sizeof(spans[0]); k++) {
    const size_t span = spans[k];
    if (span > MEMBW_MAX_SPAN) break;
    if (mem_probe_chase_build(g_buf, span, 0xBEEF0001u + (uint32_t)k) == 0u) continue;
    mem_probe_chase_run(g_buf, 2000u, &g_isink);   /* warm: a small span should be resident */
    const uint64_t c = mem_probe_chase_run(g_buf, MEMBW_LAT_STEPS, &g_isink);
    char lbl[32];
    if (span >= (1u << 20)) snprintf(lbl, sizeof(lbl), "chase %4lu MB", (unsigned long)(span >> 20));
    else                    snprintf(lbl, sizeof(lbl), "chase %4lu KB", (unsigned long)(span >> 10));
    mem_probe_print_lat(lbl, c, MEMBW_LAT_STEPS, g_core_hz);
    if (span >= (16u << 20)) g_rtt_cyc = (c * 100ull) / MEMBW_LAT_STEPS / 100ull;   /* the tail */
  }
}

/* ------------------------------------------------------------------------------------------------
 * 4. THE GATE: memory-level parallelism from N independent pointer chases
 *
 * N chains, each in its own multi-MB region, walked in lockstep. Each chain's next address depends
 * only on its own previous load, so the memory system is offered exactly N independent misses.
 *
 *   cycles per ROUND flat as N rises   -> the core overlaps N misses; a faster link can be reached
 *   cycles per round scaling with N    -> one miss at a time; the link cannot be saturated from the
 *                                         core at ANY wire clock, and only an ILA will show a change
 *
 * "overlap" below is the achieved MLP: N * (cycles/round at N=1) / (cycles/round at N). It is a
 * ratio of cycle counts, so it does not depend on resolving the core clock.
 * ---------------------------------------------------------------------------------------------- */
static void sweep_mlp(void) {
  const size_t sub = MEMBW_MLP_SUB_SPAN;
  MEMBW_LOG("[membw] --- 4. memory-level parallelism: N INDEPENDENT chases, %lu MB each ---------\r\n",
            (unsigned long)(sub >> 20));

  for (int k = 0; k < 8; k++) {
    if (mem_probe_chase_build(g_buf + (size_t)k * sub, sub, 0xC0DE0001u + (uint32_t)k) == 0u) {
      MEMBW_LOG("[membw] MLP: sub-span too small, skipped\r\n");
      return;
    }
  }

  static const int nch[4] = { 1, 2, 4, 8 };
  uint64_t round1 = 0;
  for (int i = 0; i < 4; i++) {
    const int n = nch[i];
    const size_t steps = MEMBW_LAT_STEPS;
    const uint64_t c = mem_probe_chase_parallel(g_buf, sub, n, steps, &g_isink);
    const uint64_t per_round_x100 = (c * 100ull) / steps;
    const uint64_t per_load_x100 = per_round_x100 / (uint64_t)n;
    if (n == 1) round1 = per_round_x100;
    /* achieved overlap = N * round1 / roundN */
    const uint64_t overlap_x100 = per_round_x100
        ? ((uint64_t)n * round1 * 100ull) / per_round_x100 : 0ull;
    if (n == 8) g_mlp_x100 = overlap_x100;
    MEMBW_LOG("[mem] chases=%d  %8lu Kcyc  %6lu.%02lu cyc/round  %6lu.%02lu cyc/load  "
              "overlap %2lu.%02lu misses%s\r\n",
              n, (unsigned long)(c / 1000ull),
              (unsigned long)(per_round_x100 / 100ull), (unsigned long)(per_round_x100 % 100ull),
              (unsigned long)(per_load_x100 / 100ull), (unsigned long)(per_load_x100 % 100ull),
              (unsigned long)(overlap_x100 / 100ull), (unsigned long)(overlap_x100 % 100ull),
              (n == 1) ? "  (reference)" : "");
  }
}

/* ------------------------------------------------------------------------------------------------
 * 5a. Streaming bandwidth
 * ---------------------------------------------------------------------------------------------- */
static void sweep_bandwidth(void) {
  const size_t span = MEMBW_STREAM_SPAN;
  uint8_t *const src = g_buf;
  uint8_t *const dst = g_buf + g_half;

  MEMBW_LOG("[membw] --- 5. streaming: %lu MB sequential ---------------------------------------\r\n",
            (unsigned long)(span >> 20));

  /* Write first: it leaves the range in a defined state and measures store bandwidth on a range
   * that has not just been read (so a write-allocate policy is not being flattered by a hit). */
  mem_probe_print_bw("write u64", mem_probe_write_u64(src, span), span, g_core_hz);
#if MEM_PROBE_RVV
  mem_probe_print_bw("write rvv m8", mem_probe_write_rvv(src, span), span, g_core_hz);
#endif

  mem_probe_print_bw("read  u64", mem_probe_read_u64(src, span, &g_sink), span, g_core_hz);
#if MEM_PROBE_RVV
  MEMBW_LOG("[membw] rvv vlmax e8m1=%lu e8m8=%lu B -> VLEN=%lu bits\r\n",
            (unsigned long)__riscv_vsetvlmax_e8m1(), (unsigned long)__riscv_vsetvlmax_e8m8(),
            (unsigned long)__riscv_vsetvlmax_e8m1() * 8ul);
  {
    static const int lmuls[4] = { 1, 2, 4, 8 };
    for (int i = 0; i < 4; i++) {
      char lbl[32];
      snprintf(lbl, sizeof(lbl), "read  rvv m%d (%luB/req)", lmuls[i],
               (unsigned long)(__riscv_vsetvlmax_e8m1() * (size_t)lmuls[i]));
      const uint64_t c = mem_probe_read_rvv(src, span, lmuls[i], &g_sink);
      mem_probe_print_bw(lbl, c, span, g_core_hz);
      const uint64_t cpb = mem_probe_cpb_x100(c, span);
      if (cpb && (g_read_cpb_x100 == 0ull || cpb < g_read_cpb_x100)) g_read_cpb_x100 = cpb;
    }
  }
#endif
  {
    const uint64_t cpb = mem_probe_cpb_x100(mem_probe_read_u64(src, span, &g_sink), span);
    if (cpb && (g_read_cpb_x100 == 0ull || cpb < g_read_cpb_x100)) g_read_cpb_x100 = cpb;
  }

  /* Copy moves `span` bytes in and `span` bytes out; the byte count charged is the read side, so
   * this column is directly comparable with the read line above. */
  mem_probe_print_bw("copy  u64 (rd side)", mem_probe_copy_u64(dst, src, span), span, g_core_hz);
}

/* ------------------------------------------------------------------------------------------------
 * 5b. Interleaved streams — the control and the experiment side by side
 * ---------------------------------------------------------------------------------------------- */
static void sweep_parallelism(void) {
  const size_t span = MEMBW_STREAM_SPAN;
  MEMBW_LOG("[membw] --- 5b. N interleaved read streams: dependent vs independent ---------------\r\n");
  MEMBW_LOG("[membw]      dependent = each load consumed by the next instruction (cannot overlap on\r\n"
            "[membw]      an in-order core, whatever the MSHR count); independent = all N loads of a\r\n"
            "[membw]      round issued before any is used. A gap between them is the LOOP, not the HW.\r\n");
  static const int ns[4] = { 1, 2, 4, 8 };
  for (int i = 0; i < 4; i++) {
    char lbl[32];
    snprintf(lbl, sizeof(lbl), "dep   %d stream%s", ns[i], (ns[i] == 1) ? " " : "s");
    mem_probe_print_bw(lbl, mem_probe_read_streams(g_buf, span, ns[i], &g_sink), span, g_core_hz);
  }
  for (int i = 0; i < 4; i++) {
    char lbl[32];
    snprintf(lbl, sizeof(lbl), "indep %d stream%s", ns[i], (ns[i] == 1) ? " " : "s");
    const uint64_t c = mem_probe_read_streams_indep(g_buf, span, ns[i], &g_sink);
    mem_probe_print_bw(lbl, c, span, g_core_hz);
    const uint64_t cpb = mem_probe_cpb_x100(c, span);
    if (cpb && (g_read_cpb_x100 == 0ull || cpb < g_read_cpb_x100)) g_read_cpb_x100 = cpb;
  }
}

static void sweep_stride(void) {
  /* Charge cycles per ACCESS, not per byte: the point is how a request's cost changes as the stride
   * crosses a line, which a per-byte column would hide behind the byte count. With no address
   * translation in play, a knee past 4 KB is a DRAM page/row effect, not a TLB one. */
  const size_t span = MEMBW_STREAM_SPAN;
  static const size_t strides[] = { 8, 16, 32, 64, 128, 256, 1024, 4096 };
  MEMBW_LOG("[membw] --- 5c. stride sweep: one 8 B load every N bytes over %lu MB ---------------\r\n",
            (unsigned long)(span >> 20));
  for (size_t k = 0; k < sizeof(strides) / sizeof(strides[0]); k++) {
    const size_t stride = strides[k];
    const uint64_t c = mem_probe_read_stride(g_buf, span, stride, &g_sink);
    const uint64_t accesses = span / stride;
    char lbl[32];
    snprintf(lbl, sizeof(lbl), "stride %4lu B", (unsigned long)stride);
    mem_probe_print_lat(lbl, c, accesses, g_core_hz);
  }
}

/* ------------------------------------------------------------------------------------------------
 * 6. Little's Law verdict
 *
 * Everything above, reduced to the one question: can this link be saturated from the core?
 * ---------------------------------------------------------------------------------------------- */
static void report_little(void) {
  MEMBW_LOG("[membw] --- 6. Little's Law: BW = lines_in_flight * %u B / RTT --------------------\r\n",
            (unsigned)MEM_PROBE_LINE_BYTES);
  if (!g_rtt_cyc || !g_read_cpb_x100) {
    MEMBW_LOG("[membw] not enough data (rtt=%lu cyc, read=%lu.%02lu cyc/byte)\r\n",
              (unsigned long)g_rtt_cyc, (unsigned long)(g_read_cpb_x100 / 100ull),
              (unsigned long)(g_read_cpb_x100 % 100ull));
    return;
  }

  const uint64_t lif = mem_probe_lines_in_flight_x100(g_rtt_cyc, g_read_cpb_x100);
  MEMBW_LOG("[membw] measured RTT %lu cyc, best streaming read %lu.%02lu cyc/byte\r\n",
            (unsigned long)g_rtt_cyc, (unsigned long)(g_read_cpb_x100 / 100ull),
            (unsigned long)(g_read_cpb_x100 % 100ull));
  MEMBW_LOG("[membw] => streaming sustains %lu.%02lu lines in flight; independent chases achieved "
            "%lu.%02lu\r\n",
            (unsigned long)(lif / 100ull), (unsigned long)(lif % 100ull),
            (unsigned long)(g_mlp_x100 / 100ull), (unsigned long)(g_mlp_x100 % 100ull));

  /* The bar. A byte rate needs the clock, and the clock is the one thing under dispute, so print the
   * requirement at BOTH candidates rather than picking one and being quietly wrong by 15x. */
  const uint64_t cands[2] = { (uint64_t)SYS_CLK_FREQ, g_clk.requested_hz };
  MEMBW_LOG("[membw] lines in flight needed to sustain %lu MB/s over a %lu-cycle RTT:\r\n",
            (unsigned long)MEMBW_LINK_TARGET_MBPS, (unsigned long)g_rtt_cyc);
  for (int i = 0; i < 2; i++) {
    const uint64_t need = mem_probe_lines_needed_x100(g_rtt_cyc, MEMBW_LINK_TARGET_MBPS, cands[i]);
    const uint64_t ns = mem_probe_ns_x10(g_rtt_cyc * 100ull, cands[i]);
    MEMBW_LOG("[membw]   at %4lu MHz core: RTT = %lu.%01lu us, need %lu.%02lu lines%s\r\n",
              (unsigned long)(cands[i] / 1000000ull),
              (unsigned long)(ns / 10000ull), (unsigned long)((ns / 1000ull) % 10ull),
              (unsigned long)(need / 100ull), (unsigned long)(need % 100ull),
              (cands[i] == g_core_hz) ? "   <-- resolved clock" : "");
  }

  if (g_mlp_x100 < 150ull) {
    MEMBW_LOG("[membw] VERDICT: the core sustains ~1 outstanding miss. No wire clock and no kernel\r\n"
              "[membw]          change can saturate the link from the core — use a DMA engine to\r\n"
              "[membw]          generate concurrency, or measure the link with an ILA counting\r\n"
              "[membw]          out.valid && out.ready on the serial TL port.\r\n");
  } else {
    MEMBW_LOG("[membw] VERDICT: the core DOES overlap misses. Push N until cycles/round stops being\r\n"
              "[membw]          flat, then compare bitstreams at that operating point.\r\n");
  }
}

/* ------------------------------------------------------------------------------------------------
 * Entry
 * ---------------------------------------------------------------------------------------------- */

void app_init(void) {
  /* Sample the cycles-per-mtime-tick ratio BEFORE the PLL is touched, while the core is still on the
   * reset clock (SYS_CLK_FREQ). Nothing can print yet; stash it. Comparing this against the same
   * ratio after init_test() is what separates "mtime tracks the core" from "the PLL never engaged" —
   * two very different faults that produce identical timing evidence on their own. */
  const uint64_t cpt_ref = clock_probe_sample(mtime_now, MEMBW_CLOCK_TICKS);

  /* PLL + UART, but only on real hardware: Spike models neither, and programming them faults before
   * the first character prints — which looks exactly like a hang (CLAUDE.md). */
#if defined(TERMINAL_DEVICE_UART0)
  init_test(target_frequency);
#endif
  setvbuf(stdout, NULL, _IONBF, 0);

  const uint64_t cpt_pll = clock_probe_sample(mtime_now, MEMBW_CLOCK_TICKS);
  clock_probe_resolve(&g_clk, cpt_ref, cpt_pll, target_frequency);
  g_core_hz = g_clk.core_hz;

  printf("\r\n");
  MEMBW_LOG("[membw] memory latency / MLP / bandwidth sweep\r\n");
  {
    uint64_t hartid, misa;
    __asm__ volatile("csrr %0, mhartid" : "=r"(hartid));
    __asm__ volatile("csrr %0, misa" : "=r"(misa));   /* mandatory in M-mode; cannot trap */
    MEMBW_LOG("[membw] hart %lu, misa=0x%016llx, bare M-mode\r\n",
              (unsigned long)hartid, (unsigned long long)misa);
  }
  /* Said explicitly because it removes a whole class of hypothesis from the latency curve below. */
  MEMBW_LOG("[membw] no address translation: nothing in glossy/ platform/ bmark-lib/ ever writes "
            "satp, so no knee here can be a TLB or page-walk effect (and huge pages are not a "
            "lever that exists).\r\n");

  MEMBW_LOG("[membw] --- 1. clock ------------------------------------------------------------\r\n");
  clock_probe_print(&g_clk, "[membw]");
}

void app_main(void) {
  MEMBW_LOG("[membw] --- 2. window -----------------------------------------------------------\r\n");
  if (window_check() != 0) {
    MEMBW_LOG("[membw] aborting.\r\n");
    for (;;) __asm__ volatile("wfi");
  }

  const uint64_t t_start = mem_probe_rdcycle();
  for (int pass = 0; pass < MEMBW_PASSES; pass++) {
    MEMBW_LOG("\r\n[membw] ===== pass %d/%d =====================================================\r\n",
              pass + 1, MEMBW_PASSES);
    g_read_cpb_x100 = 0ull;   /* per-pass best, so pass 2 is an independent reading */
    sweep_latency();
    sweep_mlp();
    sweep_bandwidth();
    sweep_parallelism();
    sweep_stride();
    report_little();
  }
  const uint64_t t_total = mem_probe_rdcycle() - t_start;

  MEMBW_LOG("\r\n[membw] SUMMARY core=%lu.%02lu MHz  rtt=%lu cyc  read=%lu.%02lu cyc/byte "
            "(%lu.%02lu MB/s)  mlp=%lu.%02lu lines  pll_ratio=%lu mux=%lu\r\n",
            (unsigned long)(g_core_hz / 1000000ull), (unsigned long)((g_core_hz % 1000000ull) / 10000ull),
            (unsigned long)g_rtt_cyc,
            (unsigned long)(g_read_cpb_x100 / 100ull), (unsigned long)(g_read_cpb_x100 % 100ull),
            (unsigned long)(mem_probe_mbps_x100(g_read_cpb_x100, 100ull, g_core_hz) / 100ull),
            (unsigned long)(mem_probe_mbps_x100(g_read_cpb_x100, 100ull, g_core_hz) % 100ull),
            (unsigned long)(g_mlp_x100 / 100ull), (unsigned long)(g_mlp_x100 % 100ull),
            (unsigned long)g_clk.pll_ratio, (unsigned long)g_clk.sel_uncore);

  /* The only genuinely independent check available on a chip whose one timer is derived from the
   * clock under test: a stopwatch. If the wall-clock time is far off this, the resolved clock is
   * wrong and every ns / MB/s figure above scales with it. */
  {
    const uint64_t ms = g_core_hz ? (t_total * 1000ull) / g_core_hz : 0ull;
    MEMBW_LOG("[membw] sweep took %lu cycles = %lu.%03lu s at the resolved clock — CHECK THIS "
              "AGAINST A STOPWATCH; it is the only time reference on this chip that is not derived "
              "from the clock being measured.\r\n",
              (unsigned long)t_total, (unsigned long)(ms / 1000ull), (unsigned long)(ms % 1000ull));
  }

  MEMBW_LOG("[membw] done (sink %lu).\r\n", (unsigned long)(g_sink + g_isink));
  for (;;) __asm__ volatile("wfi");
}

int main(void) {
  app_init();
  app_main();
  return 0;
}

/* Harts 1..N park: every measurement here is single-core on purpose. */
void __attribute__((weak, noreturn)) __main(void) {
  for (;;) __asm__ volatile("wfi");
}
