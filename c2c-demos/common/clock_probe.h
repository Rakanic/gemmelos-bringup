#ifndef CLOCK_PROBE_H
#define CLOCK_PROBE_H

/* ------------------------------------------------------------------------------------------------
 * clock_probe — establish what the core clock ACTUALLY is, and prove it.
 *
 * This exists because the obvious measurement is wrong on this silicon. Counting `rdcycle` against
 * CLINT `mtime` and scaling by MTIME_FREQ reports 49.99 MHz on a chip whose PLL was programmed for
 * 750 MHz — and the reason is not that the PLL failed. `mtime` appears to be DERIVED FROM the core
 * clock (the observed ratio is exactly SYS_CLK_FREQ / MTIME_FREQ = 1000 regardless of the PLL), so
 * MTIME_FREQ's 50 kHz only holds while the core runs at the 50 MHz reference. mtime is therefore not
 * an independent time base and cannot, by itself, measure the clock that drives it.
 *
 * That is worth getting right rather than shrugging at: every nanosecond and MB/s figure scales with
 * it, including the DRAM round-trip time that sets how many lines must be in flight to saturate a
 * link. At 50 MHz a 3207-cycle RTT is 64 us and needs ~33 outstanding misses to reach 33 MB/s; at
 * 750 MHz the same 3207 cycles is 4.3 us and needs ~2. Same measurement, opposite conclusion.
 *
 * So: sample the cycles-per-tick ratio twice, once at the reset clock and once after the PLL is
 * programmed, and read the PLL and clock-mux registers back. Three cases, distinguishable:
 *
 *   ratio changes by the programmed multiplier   -> mtime IS independent; trust the measurement.
 *   ratio unchanged, mux on PLL0, RATIO as asked -> mtime tracks the core; the clock is
 *                                                   SYS_CLK_FREQ * RATIO (corroborated by the
 *                                                   console being legible at all, since init_test
 *                                                   derived the UART divisor from that same number).
 *   ratio unchanged, mux NOT on PLL0             -> the PLL really did not engage; SYS_CLK_FREQ.
 *
 * The middle and last cases produce identical timing evidence, which is exactly why the register
 * readback is not optional.
 *
 * Chip-dependent by necessity (PLL and RCC register maps), unlike mem_probe.h. Both bearly25 and
 * dsp25 define PLL / RCC_CLOCK_SELECTOR / SYS_CLK_FREQ / MTIME_FREQ in their chip_config.h.
 * ---------------------------------------------------------------------------------------------- */

#include <stdint.h>
#include <stdio.h>

#include "chip_config.h"
#include "mem_probe.h"

typedef enum {
  CLOCK_PROBE_MTIME_INDEPENDENT = 0,   /* mtime is its own reference; the ratio measured the clock */
  CLOCK_PROBE_MTIME_TRACKS_CORE = 1,   /* mtime is core/K; clock inferred from the PLL registers */
  CLOCK_PROBE_PLL_NOT_ENGAGED   = 2,   /* mux is not on PLL0: still running on the reference clock */
} clock_probe_verdict_t;

typedef struct {
  uint64_t cpt_ref_x1000;   /* cycles per mtime tick before init_test (reset/slow clock) */
  uint64_t cpt_pll_x1000;   /* ... and after the PLL was programmed */
  uint64_t requested_hz;    /* what init_test was asked for */
  uint64_t core_hz;         /* resolved; use this for every ns / MB/s conversion */
  uint64_t mtime_hz;        /* effective mtime frequency at the resolved core clock */
  uint32_t pll_en, pll_bypass, pll_ratio, pll_fraction, pll_mdiv, pll_zdiv0, pll_zdiv1;
  uint32_t sel_uncore, sel_tile0, sel_tile1, sel_clktap;
  uint32_t requested_ratio;
  clock_probe_verdict_t verdict;
} clock_probe_t;

/* Sample the cycles-per-mtime-tick ratio. Call once BEFORE init_test() (nothing may print yet, so
 * stash the result) and once after. */
MEM_PROBE_UNUSED static uint64_t clock_probe_sample(uint64_t (*now_mtime)(void), uint32_t ticks) {
  return mem_probe_cycles_per_tick_x1000(now_mtime, ticks);
}

/* Resolve the core clock from the two samples plus the PLL/mux registers. */
MEM_PROBE_UNUSED static void clock_probe_resolve(clock_probe_t *cp, uint64_t cpt_ref_x1000,
                                                 uint64_t cpt_pll_x1000, uint64_t requested_hz) {
  cp->cpt_ref_x1000 = cpt_ref_x1000;
  cp->cpt_pll_x1000 = cpt_pll_x1000;
  cp->requested_hz = requested_hz;
  cp->requested_ratio = (uint32_t)(requested_hz / (uint64_t)SYS_CLK_FREQ);
  if (cp->requested_ratio == 0u) cp->requested_ratio = 1u;

  cp->pll_en       = PLL->PLLEN;
  cp->pll_bypass   = PLL->BYPASS;
  cp->pll_ratio    = PLL->RATIO;
  cp->pll_fraction = PLL->FRACTION;
  cp->pll_mdiv     = PLL->MDIV_RATIO;
  cp->pll_zdiv0    = PLL->ZDIV0_RATIO;
  cp->pll_zdiv1    = PLL->ZDIV1_RATIO;

  cp->sel_uncore = RCC_CLOCK_SELECTOR->UNCORE;
  cp->sel_tile0  = RCC_CLOCK_SELECTOR->TILE0;
  cp->sel_tile1  = RCC_CLOCK_SELECTOR->TILE1;
  cp->sel_clktap = RCC_CLOCK_SELECTOR->CLKTAP;

  /* Did the cycles-per-tick ratio move? If mtime were an independent reference it would rise by
   * exactly the PLL multiplier. Allow generous slop (>1.5x) — we are separating 1x from 15x. */
  const int ratio_moved = (cpt_ref_x1000 != 0ull) &&
                          (cpt_pll_x1000 * 2ull > cpt_ref_x1000 * 3ull);
  const int mux_on_pll = (cp->sel_uncore == (uint32_t)CLKSEL_PLL0) &&
                         (cp->sel_tile0 == (uint32_t)CLKSEL_PLL0);
  const int pll_programmed = cp->pll_en && !cp->pll_bypass && (cp->pll_ratio == cp->requested_ratio);

  if (ratio_moved) {
    cp->verdict = CLOCK_PROBE_MTIME_INDEPENDENT;
    /* mtime frequency cancels: core = SYS_CLK * (cpt_pll / cpt_ref). */
    cp->core_hz = (uint64_t)SYS_CLK_FREQ * cpt_pll_x1000 / cpt_ref_x1000;
    cp->mtime_hz = cpt_pll_x1000 ? (cp->core_hz * 1000ull) / cpt_pll_x1000 : 0ull;
  } else if (mux_on_pll && pll_programmed) {
    cp->verdict = CLOCK_PROBE_MTIME_TRACKS_CORE;
    cp->core_hz = (uint64_t)SYS_CLK_FREQ * (uint64_t)cp->pll_ratio;
    cp->mtime_hz = cpt_pll_x1000 ? (cp->core_hz * 1000ull) / cpt_pll_x1000 : 0ull;
  } else {
    cp->verdict = CLOCK_PROBE_PLL_NOT_ENGAGED;
    cp->core_hz = (uint64_t)SYS_CLK_FREQ;
    cp->mtime_hz = (uint64_t)MTIME_FREQ;
  }
}

MEM_PROBE_UNUSED static void clock_probe_print(const clock_probe_t *cp, const char *tag) {
  printf("%s PLL regs: PLLEN=%lu BYPASS=%lu RATIO=%lu (asked %lu) FRACTION=%lu MDIV=%lu "
         "ZDIV0=%lu ZDIV1=%lu\r\n", tag,
         (unsigned long)cp->pll_en, (unsigned long)cp->pll_bypass, (unsigned long)cp->pll_ratio,
         (unsigned long)cp->requested_ratio, (unsigned long)cp->pll_fraction,
         (unsigned long)cp->pll_mdiv, (unsigned long)cp->pll_zdiv0, (unsigned long)cp->pll_zdiv1);
  printf("%s clk mux:  UNCORE=%lu TILE0=%lu TILE1=%lu CLKTAP=%lu  (0=SLOW 1=PLL0 2=PLL1)\r\n", tag,
         (unsigned long)cp->sel_uncore, (unsigned long)cp->sel_tile0,
         (unsigned long)cp->sel_tile1, (unsigned long)cp->sel_clktap);
  printf("%s cyc/mtime-tick: %lu.%03lu before PLL, %lu.%03lu after  (unchanged => mtime is derived "
         "from the core clock)\r\n", tag,
         (unsigned long)(cp->cpt_ref_x1000 / 1000ull), (unsigned long)(cp->cpt_ref_x1000 % 1000ull),
         (unsigned long)(cp->cpt_pll_x1000 / 1000ull), (unsigned long)(cp->cpt_pll_x1000 % 1000ull));

  const char *why;
  switch (cp->verdict) {
    case CLOCK_PROBE_MTIME_INDEPENDENT:
      why = "mtime is an independent reference; the clock was MEASURED";
      break;
    case CLOCK_PROBE_MTIME_TRACKS_CORE:
      why = "mtime tracks the core, so it cannot measure it; clock taken from PLL RATIO x SYS_CLK "
            "(the legible console corroborates it: init_test set the UART divisor from the same "
            "number)";
      break;
    default:
      why = "!!! the clock mux is NOT on PLL0 -- the core really is on the reference clock !!!";
      break;
  }
  printf("%s core clock = %lu.%02lu MHz, effective mtime = %lu Hz (chip_config says %lu)\r\n", tag,
         (unsigned long)(cp->core_hz / 1000000ull),
         (unsigned long)((cp->core_hz % 1000000ull) / 10000ull),
         (unsigned long)cp->mtime_hz, (unsigned long)MTIME_FREQ);
  printf("%s   %s\r\n", tag, why);
  if (cp->verdict == CLOCK_PROBE_MTIME_TRACKS_CORE)
    printf("%s   NOTE: every msleep()/CLINT-timer interval in this tree is therefore %lux SHORT at "
           "this clock.\r\n", tag, (unsigned long)cp->pll_ratio);
}

#endif /* CLOCK_PROBE_H */
