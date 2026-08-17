/*
 * Host unit test for c2c-demos/common/mem_probe.h.
 *
 *   gcc -O2 -Wall -I../../common -o host_mem_probe_test host_mem_probe_test.c && ./host_mem_probe_test
 *
 * mem_probe.h is chip-free on purpose (only rdcycle/rdtsc and plain loads and stores), so the parts
 * that can be WRONG rather than merely slow are checkable in a second on the host instead of a
 * flash-and-squint cycle on silicon — the same reason bearly-smollm's model.c has a host reference.
 *
 * Two things are worth proving:
 *
 *   1. The pointer chase is a SINGLE cycle through EVERY node. If Sattolo's algorithm were written
 *      as Fisher-Yates by accident (`% (i + 1)` instead of `% i`) the permutation would decompose
 *      into several short cycles, the chase would revisit a small resident subset, and the reported
 *      "DRAM latency" would silently come out several times too low — a wrong number that looks
 *      entirely plausible, which is the worst kind for a measurement whose whole job is comparing
 *      two bitstreams.
 *   2. The fixed-point formatting is exact. Everything is integer maths to keep newlib's %f (and its
 *      kilobyte of stack) out of a demo where a stack overflow does not trap.
 *
 * The loops themselves are also checked to touch exactly the elements they claim, since a bandwidth
 * figure computed over the wrong byte count is just as misleading.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "mem_probe.h"

static int failures;

static void check_eq(const char *what, unsigned long long got, unsigned long long want) {
  if (got == want) {
    printf("  ok   %-34s = %llu\n", what, got);
  } else {
    printf("  FAIL %-34s = %llu, want %llu\n", what, got, want);
    failures++;
  }
}

int main(void) {
  const size_t span = 1u << 20;
  void *buf = aligned_alloc(MEM_PROBE_LINE_BYTES, span);
  assert(buf != NULL);

  printf("chase:\n");
  const size_t n = mem_probe_chase_build(buf, span, 12345u);
  check_eq("node count", n, span / MEM_PROBE_LINE_BYTES);

  /* Walk from node 0 back to node 0, refusing to visit anything twice. */
  char *seen = calloc(n, 1);
  uint32_t i = 0;
  size_t steps = 0;
  do {
    assert(i < n);
    if (seen[i]) { printf("  FAIL revisited node %u after %zu steps\n", i, steps); failures++; break; }
    seen[i] = 1;
    i = MEM_PROBE_NODE(buf, i);
    steps++;
  } while (i != 0);
  check_eq("cycle length (want = node count)", steps, n);
  size_t unvisited = 0;
  for (size_t k = 0; k < n; k++) if (!seen[k]) unvisited++;
  check_eq("nodes never visited", unvisited, 0);
  free(seen);

  /* A span too small for two nodes must be refused, not silently measured. */
  check_eq("build(span < 2 lines)", mem_probe_chase_build(buf, MEM_PROBE_LINE_BYTES, 1u), 0);

  printf("fixed-point formatting:\n");
  check_eq("cpb_x100(1000 cyc, 100 B)", mem_probe_cpb_x100(1000, 100), 1000);          /* 10.00 */
  check_eq("cpb_x100(_, 0 B) guards div0", mem_probe_cpb_x100(1000, 0), 0);
  check_eq("mbps_x100(1000 cyc, 100 B, 1 GHz)", mem_probe_mbps_x100(1000, 100, 1000000000ull), 10000); /* 100.00 MB/s */
  check_eq("mbps_x100(_, _, 0 Hz) guards", mem_probe_mbps_x100(1000, 100, 0), 0);
  check_eq("ns_x10(200.00 cyc, 750 MHz)", mem_probe_ns_x10(20000, 750000000ull), 2666); /* 266.6 ns */
  /* 32 MB in 1 Gcyc at 750 MHz = 29.80 cyc/byte and 25.16 MB/s. */
  check_eq("mbps_x100(1 Gcyc, 32 MB, 750 MHz)", mem_probe_mbps_x100(1000000000ull, 32u << 20, 750000000ull), 2516);

  printf("sample log lines:\n");
  mem_probe_print_lat("chase   1 MB", 200ull * 100000ull, 100000ull, 750000000ull);
  mem_probe_print_bw("read u64", 1000000000ull, 32u << 20, 750000000ull);

  printf("loop coverage:\n");
  uint64_t *p = (uint64_t *)buf;
  const size_t nwords = span / 8u;
  for (size_t k = 0; k < nwords; k++) p[k] = 1;

  uint64_t sink = 0;
  mem_probe_read_u64(buf, span, &sink);
  check_eq("read_u64 touched words", sink, nwords);

  sink = 0;
  mem_probe_read_streams(buf, span, 4, &sink);
  check_eq("read_streams(4) touched words", sink, nwords);

  sink = 0;
  mem_probe_read_stride(buf, span, MEM_PROBE_LINE_BYTES, &sink);
  check_eq("read_stride(64) touched words", sink, span / MEM_PROBE_LINE_BYTES);

  mem_probe_write_u64(buf, span);
  check_eq("write_u64 wrote last word", p[nwords - 1], (unsigned long long)(nwords - 1) * 0x0101010101010101ull);

  void *dst = aligned_alloc(MEM_PROBE_LINE_BYTES, span);
  memset(dst, 0, span);
  mem_probe_copy_u64(dst, buf, span);
  check_eq("copy_u64 matches source", memcmp(dst, buf, span) == 0, 1);
  free(dst);
  free(buf);

  /* ---------------------------------------------------------------------------------------------
   * Memory-level parallelism. The arithmetic is checked against hand-worked values, and then the
   * measurement is actually RUN on the host: an out-of-order x86 must show overlap well above 1, so
   * a structurally broken probe (chains sharing a cycle, the compiler sinking the loads, the timer
   * in the wrong place) shows up here rather than being mistaken on silicon for "this core has one
   * MSHR" — which is precisely the conclusion the sweep exists to support.
   * ------------------------------------------------------------------------------------------- */
  printf("Little's Law arithmetic:\n");
  /* 3207-cycle RTT at 31.64 cyc/byte: 3207/31.64 = 101.4 B per RTT = 1.58 lines. */
  check_eq("lines_in_flight(3207 cyc, 31.64)", mem_probe_lines_in_flight_x100(3207, 3164), 158);
  /* 33 MB/s over a 3207-cycle RTT: 64.1 us at 50 MHz -> 33.07 lines; 4.28 us at 750 MHz -> 2.20. */
  check_eq("lines_needed(33 MB/s, 50 MHz)", mem_probe_lines_needed_x100(3207, 33, 50000000ull), 3307);
  check_eq("lines_needed(33 MB/s, 750 MHz)", mem_probe_lines_needed_x100(3207, 33, 750000000ull), 220);

  printf("parallel chase (host is out-of-order: overlap must exceed 1):\n");
  {
    const size_t sub = 8u << 20;              /* per chain, well past any host LLC slice */
    const int maxch = 8;
    uint8_t *big = aligned_alloc(MEM_PROBE_LINE_BYTES, sub * (size_t)maxch);
    assert(big != NULL);
    for (int k = 0; k < maxch; k++) {
      const size_t got = mem_probe_chase_build(big + (size_t)k * sub, sub, 0xC0DE0001u + (uint32_t)k);
      assert(got == sub / MEM_PROBE_LINE_BYTES);
    }
    /* Chains must be disjoint cycles: walking chain 1 must never leave its own sub-region. That is
     * guaranteed by construction (indices are chain-relative), but assert it — if the chains shared
     * a cycle they would not be independent and the overlap figure would be fiction. */
    uint32_t idx = 0;
    for (int s = 0; s < 10000; s++) idx = MEM_PROBE_NODE(big + sub, idx);
    check_eq("chain 1 stays in range", idx < (uint32_t)(sub / MEM_PROBE_LINE_BYTES), 1);

    uint32_t sk = 0;
    const size_t steps = 200000u;
    uint64_t round1 = 0;
    static const int nch[4] = { 1, 2, 4, 8 };
    int overlap_grew = 0;
    for (int i = 0; i < 4; i++) {
      const uint64_t c = mem_probe_chase_parallel(big, sub, nch[i], steps, &sk);
      const uint64_t per_round = (c * 100ull) / steps;
      if (nch[i] == 1) round1 = per_round;
      const uint64_t overlap = per_round ? ((uint64_t)nch[i] * round1 * 100ull) / per_round : 0;
      printf("  ..   chases=%d  %6llu.%02llu cyc/round  overlap %llu.%02llu\n", nch[i],
             (unsigned long long)(per_round / 100), (unsigned long long)(per_round % 100),
             (unsigned long long)(overlap / 100), (unsigned long long)(overlap % 100));
      if (nch[i] == 8 && overlap > 150ull) overlap_grew = 1;
    }
    check_eq("overlap > 1.5 at 8 chains (host)", overlap_grew, 1);
    free(big);
  }

  printf(failures ? "\nFAILED (%d)\n" : "\nALL OK\n", failures);
  return failures ? 1 : 0;
}
