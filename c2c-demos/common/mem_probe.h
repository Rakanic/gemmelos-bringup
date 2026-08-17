#ifndef MEM_PROBE_H
#define MEM_PROBE_H

/* ------------------------------------------------------------------------------------------------
 * mem_probe — memory-system latency and bandwidth primitives, shared by every demo that cares how
 * fast this SoC can read DRAM.
 *
 * Why this exists: on both SP25 chips the interesting workloads (SmolLM, Citrinet, Moonshine) are
 * entirely WEIGHT-BANDWIDTH BOUND — bearly25 measured ~31.6 cycles per weight byte, so a 143 MB
 * model costs ~4.5 Gcyc/token no matter what the kernel does (see CLAUDE.md, "Memory streams at
 * ~31.6 cyc/byte"). That makes the memory bus the single number worth tracking across bitstreams,
 * and it needs to be measured the SAME way every time or the comparison is worthless. Hence one
 * header, included by both `c2c-demos/membw` (the full sweep) and `c2c-demos/bearly-smollm` (a
 * three-line boot summary next to the real workload).
 *
 * Two independent quantities, which a bus-frequency change moves differently:
 *
 *   LATENCY   — cycles to complete ONE dependent load that misses to DRAM. Measured with a random
 *               single-cycle pointer chase (Sattolo), so no prefetcher can hide it and no two loads
 *               can overlap. This is the number that drops when the memory clock goes up.
 *   BANDWIDTH — cycles per byte for a long sequential stream, where the hardware is free to overlap
 *               whatever it can. On bearly25 these two are nearly the same measurement because the
 *               core sustains only one line fill at a time; if a bitstream change fixes THAT, the
 *               bandwidth number improves far more than the latency number, and the multi-stream
 *               sweep says so directly.
 *
 * Everything is integer-formatted. newlib's %f pulls in _dtoa_r, which alone can want a kilobyte of
 * stack, and a stack overflow on this platform does not trap — it silently corrupts automatics or
 * hangs the core (CLAUDE.md, LINKER=llm entry).
 *
 * Chip-free by construction: only `rdcycle` and plain loads/stores, plus optional RVV intrinsics.
 * The CLINT time base needed by mem_probe_core_hz() comes in as a function pointer so this header
 * never includes a chip header and can be compiled for either chip, or for the host.
 * ---------------------------------------------------------------------------------------------- */

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

#if defined(__riscv_v_intrinsic)
#include <riscv_vector.h>
#define MEM_PROBE_RVV 1
#else
#define MEM_PROBE_RVV 0
#endif

/* Cache line size. Both chips report 64 B in the ccache Config register. The pointer chase steps by
 * this so every dependent load is a separate line fill. */
#ifndef MEM_PROBE_LINE_BYTES
#define MEM_PROBE_LINE_BYTES 64u
#endif

/* Header-only: a translation unit that uses half of these should not be warned about the rest. */
#define MEM_PROBE_UNUSED __attribute__((unused))

/* The x86 arm is not decoration: it is what lets this header be compiled and unit-tested on the
 * host (the chase really is a single full cycle, the fixed-point formatting really rounds the way
 * the log lines claim) without a 40-second flash per iteration — the same trick that made
 * bearly-smollm's model.c host-testable. */
static inline uint64_t mem_probe_rdcycle(void) {
#if defined(__riscv)
  uint64_t c;
  __asm__ volatile("rdcycle %0" : "=r"(c));
  return c;
#elif defined(__x86_64__)
  uint32_t lo, hi;
  __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
  return ((uint64_t)hi << 32) | lo;
#else
  return 0;
#endif
}

/* ------------------------------------------------------------------------------------------------
 * Clock
 * ---------------------------------------------------------------------------------------------- */

/* Measure the actual core clock by counting rdcycle over a fixed number of CLINT mtime ticks.
 *
 * Do not skip this and trust the configured PLL target. `init_test()` programs a ratio of
 * SYS_CLK_FREQ and the PLL may not lock where it was asked to (1 GHz did not lock on dsp25), and a
 * bitstream that changes bus clocking may change the core clock too — in which case cyc/byte can
 * IMPROVE while wall-clock throughput gets worse. mtime runs off its own MTIME_FREQ reference, so
 * this ratio is the ground truth that turns cycles into seconds.
 *
 * `now_mtime`  — CLINT mtime reader, e.g. a wrapper around clint_get_time(CLINT).
 * `mtime_freq` — MTIME_FREQ from chip_config.h (50 kHz on both chips = 20 us/tick).
 * `ticks`      — how long to sample; 500 ticks = 10 ms is plenty and costs nothing.
 * Returns the measured core frequency in Hz (0 if now_mtime is NULL).
 */
MEM_PROBE_UNUSED static uint64_t mem_probe_core_hz(uint64_t (*now_mtime)(void), uint32_t mtime_freq, uint32_t ticks) {
  if (now_mtime == NULL || ticks == 0u) return 0ull;
  const uint64_t t0 = now_mtime();
  while (now_mtime() == t0) { }          /* align to a tick edge so we never sample a partial one */
  const uint64_t t1 = now_mtime();
  const uint64_t c0 = mem_probe_rdcycle();
  while ((now_mtime() - t1) < (uint64_t)ticks) { }
  const uint64_t dc = mem_probe_rdcycle() - c0;
  const uint64_t dt = now_mtime() - t1;
  if (dt == 0ull) return 0ull;
  return dc * (uint64_t)mtime_freq / dt;
}

/* Cycles per CLINT mtime tick, x1000. This is the RAW ratio, deliberately not converted to a
 * frequency: on this silicon mtime appears to be derived from the core clock (the ratio is exactly
 * SYS_CLK_FREQ/MTIME_FREQ = 1000 whatever the PLL is programmed to), so mtime is NOT an independent
 * time base and mem_probe_core_hz() cannot be trusted on its own. Sampling this once at the reset
 * clock and once after the PLL is programmed is what distinguishes "mtime tracks the core" from
 * "the PLL never engaged" — see c2c-demos/common/clock_probe.h. */
MEM_PROBE_UNUSED static uint64_t mem_probe_cycles_per_tick_x1000(uint64_t (*now_mtime)(void),
                                                                 uint32_t ticks) {
  if (now_mtime == NULL || ticks == 0u) return 0ull;
  const uint64_t t0 = now_mtime();
  while (now_mtime() == t0) { }          /* align to a tick edge so we never sample a partial one */
  const uint64_t t1 = now_mtime();
  const uint64_t c0 = mem_probe_rdcycle();
  while ((now_mtime() - t1) < (uint64_t)ticks) { }
  const uint64_t dc = mem_probe_rdcycle() - c0;
  const uint64_t dt = now_mtime() - t1;
  return dt ? (dc * 1000ull) / dt : 0ull;
}

/* ------------------------------------------------------------------------------------------------
 * Latency: random dependent-load pointer chase
 * ---------------------------------------------------------------------------------------------- */

static inline uint32_t mem_probe_rnd(uint32_t *s) {
  uint32_t x = *s;
  x ^= x << 13; x ^= x >> 17; x ^= x << 5;
  *s = x;
  return x;
}

#define MEM_PROBE_NODE(base, i) (*(volatile uint32_t *)((uint8_t *)(base) + (size_t)(i) * MEM_PROBE_LINE_BYTES))

/* Lay out a chase over the first `span` bytes of `buf`: one node per cache line, each holding the
 * index of the next.
 *
 * Sattolo's algorithm applied to the identity permutation yields a permutation that is a SINGLE
 * cycle of length n — so the chase visits every line exactly once before repeating, and there is no
 * short loop that would sit in cache and flatter the result. A constant-stride chase would be
 * simpler but a stride prefetcher can run ahead of it, which turns a latency measurement into a
 * bandwidth measurement without saying so.
 *
 * `buf` must be writable and at least `span` bytes. Returns the node count (0 if span is too small).
 * Setup itself is O(n) random DRAM writes — ~0.3 s for a 64 MB span at 750 MHz. */
MEM_PROBE_UNUSED static size_t mem_probe_chase_build(void *buf, size_t span, uint32_t seed) {
  const size_t n = span / MEM_PROBE_LINE_BYTES;
  if (n < 2u) return 0u;
  for (size_t i = 0; i < n; i++) MEM_PROBE_NODE(buf, i) = (uint32_t)i;
  uint32_t s = seed ? seed : 0x9E3779B9u;
  for (size_t i = n - 1u; i > 0u; i--) {
    const size_t j = (size_t)(mem_probe_rnd(&s) % (uint32_t)i);   /* j < i: Sattolo, not Fisher-Yates */
    const uint32_t a = MEM_PROBE_NODE(buf, i);
    MEM_PROBE_NODE(buf, i) = MEM_PROBE_NODE(buf, j);
    MEM_PROBE_NODE(buf, j) = a;
  }
  return n;
}

/* Walk `steps` dependent loads around a chase built by mem_probe_chase_build(). Returns cycles.
 * Divide by `steps` for cycles per load-to-use, which for a span far larger than the last-level
 * cache IS the DRAM latency the core sees. */
MEM_PROBE_UNUSED static uint64_t mem_probe_chase_run(const void *buf, size_t steps, uint32_t *sink) {
  uint32_t i = 0;
  const uint64_t t = mem_probe_rdcycle();
  for (size_t k = 0; k < steps; k++) i = MEM_PROBE_NODE(buf, i);
  const uint64_t c = mem_probe_rdcycle() - t;
  *sink += i;
  return c;
}

/* ------------------------------------------------------------------------------------------------
 * Memory-level parallelism: N INDEPENDENT pointer chases walked in lockstep
 *
 * THE measurement for "how many outstanding misses can this core sustain", and the one that decides
 * whether a faster link can ever be saturated from the core. Each chain's next address depends only
 * on its own previous load, so N chains offer the memory system exactly N independent misses and
 * nothing else. Then:
 *
 *   cycles per ROUND stays flat as N rises  -> the core overlaps N misses (MSHRs >= N)
 *   cycles per round scales linearly with N -> one miss at a time (MSHRs == 1); no software change
 *                                              and no wire clock will saturate the link from a core
 *
 * This is strictly better evidence than interleaved sequential streams, because a sequential stream
 * mixes in the prefetcher, the row buffer and the loop's own scheduling. It is also independent of
 * the core clock — it is a ratio of cycle counts — so it stays valid while the PLL question is open.
 *
 * `buf` is split into `nchains` disjoint sub-regions of `sub_span` bytes; the caller must have run
 * mem_probe_chase_build() on each one first (they must be built separately or the chains would share
 * a cycle and stop being independent). nchains must be 1, 2, 4 or 8; returns cycles for `steps`
 * rounds, i.e. steps*nchains dependent loads total.
 * ---------------------------------------------------------------------------------------------- */
MEM_PROBE_UNUSED static uint64_t mem_probe_chase_parallel(void *buf, size_t sub_span, int nchains,
                                                          size_t steps, uint32_t *sink) {
  uint8_t *const b = (uint8_t *)buf;
  #define MEM_PROBE_CH(k) (b + (size_t)(k) * sub_span)
  uint32_t i0 = 0, i1 = 0, i2 = 0, i3 = 0, i4 = 0, i5 = 0, i6 = 0, i7 = 0;
  uint64_t c;
  const uint64_t t = mem_probe_rdcycle();
  switch (nchains) {
    case 1:
      for (size_t k = 0; k < steps; k++) i0 = MEM_PROBE_NODE(MEM_PROBE_CH(0), i0);
      break;
    case 2:
      for (size_t k = 0; k < steps; k++) {
        i0 = MEM_PROBE_NODE(MEM_PROBE_CH(0), i0);
        i1 = MEM_PROBE_NODE(MEM_PROBE_CH(1), i1);
      }
      break;
    case 4:
      for (size_t k = 0; k < steps; k++) {
        i0 = MEM_PROBE_NODE(MEM_PROBE_CH(0), i0);
        i1 = MEM_PROBE_NODE(MEM_PROBE_CH(1), i1);
        i2 = MEM_PROBE_NODE(MEM_PROBE_CH(2), i2);
        i3 = MEM_PROBE_NODE(MEM_PROBE_CH(3), i3);
      }
      break;
    default:
      for (size_t k = 0; k < steps; k++) {
        i0 = MEM_PROBE_NODE(MEM_PROBE_CH(0), i0);
        i1 = MEM_PROBE_NODE(MEM_PROBE_CH(1), i1);
        i2 = MEM_PROBE_NODE(MEM_PROBE_CH(2), i2);
        i3 = MEM_PROBE_NODE(MEM_PROBE_CH(3), i3);
        i4 = MEM_PROBE_NODE(MEM_PROBE_CH(4), i4);
        i5 = MEM_PROBE_NODE(MEM_PROBE_CH(5), i5);
        i6 = MEM_PROBE_NODE(MEM_PROBE_CH(6), i6);
        i7 = MEM_PROBE_NODE(MEM_PROBE_CH(7), i7);
      }
      break;
  }
  c = mem_probe_rdcycle() - t;
  *sink += i0 + i1 + i2 + i3 + i4 + i5 + i6 + i7;
  #undef MEM_PROBE_CH
  return c;
}

/* ------------------------------------------------------------------------------------------------
 * Bandwidth: sequential streams
 *
 * All of these take a byte span and return cycles. Scalar variants use 64-bit loads (a byte loop
 * measures the loop, not the memory); the RVV variants issue one load of exactly LMUL*VLEN/8 bytes,
 * which is how the LMUL sweep answers "does this memory system pipeline requests at all?".
 * ---------------------------------------------------------------------------------------------- */

MEM_PROBE_UNUSED static uint64_t mem_probe_read_u64(const void *buf, size_t span, uint64_t *sink) {
  const uint64_t *p = (const uint64_t *)buf;
  const size_t n = (span / 8u) & ~(size_t)7u;   /* whole 8-word groups; the ragged tail is noise */
  uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
  const uint64_t t = mem_probe_rdcycle();
  for (size_t i = 0; i < n; i += 8u) {
    a0 += p[i + 0]; a1 += p[i + 1]; a2 += p[i + 2]; a3 += p[i + 3];
    a0 += p[i + 4]; a1 += p[i + 5]; a2 += p[i + 6]; a3 += p[i + 7];
  }
  const uint64_t c = mem_probe_rdcycle() - t;
  *sink += a0 + a1 + a2 + a3;
  return c;
}

MEM_PROBE_UNUSED static uint64_t mem_probe_write_u64(void *buf, size_t span) {
  uint64_t *p = (uint64_t *)buf;
  const size_t n = span / 8u;
  const uint64_t t = mem_probe_rdcycle();
  for (size_t i = 0; i < n; i++) p[i] = (uint64_t)i * 0x0101010101010101ull;
  const uint64_t c = mem_probe_rdcycle() - t;
  return c;
}

MEM_PROBE_UNUSED static uint64_t mem_probe_copy_u64(void *dst, const void *src, size_t span) {
  uint64_t *d = (uint64_t *)dst;
  const uint64_t *s = (const uint64_t *)src;
  const size_t n = span / 8u;
  const uint64_t t = mem_probe_rdcycle();
  for (size_t i = 0; i < n; i++) d[i] = s[i];
  return mem_probe_rdcycle() - t;
}

/* `nstream` sequential reads interleaved one element at a time, accumulated into ONE variable.
 *
 * KEEP THIS, but do not read it as a memory-level-parallelism measurement — it is the control, not
 * the experiment. Every load is consumed by the very next instruction (`acc += load`), so on an
 * in-order core the pipeline stalls at each use and the misses CANNOT overlap however many MSHRs the
 * D$ has. Adding streams then only costs locality, which is exactly the "parallelism makes it worse"
 * shape (bearly25: 1/2/4/8 = 31.6/66/122/205 cyc/byte). Compare against the _indep variant below:
 * if dependent degrades while independent scales, the earlier result was the loop, not the hardware. */
MEM_PROBE_UNUSED static uint64_t mem_probe_read_streams(const void *buf, size_t span, int nstream, uint64_t *sink) {
  const uint64_t *p = (const uint64_t *)buf;
  const size_t chunk = (span / 8u) / (size_t)nstream;
  uint64_t acc = 0;
  const uint64_t t = mem_probe_rdcycle();
  for (size_t i = 0; i < chunk; i++)
    for (int k = 0; k < nstream; k++) acc += p[(size_t)k * chunk + i];
  const uint64_t c = mem_probe_rdcycle() - t;
  *sink += acc;
  return c;
}

/* The experiment: the same N sequential streams, but every load of a round is ISSUED BEFORE any of
 * them is used, and each lands in its own accumulator. That is what lets an in-order core with a
 * non-blocking D$ have N misses outstanding at once. If this scales and the dependent version above
 * does not, the core does have MSHRs and the fix is in the kernels; if neither scales, the core
 * sustains one miss at a time and no amount of streaming will saturate a faster link. */
MEM_PROBE_UNUSED static uint64_t mem_probe_read_streams_indep(const void *buf, size_t span,
                                                              int nstream, uint64_t *sink) {
  const uint64_t *const p = (const uint64_t *)buf;
  const size_t chunk = (span / 8u) / (size_t)nstream;
  const uint64_t *const p0 = p, *const p1 = p + chunk, *const p2 = p + 2u * chunk;
  const uint64_t *const p3 = p + 3u * chunk, *const p4 = p + 4u * chunk;
  const uint64_t *const p5 = p + 5u * chunk, *const p6 = p + 6u * chunk;
  const uint64_t *const p7 = p + 7u * chunk;
  uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
  const uint64_t t = mem_probe_rdcycle();
  switch (nstream) {
    case 1:
      for (size_t i = 0; i < chunk; i++) a0 += p0[i];
      break;
    case 2:
      for (size_t i = 0; i < chunk; i++) {
        const uint64_t v0 = p0[i], v1 = p1[i];   /* both loads issue before either is used */
        a0 += v0; a1 += v1;
      }
      break;
    case 4:
      for (size_t i = 0; i < chunk; i++) {
        const uint64_t v0 = p0[i], v1 = p1[i], v2 = p2[i], v3 = p3[i];
        a0 += v0; a1 += v1; a2 += v2; a3 += v3;
      }
      break;
    default:
      for (size_t i = 0; i < chunk; i++) {
        const uint64_t v0 = p0[i], v1 = p1[i], v2 = p2[i], v3 = p3[i];
        const uint64_t v4 = p4[i], v5 = p5[i], v6 = p6[i], v7 = p7[i];
        a0 += v0; a1 += v1; a2 += v2; a3 += v3; a4 += v4; a5 += v5; a6 += v6; a7 += v7;
      }
      break;
  }
  const uint64_t c = mem_probe_rdcycle() - t;
  *sink += a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
  return c;
}

/* One 8-byte load every `stride` bytes. Sweeping the stride shows where the line size is (cost per
 * ACCESS is flat until the stride exceeds one line, then flat again at the miss cost) and whether
 * anything is prefetching. Returns cycles; the caller divides by span/stride for cycles per access. */
MEM_PROBE_UNUSED static uint64_t mem_probe_read_stride(const void *buf, size_t span, size_t stride, uint64_t *sink) {
  const uint8_t *p = (const uint8_t *)buf;
  uint64_t acc = 0;
  const uint64_t t = mem_probe_rdcycle();
  for (size_t off = 0; off + 8u <= span; off += stride) acc += *(const uint64_t *)(p + off);
  const uint64_t c = mem_probe_rdcycle() - t;
  *sink += acc;
  return c;
}

#if MEM_PROBE_RVV
/* Sequential read with one vector load per iteration at LMUL 1/2/4/8, i.e. VLEN/8, VLEN/4, VLEN/2
 * and VLEN bytes per request (32/64/128/256 B at VLEN=256). bearly25 measured 49.45 cyc/byte at 32 B
 * and 31.64 at 64 B with nothing beyond — one full line per request is the floor there. */
MEM_PROBE_UNUSED static uint64_t mem_probe_read_rvv(const void *buf, size_t span, int lmul, uint64_t *sink) {
  const int8_t *p = (const int8_t *)buf;
  uint64_t c;
  int32_t acc = 0;
  size_t vl;
  const uint64_t t = mem_probe_rdcycle();
  switch (lmul) {
    case 1: {
      vl = __riscv_vsetvlmax_e8m1();
      vint8m1_t v = __riscv_vmv_v_x_i8m1(0, vl);
      for (size_t i = 0; i + vl <= span; i += vl) v = __riscv_vadd_vv_i8m1(v, __riscv_vle8_v_i8m1(p + i, vl), vl);
      acc = __riscv_vmv_x_s_i8m1_i8(v);
    } break;
    case 2: {
      vl = __riscv_vsetvlmax_e8m2();
      vint8m2_t v = __riscv_vmv_v_x_i8m2(0, vl);
      for (size_t i = 0; i + vl <= span; i += vl) v = __riscv_vadd_vv_i8m2(v, __riscv_vle8_v_i8m2(p + i, vl), vl);
      acc = __riscv_vmv_x_s_i8m2_i8(v);
    } break;
    case 4: {
      vl = __riscv_vsetvlmax_e8m4();
      vint8m4_t v = __riscv_vmv_v_x_i8m4(0, vl);
      for (size_t i = 0; i + vl <= span; i += vl) v = __riscv_vadd_vv_i8m4(v, __riscv_vle8_v_i8m4(p + i, vl), vl);
      acc = __riscv_vmv_x_s_i8m4_i8(v);
    } break;
    default: {
      vl = __riscv_vsetvlmax_e8m8();
      vint8m8_t v = __riscv_vmv_v_x_i8m8(0, vl);
      for (size_t i = 0; i + vl <= span; i += vl) v = __riscv_vadd_vv_i8m8(v, __riscv_vle8_v_i8m8(p + i, vl), vl);
      acc = __riscv_vmv_x_s_i8m8_i8(v);
    } break;
  }
  c = mem_probe_rdcycle() - t;
  *sink += (uint64_t)(uint32_t)acc;
  return c;
}

/* Vector store bandwidth, LMUL=8 (widest request this core can make). */
MEM_PROBE_UNUSED static uint64_t mem_probe_write_rvv(void *buf, size_t span) {
  int8_t *p = (int8_t *)buf;
  const size_t vl = __riscv_vsetvlmax_e8m8();
  const vint8m8_t v = __riscv_vmv_v_x_i8m8(0x5a, vl);
  const uint64_t t = mem_probe_rdcycle();
  for (size_t i = 0; i + vl <= span; i += vl) __riscv_vse8_v_i8m8(p + i, v, vl);
  return mem_probe_rdcycle() - t;
}
#endif /* MEM_PROBE_RVV */

/* ------------------------------------------------------------------------------------------------
 * Integer formatting helpers
 * ---------------------------------------------------------------------------------------------- */

/* cycles per byte, x100 */
static inline uint64_t mem_probe_cpb_x100(uint64_t cycles, uint64_t bytes) {
  return bytes ? (cycles * 100ull) / bytes : 0ull;
}

/* MB/s (10^6 B/s), x100, from the MEASURED core clock. 0 if the clock is unknown. */
static inline uint64_t mem_probe_mbps_x100(uint64_t cycles, uint64_t bytes, uint64_t core_hz) {
  if (!cycles || !core_hz) return 0ull;
  return (bytes * core_hz) / cycles / 10000ull;
}

/* ------------------------------------------------------------------------------------------------
 * Little's Law, in cycles
 *
 * BW = lines_in_flight * LINE / RTT, so lines_in_flight = BW * RTT / LINE. Working in CYCLES rather
 * than seconds makes this independent of the core clock — which matters here, because the clock is
 * exactly the thing that is in doubt, and the "can this link ever be saturated" question must not
 * wait on resolving it. Both inputs come straight out of this header: `rtt_cyc` is the flat tail of
 * the pointer-chase curve, `cpb_x100` the streaming read cost.
 *
 * Bytes delivered during one RTT = rtt_cyc / cyc_per_byte, so lines = that / 64.
 * Returns lines in flight, x100. A result near 100 means one miss at a time.
 * ---------------------------------------------------------------------------------------------- */
static inline uint64_t mem_probe_lines_in_flight_x100(uint64_t rtt_cyc, uint64_t cpb_x100) {
  if (!cpb_x100) return 0ull;
  return (rtt_cyc * 100ull * 100ull) / cpb_x100 / MEM_PROBE_LINE_BYTES;
}

/* The inverse: how many lines must be in flight to sustain `target_mbps` at `core_hz`, given an RTT
 * of `rtt_cyc`. This is the bar a link-saturation experiment has to clear. Returns lines, x100.
 * Needs the clock (a byte rate is a rate), which is why membw prints it for several candidates. */
static inline uint64_t mem_probe_lines_needed_x100(uint64_t rtt_cyc, uint64_t target_mbps,
                                                   uint64_t core_hz) {
  if (!core_hz) return 0ull;
  /* bytes per RTT = target_mbps*1e6 * rtt_cyc/core_hz ; lines = /64 */
  return (target_mbps * 1000000ull / MEM_PROBE_LINE_BYTES) * rtt_cyc * 100ull / core_hz;
}

/* nanoseconds per access, x10, from cycles-per-access-x100 and the measured core clock. */
static inline uint64_t mem_probe_ns_x10(uint64_t cyc_x100, uint64_t core_hz) {
  if (!core_hz) return 0ull;
  return cyc_x100 * 100000000ull / core_hz;
}

/* One bandwidth line: "<label>   12345 Kcyc   31.64 cyc/byte    23.72 MB/s". */
MEM_PROBE_UNUSED static void mem_probe_print_bw(const char *label, uint64_t cycles, uint64_t bytes, uint64_t core_hz) {
  const uint64_t cpb = mem_probe_cpb_x100(cycles, bytes);
  const uint64_t bw = mem_probe_mbps_x100(cycles, bytes, core_hz);
  printf("[mem] %-22s %9lu Kcyc  %4lu.%02lu cyc/byte  %6lu.%02lu MB/s\r\n", label,
         (unsigned long)(cycles / 1000ull),
         (unsigned long)(cpb / 100ull), (unsigned long)(cpb % 100ull),
         (unsigned long)(bw / 100ull), (unsigned long)(bw % 100ull));
}

/* One latency line: "<label>    214.50 cyc/load    286.0 ns". */
MEM_PROBE_UNUSED static void mem_probe_print_lat(const char *label, uint64_t cycles, uint64_t accesses,
                                uint64_t core_hz) {
  const uint64_t cy = accesses ? (cycles * 100ull) / accesses : 0ull;
  const uint64_t ns = mem_probe_ns_x10(cy, core_hz);
  printf("[mem] %-22s %9lu Kcyc  %4lu.%02lu cyc/load  %6lu.%01lu ns\r\n", label,
         (unsigned long)(cycles / 1000ull),
         (unsigned long)(cy / 100ull), (unsigned long)(cy % 100ull),
         (unsigned long)(ns / 10ull), (unsigned long)(ns % 10ull));
}

/* ------------------------------------------------------------------------------------------------
 * Compact summary — three lines, for demos that want a figure of merit next to their real workload
 * rather than a full sweep. Costs ~1 s.
 *
 * `buf`/`bytes` must be WRITABLE (the chase is built in place). `bytes` should be several times the
 * last-level cache; 16 MB is plenty against a 256 KB L2. `core_hz` should come from
 * mem_probe_core_hz() — pass 0 and the MB/s and ns columns read 0.
 * ---------------------------------------------------------------------------------------------- */
MEM_PROBE_UNUSED static void mem_probe_summary(void *buf, size_t bytes, uint64_t core_hz) {
  uint64_t sink = 0;
  uint32_t isink = 0;

  /* Cached reference point: a span that fits in L1 measures the pipeline, not the bus, so it should
   * NOT move when the memory clock changes. If it does, the core clock moved too and every other
   * number needs re-reading. */
  const size_t small = 4u << 10;
  if (bytes >= small) {
    const size_t nodes = mem_probe_chase_build(buf, small, 0xC0FFEEu);
    if (nodes) {
      const size_t steps = 200000u;
      mem_probe_chase_run(buf, 1000u, &isink);            /* warm */
      mem_probe_print_lat("latency  4 KB (L1)", mem_probe_chase_run(buf, steps, &isink), steps, core_hz);
    }
  }

  /* The real number: random dependent loads over a span far past any cache. */
  const size_t big = (bytes < (16u << 20)) ? bytes : (16u << 20);
  const size_t nodes = mem_probe_chase_build(buf, big, 0xBEEF01u);
  if (nodes) {
    const size_t steps = 100000u;
    char lbl[32];
    snprintf(lbl, sizeof(lbl), "latency %2lu MB (DRAM)", (unsigned long)(big >> 20));
    mem_probe_print_lat(lbl, mem_probe_chase_run(buf, steps, &isink), steps, core_hz);
  }

  /* Streaming read, the quantity a weight-bandwidth-bound model actually pays. */
  mem_probe_print_bw("seq read (u64)", mem_probe_read_u64(buf, big, &sink), big, core_hz);
#if MEM_PROBE_RVV
  mem_probe_print_bw("seq read (rvv m8)", mem_probe_read_rvv(buf, big, 8, &sink), big, core_hz);
#endif
  (void)sink;
  (void)isink;
}

#endif /* MEM_PROBE_H */
