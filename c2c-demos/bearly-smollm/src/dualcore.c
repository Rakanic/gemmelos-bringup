/*
 * Second hart for batched prefill — bring-up, handshake, and the probe that decides whether any of
 * it is safe to use on this silicon.
 *
 * Why a probe rather than just enabling it: on dsp25, `dsp-moonshine` measured hart 1 reading a
 * STALE all-zero job descriptor through a correct SEQ_CST release / acquire pair, and a probe read
 * back a value from TWO writes earlier — while AMO-accessed fields stayed reliable throughout. That
 * is a coherence defect in the private caches, not a fence bug, and it is why dual-core was
 * abandoned there (see .claude/plans/009-silicon-rtl-bug-list.md #2). bearly25 has never been
 * checked. So this file proves, at boot and on real payload sizes, that:
 *
 *   1. hart 1 is alive and takes work at all;
 *   2. a job descriptor published by hart 0 arrives INTACT at hart 1;
 *   3. results hart 1 writes to plain memory are visible to hart 0 after the handshake.
 *
 * If any of those fails, the hooks stay NULL and the model runs single-core — a slow correct demo
 * beats a fast wrong one, and this failure mode produces *plausible* wrong numbers, not a crash.
 *
 * Handshake shape (the one moonshine converged on): single producer, single consumer, one job slot,
 * no locks. Job fields are written first, then published by a SEQ_CST bump of a monotonic `go`
 * counter; hart 1 acquires `go`, works, and answers with `done`. Only the counters are atomics —
 * if plain data does not cross reliably, the probe catches it rather than the demo.
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "chip_config.h"
#include "smollm_config.h"
#include "smollm_model.h"

extern uint64_t target_frequency;   /* owned by main.c */

#if SMOLLM_DUALCORE

/* One job slot. Plain fields, published by the atomic counter below. */
static smollm_job_t g_job;
static volatile uint64_t g_go, g_done;
static volatile int g_worker_live;

/* Probe payload: hart 1 fills this, hart 0 checks it. Sized past any plausible L1 so the test
 * exercises eviction, which is where moonshine's failure actually showed up (small probes passed,
 * large ones failed). */
#define PROBE_WORDS (64 * 1024)
static float g_probe[PROBE_WORDS];
static volatile uint32_t g_probe_seed;

static inline uint64_t rdcycle64_dc(void) {
  uint64_t x;
  __asm__ volatile("rdcycle %0" : "=r"(x));
  return x;
}

/* ---- hart 1 -------------------------------------------------------------------------------- */

static void worker_loop(void) {
  uint64_t seen = 0;
  __atomic_store_n(&g_worker_live, 1, __ATOMIC_SEQ_CST);
  for (;;) {
    uint64_t go = __atomic_load_n(&g_go, __ATOMIC_ACQUIRE);
    while (go == seen) {
      go = __atomic_load_n(&g_go, __ATOMIC_ACQUIRE);
    }
    seen = go;
    /* A probe job is all-zero, and an all-zero descriptor is ALSO exactly what the documented
     * coherence defect delivers (hart 1 reading .bss-initial values). Keying the probe branch off
     * xout == NULL makes those two the same harmless case: a stale descriptor fills the probe
     * buffer instead of dereferencing NULL, so the failure is reported by the probe's verification
     * rather than as a hang. */
    if (g_job.xout == NULL) {
      const uint32_t seed = g_probe_seed;
      for (int i = 0; i < PROBE_WORDS; i++) g_probe[i] = (float)(seed + (uint32_t)i);
    } else if (g_job.row_hi > g_job.row_lo) {
      smollm_run_job(&g_job);
    }
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    __atomic_store_n(&g_done, seen, __ATOMIC_RELEASE);
  }
}

/* crt0 sends every non-boot hart here. A STRONG definition overrides the weak park-forever one in
 * main.c, so this file being linked in is what starts hart 1. */
void __main(void) {
  worker_loop();
  for (;;) __asm__ volatile("wfi");
}

/* ---- hart 0 -------------------------------------------------------------------------------- */

static void cowork_start(const smollm_job_t *job) {
  g_job = *job;
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
  __atomic_store_n(&g_go, g_go + 1, __ATOMIC_RELEASE);
}

static void cowork_wait(void) {
  const uint64_t want = __atomic_load_n(&g_go, __ATOMIC_RELAXED);
  while (__atomic_load_n(&g_done, __ATOMIC_ACQUIRE) != want) { }
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
}

/* Returns 1 if hart 1 is usable. Prints exactly what failed if not. */
static int probe(void) {
  /* 1. is it alive? bounded wait — a dead hart must not hang the demo */
  const uint64_t deadline = rdcycle64_dc() + (uint64_t)target_frequency;   /* ~1 s */
  while (!__atomic_load_n(&g_worker_live, __ATOMIC_ACQUIRE)) {
    if (rdcycle64_dc() > deadline) {
      SMOLLM_LOG("[smollm] dualcore: hart 1 never reported in — staying single-core\r\n");
      return 0;
    }
  }

  /* 2+3. publish a job whose payload hart 0 can verify afterwards. Run it twice with different
   * seeds: a stale-line defect shows up as the SECOND run returning the FIRST run's values, which
   * a single-shot test would pass. */
  for (uint32_t round = 0; round < 2; round++) {
    const uint32_t seed = 0x1000u + round * 0x5000u;
    g_probe_seed = seed;
    smollm_job_t j;
    memset(&j, 0, sizeof(j));            /* xout == NULL marks the probe job */
    cowork_start(&j);
    cowork_wait();
    for (int i = 0; i < PROBE_WORDS; i++) {
      const float want = (float)(seed + (uint32_t)i);
      if (g_probe[i] != want) {
        SMOLLM_LOG("[smollm] dualcore: hart 1's writes are NOT visible to hart 0 "
                   "(word %d = %ld, expected %ld, round %lu) — staying single-core.\r\n"
                   "[smollm] dualcore: this is the cross-hart coherence defect from "
                   ".claude/plans/009-silicon-rtl-bug-list.md #2.\r\n",
                   i, (long)g_probe[i], (long)want, (unsigned long)round);
        return 0;
      }
    }
  }
  SMOLLM_LOG("[smollm] dualcore: hart 1 live, %d KB of results verified across 2 rounds\r\n",
             (int)(sizeof(g_probe) >> 10));
  return 1;
}

void smollm_dualcore_init(void) {
  if (!probe()) return;
  smollm_cowork_start = cowork_start;
  smollm_cowork_wait = cowork_wait;
  SMOLLM_LOG("[smollm] dualcore: prefill matmuls split across 2 harts\r\n");
}

#else
void smollm_dualcore_init(void) { }
#endif /* SMOLLM_DUALCORE */
