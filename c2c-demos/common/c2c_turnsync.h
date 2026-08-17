#ifndef C2C_TURNSYNC_H
#define C2C_TURNSYNC_H

/*
 * c2c_turnsync — reliable turn-taking synchronization layer for the SP25 C2C link.
 *
 * This is the pattern proven on silicon in the hello-wfi demos (2026-07-12), factored for reuse
 * (KWS demos and beyond). See /CLAUDE.md "Reliable C2C turn-taking synchronization" for the full
 * rationale. Three layers, each covering a distinct failure mode:
 *
 *   1. TURN REGISTER (correctness): a word in each spad says whose turn it is. A chip reads it from
 *      its OWN spad and acts only when it equals its own id; any other wake goes back to sleep. So
 *      spurious / duplicate / early wakes never cause double-processing.
 *   2. CLINT TIMER (liveness): both cores arm a periodic machine-timer interrupt alongside MSIE, so
 *      a sleeper re-checks its turn register at least every C2C_POLL_INTERVAL_TICKS even if the wake
 *      MSIP was dropped. A dropped MSIP costs latency, not a deadlock.
 *   3. HARDENED WRITES (delivery): every cross-link store goes through c2c_shm (repeated + flushed).
 *
 * Cross-link addressing (see /CLAUDE.md): reach the peer's CLINT MSIP by prepending a leading 1 to
 * the local MSIP address. CLINT layout relative to CLINT_BASE: MSIP @ +0x0000, mtimecmp @ +0x4000,
 * mtime @ +0xBFF8. MTIME_FREQ = 50 kHz (20 us/tick).
 *
 * Built on c2c_shm (c2c_full_flush / c2c_remote_write_u32 / c2c_local_read_u32 / c2c_fence_rw), so
 * it shares the one flush implementation and the repeat/flush write discipline.
 */

#include <stdint.h>

#include "chip_config.h"   /* CLINT_BASE */
#include "c2c_shm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Own CLINT MSIP (hart 0, CLINT offset 0). */
#ifndef C2C_OWN_MSIP_ADDR
#define C2C_OWN_MSIP_ADDR ((uintptr_t)CLINT_BASE + 0x0000U)
#endif
/* Peer CLINT MSIP across the link (own MSIP with a leading 1). */
#ifndef C2C_PEER_MSIP_ADDR
#define C2C_PEER_MSIP_ADDR 0x102000000ULL
#endif
/* CLINT hart-0 timer registers. */
#ifndef C2C_MTIMECMP_ADDR
#define C2C_MTIMECMP_ADDR ((uintptr_t)CLINT_BASE + 0x4000U)
#endif
#ifndef C2C_MTIME_ADDR
#define C2C_MTIME_ADDR ((uintptr_t)CLINT_BASE + 0xBFF8U)
#endif

/* Safety-net timer period in mtime ticks. MTIME_FREQ = 50 kHz -> 2500 ticks ~= 50 ms. */
#ifndef C2C_POLL_INTERVAL_TICKS
#define C2C_POLL_INTERVAL_TICKS 2500ULL
#endif

/* Turn-register values (role-based). */
#define C2C_TURN_DSP 0u
#define C2C_TURN_BML 1u

static inline uint64_t c2c_mtime(void) {
  return *(volatile uint64_t *)(uintptr_t)(C2C_MTIME_ADDR);
}

static inline void c2c_set_mtimecmp(uint64_t v) {
  *(volatile uint64_t *)(uintptr_t)(C2C_MTIMECMP_ADDR) = v;
  c2c_fence_rw();
}

/* Arm BOTH wake sources: MSIE (fast peer wake) and MTIE (periodic safety net), with mstatus.MIE
 * cleared so neither traps — wfi resumes and we poll. Call once, after boot, before the wait loop. */
static inline void c2c_arm_wake(void) {
  __asm__ volatile("csrc mstatus, %0" :: "r"(1UL << 3) : "memory");            /* mstatus.MIE = 0 */
  __asm__ volatile("csrs mie, %0" :: "r"((1UL << 3) | (1UL << 7)) : "memory"); /* MSIE + MTIE   */
}

/* Clear our own CLINT MSIP (idempotent) + full flush. */
static inline void c2c_clear_own_msip(void) {
  volatile uint32_t *m = (volatile uint32_t *)(uintptr_t)(C2C_OWN_MSIP_ADDR);
  *m = 0u;
  c2c_fence_rw();
  c2c_full_flush();
}

/* Wake the peer: cross-link write of 1 to its CLINT MSIP (repeated via c2c_shm to defeat drops). */
static inline void c2c_wake_peer(void) {
  c2c_remote_write_u32((volatile uint32_t *)(uintptr_t)(C2C_PEER_MSIP_ADDR), 1u);
}

/* Sleep in wfi for `ticks` mtime ticks OR until an MSIP wake, whichever is first. Re-arming
 * mtimecmp each call schedules the next tick and clears any pending MTIP; clear our own MSIP after
 * waking so a stale pending bit can't spin us.
 *
 * The interval is a parameter because the right value depends on what is being waited FOR, and the
 * cost of guessing low is not small: every wake costs at least one c2c_full_flush (a 256 KiB
 * buffer walk), so a few-millisecond interval means the core is walking the cache continuously
 * rather than sleeping. The timer is only a SAFETY NET for a dropped MSIP — the peer's wake still
 * arrives immediately whatever this is set to — so a waiter expecting a slow peer should pick an
 * interval measured against that peer's latency, not against the default. */
static inline void c2c_sleep_ticks(uint64_t ticks) {
  c2c_set_mtimecmp(c2c_mtime() + ticks);
  __asm__ volatile("wfi");
  c2c_clear_own_msip();
}

/* The default-cadence form, unchanged, as used by the KWS demos. */
static inline void c2c_sleep_until_tick(void) {
  c2c_sleep_ticks((uint64_t)C2C_POLL_INTERVAL_TICKS);
}

/* Block (timer-paced wfi) until the turn register in our OWN spad equals my_turn. Each wake (MSIP
 * or timer) flushes and re-reads the register; if it is not yet ours we sleep again. */
static inline void c2c_await_turn(volatile const uint32_t *own_turn, uint32_t my_turn) {
  while (c2c_local_read_u32(own_turn) != my_turn) {
    c2c_sleep_until_tick();
  }
}

#ifdef __cplusplus
}
#endif

#endif /* C2C_TURNSYNC_H */
