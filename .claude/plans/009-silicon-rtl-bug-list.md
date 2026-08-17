# 009 — Silicon defects to reproduce in RTL simulation

A hit list for the next RTL/VCS session. Everything here was observed on **real silicon** and either
does not reproduce in Spike or is invisible there by construction. Ordered by how much they cost us
and how cheaply they should reproduce.

Each entry gives the observation, the *discriminating* evidence (what narrows it to specific
hardware rather than "something is wrong"), and a directed test to run in RTL. Where an entry is
inference rather than measurement, it says so.

**Read the "Known Chip Bugs & Quirks" log in CLAUDE.md alongside this** — that is the living record;
this file is the subset worth chasing in simulation, with repro steps attached.

---

## 1. `vwredsum.vs` (or `vmv.x.s` after it) returns garbage — bearly25

**Severity: highest.** It silently produces fluent-looking wrong answers, and it has now bitten two
unrelated demos.

**Observed (2026-08-15, `c2c-demos/bearly-smollm` on-chip benchmark).** Two int8 matmul kernels over
the *same* 1536×576 weight tensor and the *same* activations, cross-checked against a scalar
reference on the same chip in the same boot:

```
[bench] matmul scalar    54075 Kcyc   61.12 cyc/byte  (reference)
[bench] matmul rowdot    46530 Kcyc   52.59 cyc/byte  max_diff 2147483647/1e6   <-- garbage
[bench] matmul lane      45629 Kcyc   51.57 cyc/byte  max_diff 0/1e6
[bench] matmul transp      534 Kcyc    0.60 cyc/byte  max_diff 0/1e6
```

`max_diff 2147483647` is a saturated int cast of a huge float difference — the row-dot result is not
slightly wrong, it is unrelated to the correct answer. In the full demo it produced
`"atimstakingstaking elephstakingarxiv kilow"` where the scalar build produced
`"hello! how can i help you today?"`.

**What this narrows it to.** The failing and working kernels differ by exactly two instructions:

| | row-dot (WRONG) | lane-float (CORRECT) |
|---|---|---|
| widening multiply | `vwmul.vv` i8→i16 | `vwmul.vv` i8→i16 |
| accumulate | **`vwredsum.vs` i16m2→i32m1** | `vwcvt.x.x.v`, `vfcvt.f.x.v`, `vfmacc.vf` |
| reduce / extract | **`vmv.x.s` i32m1→x** | `vfredusum.vs` f32m4→f32m1, `vfmv.f.s` |

So `vwmul.vv`, `vle8.v`, `vadd.vv`, `vfredusum.vs` and `vfmv.f.s` are all **exonerated** — they are
exercised heavily by the passing kernels. The suspects are exactly:

- **`vwredsum.vs`** — integer widening reduction (i16 elements into an i32 accumulator)
- **`vmv.x.s`** — vector-to-scalar integer move, possibly reading before the reduction retires

Note the float reduction (`vfredusum.vs`) works, so "reductions" as a class are not broken — it is
the *integer widening* reduction, or the scalar readout after it.

**Almost certainly the same bug as the TinySpeech int8 conv2 defect** already in CLAUDE.md: that one
reports the conv2 activation maximum coming back as "deterministic garbage ≈ `INT32_MAX`" from an
RVV microkernel, with the float pipeline used as the workaround. Same signature, same instruction
family, different demo, months apart. Treat them as one investigation.

**Directed RTL test.**
1. Minimal: `vsetvli` e8m1 → `vle8.v` two known vectors → `vwmul.vv` → `vwredsum.vs` into a zeroed
   i32m1 → `vmv.x.s`. Compare against Spike for the same inputs. Sweep `vl` from 1 to vlmax, and
   VLEN-relevant LMUL combinations (i16m2→i32m1 is what the kernel uses).
2. Split the suspects: after `vwredsum.vs`, store the accumulator with `vse32.v` instead of reading
   it with `vmv.x.s`. If the stored value is correct, the reduction is fine and `vmv.x.s` is the bug
   (a forwarding/hazard problem); if the stored value is already wrong, the reduction is the bug.
3. Back-to-back stress: the kernel issues ~2.1M of these per token with a dependent scalar consumer
   immediately after, so try tight repetition — a hazard that only shows under back-to-back issue
   would explain why small tests pass.

**Cheap repro on hardware, no RTL needed:** build `bearly-smollm` with
`-DSMOLLM_BENCH_KERNELS=ON`; the `max_diff` columns above print at boot in about a second.

---

## 2. Cross-hart plain memory is not coherent — dsp25

**Severity: high** — it is why dual-core was abandoned in `dsp-moonshine` (2026-08-10).

**Observed (2026-08-09, `c2c-demos/dsp-moonshine`).** With a *correct* SEQ_CST release/acquire pair,
hart 1 read a stale all-zero (`.bss`-initial) job descriptor: `mcause=5 mepc=0x80000d5e mtval=0x20`,
where offset `0x20` is the first field dereferenced through the shared pointer, i.e. it saw `NULL`
where hart 0 had published a real pointer. On the output side a probe read back `0` where hart 0 had
most recently written `-1e30` — **a value from two writes earlier resurrecting**.

**What this narrows it to.** AMO-accessed fields (`go`/`done`/`alive`/`stage`) were reliable
throughout the same runs; only plain cached lines went stale. So AMOs are performed at a coherent
point and ordinary lines are not — i.e. the private caches are not participating in coherence the
way the ISA requires, rather than a fence/ordering bug in software (a missing release fence *was*
found and fixed separately, and the failure persisted after).

Non-deterministic run to run and eviction-timing dependent, which is why small probes pass and large
working sets fail. Same family as the C2C shared-region quirk (entry 5).

**Directed RTL test.** Two harts, one cache line: hart 0 writes a payload then an AMO flag; hart 1
spins on the flag with an acquire and reads the payload. Vary payload size to force evictions and
check whether hart 1's L1 ever returns a pre-write value after the AMO is visible. Then check
whether hart 1's L1 is snooped/invalidated at all on hart 0's writeback.

**Second, possibly separate dual-core symptom:** hart 1 entered a large RVV convolution
(`conv_worker`, 288ch/k127/stride64) and never returned — `DC-WEDGE taskno=1 kind=conv stage=1 go=1
done=0` on the very first fork, with the handshake flags coherent (hart 0 saw `stage=1`). Suspected
strided `vlse` vector state on hart 1. Worth a directed test of sustained vector work on the
non-boot hart.

---

## 2b. ccache WayEnable reads back 0 — NOT a defect (resolved 2026-08-15)

Recorded so nobody re-investigates it. The probe reported:

```
[probe] ccache Config=0x06080802 -> banks=2 ways=8 sets=256 block=64 B => 256 KB total
[probe] ccache WayEnable=0 ; writing 7 read back 0
```

which looked like "7 of 8 ways are disabled". **It is not.** This is the open-source SiFive cache,
where all ways are always active and `WayEnable` is not the gating control it is on the commercial
part. The full 256 KB is in use. The write being ignored is consistent with the register simply not
being implemented here.

The one thing still worth a glance in RTL: the **flush** register at `0x02010200` also does nothing
(long-standing entry in CLAUDE.md), so the register block's write path is worth a look — but the
cache capacity is fine and nothing above depends on it.

## 3. DRAM read throughput anomaly — bearly25

**Severity: it is the entire performance story for LLM work**, but it may be expected behaviour for
this uncore/FPGA configuration rather than a defect. Confirm before chasing.

**Observed (2026-08-15).** Sequential reads of the `.incbin`'d model in DRAM `.rodata`:

```
scalar read (byte loop)   54.72 cyc/byte   ~3500 cycles per 64-byte line
vector read (vle8, 32B)   31.30 cyc/byte   ~2000 cycles per 64-byte line
cache-resident matmul      0.60 cyc/byte   ~38 cycles per line
```

A full model pass is 143 MB, so 55 cyc/byte × 143 MB = ~7.9 Gcyc ≈ 10.5 s per token at 750 MHz —
which matches the measured 8523 Mcyc/token exactly. The arithmetic is free; this number is the
product.

**Measured answers (2026-08-15), which settle most of it:**

- **It is not `.rodata`.** An 8 MB heap read is 55.41 cyc/byte against the blob's 55.19 — identical.
- **Reads DO allocate into the cache.** A 64 KB re-read drops 55.5 -> 8.0 cyc/byte.
- **One request returns one 64-byte line, and requests do not overlap.** Load-width sweep:
  `LMUL=1 (32 B) 49.29 | LMUL=2 (64 B) 31.65 | LMUL=4 31.64 | LMUL=8 31.93` — asking for a full
  line is worth 1.56x, and asking for more than a line is worth nothing.
- **Concurrency makes it worse, monotonically:** 1/2/4/8 interleaved streams give
  56.8 / 66.5 / 122.0 / 204.7 cyc/byte. With only one way of L2 enabled (entry 2b) multiple streams
  have nowhere to live, so this may be a consequence of that defect rather than an independent one —
  worth re-measuring if the way-enable write path is ever fixed.

So the streaming floor is **~31.6 cyc/byte (~24 MB/s at 750 MHz)** and the core cannot have two
line-fills in flight. That is the number that makes a 136 MB model cost ~4.3 Gcyc/token no matter
how good the kernel is.

**Directed RTL test.** Count outstanding misses at the L1/L2 boundary during a sequential stream:
is it ever more than one? Check whether any prefetcher exists and whether it is enabled (see also
the SiFive feature-disable CSR, which this probe has not yet read).

**Directed RTL test.** Stream a few MB sequentially and count outstanding misses at the L1/L2
boundary; check whether more than one line fill is ever in flight, and whether a hardware prefetcher
exists/fires on a pure sequential pattern.

---

## 4. A bad access hangs the core silently instead of trapping — dsp25

**Observed (plan 007, `dsp-citrinet`).** Writing into address space the SoC does not back **hangs
the core outright** — no trap, no `trap_handler` print — so it presents as an infinite loop in
whatever ran last. Measured with a stack placed past the end of real DRAM: at `0xFFFFB000` stores
silently failed to read back; at `0xFFEC0000` the write hung the core.

There is also a partially-backed band near the top of the 32-bit space where **stores do not stick
but do not fault** — reads return non-pattern garbage. That asymmetry (register values stay correct,
only memory-backed automatics corrupt) is what made the original bug take a week.

**Directed RTL test.** Write to an unmapped/undecoded address and observe the bus response: is an
error response generated at all, and does the core's LSU handle it or wait forever? Same for reads
into the partially-backed band.

---

## 5. C2C link quirks (all worked around, all worth confirming in RTL) — both chips

From the CLAUDE.md log; these shape every line of C2C code we write, so it is worth knowing which
are RTL defects and which are as-designed:

- **Scratchpads are 32-bit-access-only.** A byte-granular store to `0xC0000000`/`0xD0000000` **hangs
  the core**. Word accesses are fine. (Hanging rather than trapping is entry 4 again.)
- **Cross-spad (remote) writes are unstable** — a single write may not take; we repeat every remote
  write N times. Never quantified how many repeats are actually needed.
- **A cross-link write into a chip while it is BOOTING kills that chip** — a *single* write is
  enough, and it is symmetric. Needs a boot barrier.
- **A cross-link write to an absent/wedged peer hangs the writer** (entry 4 again).
- **Cross-chip MSIP wake can drop**, and a dropped wake into a `wfi`-sleeping core is unrecoverable
  without the timer safety net.
- **Writing the cache-controller flush register (`0x02010200`) does not evict** — only the 256 KiB
  buffer-walk force-eviction works. This one is a concrete, easily-simulated register-behaviour
  question.

---

## 6. DMA strided gather corrupts — dsp25

**Observed (`dsp-citrinet`).** Contiguous DMA transfers into the 64 KB scratchpad work and were
timed; the **strided gather corrupts** and was disqualified. Also measured absurdly slow into the
scratchpad (0.0022 B/cycle, 72× slower per byte than DRAM→DRAM), which is its own question.

**Directed RTL test.** A strided descriptor with a small element size and a large stride, checked
byte-for-byte against the expected gather.

---

## 7. I2S HAL watermark hangs the core — dsp25

**Observed** during I2S bring-up: a watermark configuration hangs the core (recorded in the I2S
notes). Unclear whether this is a HAL misuse or a FIFO/watermark RTL issue; a directed test of the
watermark comparison at the boundary values would settle it.

---

## Reproducing #1 and #3 without RTL

Both print at boot from one image, in about a second:

```bash
make build CHIP=bearly25 PLATFORM=CHIP TARGET=bearly-smollm \
  EXTRA_CMAKE_ARGS="-DLINKER=llm -DSMOLLM_CHECK_BLOB=ON -DSMOLLM_BENCH_KERNELS=ON"
```

`smollm_bench()` in `c2c-demos/bearly-smollm/src/model.c` is deliberately chip-free and
self-checking: every kernel is compared against the scalar result computed on the same chip in the
same boot, so "the vector unit is wrong" cannot be confused with "the weights are corrupt" (which
the FNV-1a blob check rules out separately).
