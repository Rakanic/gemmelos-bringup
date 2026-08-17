# membw — memory latency / bandwidth sweep

Standalone ELF that measures what the memory system actually delivers, so a bitstream change (bus
frequency, cache configuration, DRAM controller settings) can be attributed with numbers instead of
inferred from a model's tok/s.

It exists because every heavy workload on these chips is memory bound, not compute bound: bearly25
measured **~31.6 cycles per byte** to stream DRAM and **one line fill at a time** (2/4/8 concurrent
read streams got monotonically *worse*). SmolLM's 143 MB of weights therefore cost ~4.5 Gcyc/token
no matter which matmul kernel runs. See the entries in `CLAUDE.md`.

## Build & run

Builds for either chip — it touches no accelerator, only DRAM:

```bash
make build CHIP=dsp25    PLATFORM=CHIP TARGET=membw
make build CHIP=bearly25 PLATFORM=CHIP TARGET=membw
make tsi-run TTY=<tty> BINARY=build/c2c-demos/membw/membw.elf
```

The ELF is ~57 KB of text and ~210 KB of image: the test window is a fixed **absolute** DRAM range,
not a `.bss` array or a heap allocation, so `uart_tsi` has nothing large to zero-fill and a flash
takes a second. (`__heap_size` is pinned to 64 KiB for the same reason — the default 1 MB heap alone
would add ~12 s to every load, for memory this program never allocates from.)

> If you build into a build directory that was previously configured for something else, pass
> `-DLINKER=chip`. `LINKER` is a CMake **cache** variable and the top-level `Makefile` does not
> forward it, so a stale `LINKER=llm` survives a change of `CHIP` and silently selects the wrong
> linker script.

## The question it is built to answer

Can a faster off-chip link ever be *visible from the core*? Little's Law fixes the shape:

```
BW = lines_in_flight × 64 B / RTT
```

so it turns on exactly two measured numbers — the RTT (section 3) and how many misses the core
actually keeps outstanding (section 4). **Section 4 is the gate.** If the core sustains one miss at a
time, no wire clock and no kernel change can saturate the link from the core, and an ILA on the
serial TL port counting `out.valid && out.ready` is the only way to see a difference. Section 6
prints both numbers and the bar to clear, so the run itself says which world you are in.

## What it reports

1. **Core clock** — and this is not a formality. The naive `rdcycle`-vs-`mtime` measurement is
   **wrong on this silicon**: `mtime` is derived from the core clock (the ratio is exactly
   `SYS_CLK_FREQ/MTIME_FREQ` = 1000 whatever the PLL is set to), so it reports 49.99 MHz on a chip
   programmed for 750 MHz. It is wrong by 15×, in the direction that changes the conclusion: a
   3207-cycle RTT is 64 µs at 50 MHz and needs ~33 lines in flight to reach 33 MB/s, but 4.3 µs at
   750 MHz and needs ~2. So the probe samples cycles-per-tick before *and* after the PLL is
   programmed and reads the PLL/clock-mux registers back, which separates the three cases
   (`mtime` independent / `mtime` tracks the core / PLL genuinely not engaged) that otherwise produce
   identical timing evidence. See `../common/clock_probe.h`.
2. **Window check** — see Safety below.
3. **Latency curve** — random dependent-load pointer chase over 4 KB … 64 MB, one load per line in a
   single full-length cycle (Sattolo), so nothing prefetches and no two loads overlap. The knees name
   the cache hierarchy; the flat tail is the RTT. There are **no page tables** here — bare M-mode,
   nothing in the runtime ever writes `satp` — so no knee can be a TLB or page-walk effect and huge
   pages are not a lever that exists. The 256 KB → 1 MB cliff is the last-level cache boundary.
4. **Memory-level parallelism** — N *independent* pointer chases (1/2/4/8), each in its own 4 MB
   region, walked in lockstep. Each chain's next address depends only on its own previous load, so
   the memory system is offered exactly N independent misses and nothing else — no prefetcher, no row
   buffer, no loop scheduling mixed in. Flat cycles-per-round as N rises means the core overlaps N
   misses; linear scaling means one at a time. Reported as an **overlap** figure, a ratio of cycle
   counts, so it is valid regardless of how the clock question resolves.
5. **Streaming bandwidth** — read (scalar 64-bit and RVV at LMUL 1/2/4/8 = 32/64/128/256 B per
   request at VLEN=256), write, copy; then N interleaved streams in **two forms**:
   - *dependent* — each load consumed by the very next instruction. On an in-order core the pipeline
     stalls at each use, so the misses **cannot** overlap however many MSHRs the D$ has. This alone
     produces the "parallelism makes it worse" shape (1→8 streams getting monotonically slower) with
     no hardware defect required.
   - *independent* — all N loads of a round issued before any is used, each into its own accumulator.

   A gap between the two is the **loop**, not the hardware. Then a stride sweep (8…4096 B, charged
   per access) for the line size and any prefetching.
6. **Little's Law verdict** — lines in flight the streaming read actually sustains, the overlap the
   independent chases achieved, and the lines required to hit `MEMBW_LINK_TARGET_MBPS` (default 33)
   over the measured RTT, tabulated at **both** candidate clocks so the answer does not silently
   depend on the unresolved one.

Everything runs `MEMBW_PASSES` (default 2) times so run-to-run variance is visible. The final
`SUMMARY` line is one line to paste into a comparison, followed by the total sweep time in seconds at
the resolved clock — **check that against a stopwatch**, it is the only time reference on this chip
that is not derived from the clock being measured.

## Safety

A load or store to an address this SoC does not back **hangs the core silently** — no trap, no
print. Before writing 128 MB, `membw` writes and reads back the first and last word of the window,
announcing each access first, and then re-reads the first word to detect **aliasing** (which is how
a too-large window fails on a board with less DRAM than assumed, and would otherwise look like a
bandwidth anomaly rather than a configuration error).

## Configuration

All knobs are `#ifndef`-guarded in `include/membw_config.h` and wired to CMake cache variables:

| CMake variable | Default | Meaning |
|---|---|---|
| `MEMBW_BUF_ADDR` | `0x84000000` | Base of the test window (64 MB into DRAM, past the image) |
| `MEMBW_BUF_BYTES` | 128 MB | Window size; split into a source half and a destination half |
| `MEMBW_MAX_SPAN` | 64 MB | Largest latency-curve span (must be ≤ window/2) |
| `MEMBW_STREAM_SPAN` | 32 MB | Span for the streaming tests (≫ any cache here) |
| `MEMBW_MLP_SUB_SPAN` | 4 MB | Bytes per independent chase chain (8 are built; each must be ≫ LLC) |
| `MEMBW_LINK_TARGET_MBPS` | 33 | Bandwidth the link change should deliver; sets the Little's Law bar |
| `MEMBW_PASSES` | 2 | How many times to run the whole sweep |
| `MEMBW_TARGET_FREQ_HZ` | 750 MHz | PLL target; the sweep resolves the real clock anyway |

Defaults stay inside the 256 MB both chips' stock linker scripts declare. On a board with more DRAM,
raise `MEMBW_BUF_BYTES` — the aliasing check will report it if the extra memory is not really there.

## Shared code, and the host test

The measurement primitives live in `c2c-demos/common/mem_probe.h` so that `bearly-smollm` can print
the same numbers, in the same units, at boot (`SMOLLM_MEM_PROBE=1`, on by default) right next to the
workload they explain. Measuring both places with one implementation is the point: otherwise the two
sets of numbers are not comparable.

The header is chip-free, so the parts that can be *wrong* rather than merely slow are unit-tested on
the host in a second instead of a flash-and-squint cycle:

```bash
cd c2c-demos/membw/test
gcc -O2 -Wall -I../../common -o host_mem_probe_test host_mem_probe_test.c && ./host_mem_probe_test
```

It proves the chase really is a single cycle through every node (a Fisher-Yates/Sattolo mix-up would
break it into short cycles that sit in cache, reporting a DRAM latency several times too low and
entirely plausible-looking) and that the integer formatting is exact.
