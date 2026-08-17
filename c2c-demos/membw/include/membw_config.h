#ifndef MEMBW_CONFIG_H
#define MEMBW_CONFIG_H

/* ------------------------------------------------------------------------------------------------
 * membw — standalone memory latency / bandwidth sweep. Tunables.
 *
 * Every knob is `#ifndef`-guarded so CMake (or a -D on the command line) can override it without
 * editing this file.
 * ---------------------------------------------------------------------------------------------- */

/* PLL target handed to init_test(). The sweep does NOT trust this: it measures the real core clock
 * against the CLINT mtime reference and reports both, because a bitstream that retimes the memory
 * bus may have moved the core clock too — in which case cyc/byte can improve while wall-clock
 * throughput gets worse. 750 MHz is what dsp-citrinet and boraiq run at; 1 GHz did not lock. */
#ifndef MEMBW_TARGET_FREQ_HZ
#define MEMBW_TARGET_FREQ_HZ 750000000ULL
#endif

/* Test window: a fixed absolute range of DRAM, deliberately NOT a linker section.
 *
 * A .bss array or a heap allocation of this size would be a PT_LOAD segment with memsz >> filesz,
 * which uart_tsi zero-fills over the serial link on every flash — tens of megabytes of transfer for
 * memory whose contents this program overwrites immediately (the same trap bearly25-llm.ld exists to
 * avoid; see CLAUDE.md). An absolute window keeps the ELF at a few KB and loads instantly.
 *
 * Defaults sit 64 MB into DRAM (well past the image) and run to 192 MB, inside the 256 MB both
 * chips' stock linker scripts declare. On a board with more DRAM, raise MEMBW_BUF_BYTES — the boot
 * check below writes and reads back the first and last word before anything else touches the range,
 * so an unbacked window is reported instead of silently producing nonsense. */
#ifndef MEMBW_BUF_ADDR
#define MEMBW_BUF_ADDR 0x84000000UL
#endif
#ifndef MEMBW_BUF_BYTES
#define MEMBW_BUF_BYTES (128u << 20)   /* 128 MB: src half + dst half for the copy test */
#endif

/* Largest span in the latency curve and the streaming tests. Must be <= MEMBW_BUF_BYTES / 2 (the
 * buffer is split into two halves so copy has a distinct source and destination). */
#ifndef MEMBW_MAX_SPAN
#define MEMBW_MAX_SPAN (64u << 20)
#endif

/* Dependent loads per latency point. 100k at ~200 cyc each is ~20 Mcyc = 27 ms — long enough to
 * swamp timer overhead, short enough that the eight-point curve is instant. */
#ifndef MEMBW_LAT_STEPS
#define MEMBW_LAT_STEPS 100000u
#endif

/* Span for the streaming bandwidth tests. 32 MB is >> any cache here, so nothing is resident. */
#ifndef MEMBW_STREAM_SPAN
#define MEMBW_STREAM_SPAN (32u << 20)
#endif

/* Bytes per INDEPENDENT pointer chase in the memory-level-parallelism sweep. Eight chains are built,
 * so 8 * this must fit in MEMBW_MAX_SPAN. Each chain has to be far larger than the last-level cache
 * on its own, or the "misses" it offers are hits and the overlap figure is meaningless — 4 MB against
 * a 256 KB LLC is a 16x margin. */
#ifndef MEMBW_MLP_SUB_SPAN
#define MEMBW_MLP_SUB_SPAN (4u << 20)
#endif

/* The bandwidth a link change is supposed to deliver. Section 6 converts it into the number of cache
 * lines that must be in flight to reach it over the measured RTT — the bar any saturation experiment
 * has to clear, and the reason the MLP sweep is the gate. */
#ifndef MEMBW_LINK_TARGET_MBPS
#define MEMBW_LINK_TARGET_MBPS 33u
#endif

/* Run the whole sweep this many times. Two passes make run-to-run variance visible, which matters
 * when the point is to attribute a change to a bitstream rather than to noise. */
#ifndef MEMBW_PASSES
#define MEMBW_PASSES 2
#endif

/* CLINT mtime ticks sampled when measuring the core clock (50 kHz => 500 ticks = 10 ms). */
#ifndef MEMBW_CLOCK_TICKS
#define MEMBW_CLOCK_TICKS 500u
#endif

#ifndef MEMBW_LOG_ENABLE
#define MEMBW_LOG_ENABLE 1
#endif

#if MEMBW_LOG_ENABLE
#define MEMBW_LOG(...) printf(__VA_ARGS__)
#else
#define MEMBW_LOG(...) do { } while (0)
#endif

#endif /* MEMBW_CONFIG_H */
