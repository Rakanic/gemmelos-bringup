#ifndef MIC_ARRAY_H
#define MIC_ARRAY_H

/* ------------------------------------------------------------------------------------------------
 * 3-element endfire microphone array for the wake gate — DSP 25.
 *
 * WHAT IT IS FOR: the gate should wake for someone in FRONT of the array and ignore the same words
 * spoken from the side. The elements are colinear, 2.15 cm apart, and the beam looks out past
 * element 0 — so point element 0 at the user.
 *
 * WHAT IT CAN AND CANNOT DELIVER. Measured on silicon (dsp-mic-bench, 2026-08-20): the endfire sum
 * is +2.78 dB on axis and +0.85 dB at 57 deg off — about **2 dB** of front/side discrimination, which
 * is what a 4.3 cm aperture allows and no implementation effort will improve. That is enough to
 * prefer a direction; it is NOT enough to reject a loud or close side talker on level alone. Hence
 * two gates in wake_config.h, and the DIRECTION one is the load-bearing one:
 *   - `tau` (this header) is a LAG RATIO, so it is immune to source loudness, to range, and to the
 *     gain calibration being stale. It is the reliable discriminator.
 *   - `gain_db` is an energy ratio and moves with all three of those. Useful, and what an "energy
 *     threshold" means here, but tune it from the console rather than trusting a default.
 *
 * WHY ALIGNMENT IS RE-MEASURED EVERY CAPTURE. The two I2S RX FIFOs of a channel slip an INTEGER
 * sample against each other whenever the read loop stops long enough for them to overflow (see the
 * I2S entry in /CLAUDE.md), and the wake gate stops for ~240 ms on every classification. So a
 * boot-time alignment would be stale by the second onset. Re-measuring per capture is cheap (one
 * correlation over a slice of the window, against audio that by definition has signal in it) and it
 * self-heals whatever the rest of the demo did to the FIFOs in between — including Citrinet's own
 * single-mic capture, which drains only one of the three.
 *
 * WHAT MUST SURVIVE THE ALIGNMENT: the ACOUSTIC lag. Removing it would steer the beam onto whoever
 * is talking, and a beamformer whose look direction follows the source is not a beamformer. Only the
 * instrument part (channel offset + slip, both integers) is taken out; the acoustic part is left in
 * place for the fixed endfire delays to act on. That distinction is the whole design.
 * ---------------------------------------------------------------------------------------------- */

#include <stdint.h>

/* Elements, front first. Element 0 is the front mic and the reference for every lag and level.
 * Three, not four: the fourth slot is dead on this board (see the README). */
#define MIC_ARRAY_ELEMS 3

/* What one capture's alignment worked out, for logging and for the gates. */
typedef struct {
  float tau;                        /* per-gap acoustic lag, samples. +1 = on axis front,
                                     * 0 = broadside, -1 = on axis behind. This is cos(angle). */
  float angle_deg;                  /* acos(tau/bound): 0 = front, 90 = broadside, 180 = rear */
  float gain_db;                    /* beamformed RMS / element-0 RMS, dB — the array's own gain */
  float lag[MIC_ARRAY_ELEMS];       /* measured delay vs element 0, samples (instrument + acoustic) */
  float gamma[MIC_ARRAY_ELEMS];     /* correlation quality of that lag, [0,1] */
  float lvl_db[MIC_ARRAY_ELEMS];    /* level vs element 0 AFTER gain correction */
  int   shift[MIC_ARRAY_ELEMS];     /* integer instrument correction actually applied */
  uint32_t used;                    /* elements that made it into the sum (bitmask) */
  uint32_t nused;                   /* popcount(used) */
  int   level_decided;              /* 1 = the level cue broke the front/back tie, 0 = deadband */
} mic_array_info_t;

/* Configure the second I2S channel and probe all three slots. Call from wake_gate_init(), i.e.
 * inside app_init: this touches a local peripheral only, never the C2C link, so it is safe there.
 * Returns 0 on success, -1 if fewer than two slots carry signal (an array needs at least a pair). */
int mic_array_init(void);

/* MONITOR PATH. Read one 64-bit RX block (two samples) from EVERY element and push it into the
 * pre-roll ring. Returns element 0's two samples so the caller can run its energy test on the
 * reference mic unchanged.
 *
 * Every element must be drained on every iteration or its FIFO overflows and the alignment for the
 * next capture is measured against a stream that jumped — which is why this reads all three even
 * though only element 0's samples come back. */
void mic_array_monitor_pair(float *a, float *b);

/* CAPTURE PATH. Emit the pre-roll into the per-element windows, then fill to `n` samples with fresh
 * audio. Blocking reads only, so the window is contiguous. Returns samples captured (== n). */
uint32_t mic_array_capture(uint32_t n);

/* Align the captured windows and endfire-sum them into `dst` (n samples). Removes each element's DC
 * first (per element — they sit on different offsets), takes out the integer instrument offset, and
 * leaves the acoustic lag for the fixed steering delays. Fills `info`. */
void mic_array_beamform(float *dst, uint32_t n, mic_array_info_t *info);

#endif /* MIC_ARRAY_H */
