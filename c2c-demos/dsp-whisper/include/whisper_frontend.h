#ifndef C2C_DSP_WHISPER_FRONTEND_H
#define C2C_DSP_WHISPER_FRONTEND_H

/* Whisper log-mel front-end (plan 005, P0). Matches whisper.audio.log_mel_spectrogram up to (but
 * not including) the global-max normalization, which is a global reduction handled by the caller.
 *
 * Dimensions (WF_N_FFT=400, WF_HOP=160, WF_N_BINS=201, WF_N_MELS=80) and the Hann window + mel
 * filterbank come from whisper_frontend_tables.h (auto-generated from whisper's own assets). */
#include "whisper_frontend_tables.h"

/* Build the direct-DFT cos/sin tables once. Must be called before whisper_logmel_frame(). */
void whisper_frontend_init(void);

/* Compute the RAW log-mel (log10(clamp(mel_power, 1e-10))) for one frame.
 *   frame:   WF_N_FFT contiguous audio samples (the caller windows by hop externally).
 *   out_mel: WF_N_MELS outputs.
 * This is per-frame independent; Whisper's final (log_spec+4)/4 after a global-max clamp is applied
 * downstream over the whole spectrogram. */
void whisper_logmel_frame(const float *frame, float *out_mel);

/* Compute the full normalized log-mel spectrogram from a raw 16 kHz waveform, matching
 * whisper.audio.log_mel_spectrogram: center reflect-pad by WF_N_FFT/2, Hann-400 / hop-160 STFT,
 * 201-bin power, mel filterbank, log10, global-max clamp (max-8), then (x+4)/4.
 *   audio:     n_samples float samples (~[-1,1]).
 *   out_mel:   caller buffer of at least WF_N_MELS*max_frames floats; written COEFF-MAJOR
 *              (out_mel[coeff*n_frames + frame]) — the layout the encoder's conv1d expects.
 * Returns n_frames (= n_samples/WF_HOP, capped at max_frames). Call whisper_frontend_init() first. */
int whisper_logmel_full(const float *audio, int n_samples, float *out_mel, int max_frames);

#endif /* C2C_DSP_WHISPER_FRONTEND_H */
