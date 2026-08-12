/*
 * moonshine_rvv.S — hand-written RVV inner kernel for the Moonshine matvec (plan 006).
 *
 * float ms_matrow_q8(const int8_t *q, const float *x, const float *s, int in, int gs)
 *   returns sum_i ( q[i] * s[i/gs] ) * x[i]   for i in [0, in)
 *
 * Written in asm because GCC 13.2's vsetvl-insertion pass miscompiles the intrinsic form: it emits
 * `vsext.vf4` under an e8,m1 vtype (illegal — source would be 2-bit), instead of e32,m4. Here we
 * hold ONE vtype (e32,m4) across the whole inner loop; `vle8.v` self-encodes EEW=8 (EMUL=1) and
 * `vsext.vf4` correctly widens int8->int32 under e32,m4. Each group's scale is folded into the
 * dequantized weight, and we FMA into a persistent f32 accumulator (v28), one reduction per row.
 *
 * VLEN-agnostic: vl comes from vsetvli each iteration (32 f32 lanes at VLEN=256). `in` and `gs` are
 * multiples of the lane count here (in in {384,1536}, gs=64), and the vl-per-iter handles any tail.
 *
 * Registers (all caller-saved on RISC-V): a0=q a1=x a2=s a3=in a4=gs; t0-t4 scratch; fa5 scale;
 * v8 int8 load, v12-15 x, v16 reduction scalar, v24-27 widened weight, v28-31 accumulator.
 */
  .text
  .balign 4
  .globl ms_matrow_q8
  .type  ms_matrow_q8, @function
ms_matrow_q8:
  vsetvli t0, zero, e32, m4, ta, ma   /* vtype = e32,m4 for the whole routine */
  vmv.v.i v28, 0                      /* acc = 0 across VLMAX lanes */
  li      t1, 0                       /* i = 0 (elements consumed in this row) */
.Lgrp:
  bge     t1, a3, .Lred               /* while (i < in) */
  flw     fa5, 0(a2)                  /* scale = *s */
  addi    a2, a2, 4                   /* s++ */
  mv      t2, a4                      /* remaining = gs */
.Linner:
  vsetvli t3, t2, e32, m4, ta, ma     /* vl = min(remaining, VLMAX); vtype stays e32,m4 */
  vle8.v  v8, (a0)                    /* load vl int8 weights (EMUL=1) */
  add     a0, a0, t3                  /* q += vl (1 byte each) */
  vsext.vf4   v24, v8                 /* sign-extend int8 -> int32 (e32,m4) */
  vfcvt.f.x.v v24, v24                /* int32 -> f32 */
  vfmul.vf    v24, v24, fa5           /* * scale (fold group scale into weight) */
  vle32.v v12, (a1)                   /* load vl activations */
  slli    t4, t3, 2
  add     a1, a1, t4                  /* x += vl (4 bytes each) */
  vfmacc.vv v28, v24, v12             /* acc += weight * x */
  sub     t2, t2, t3                  /* remaining -= vl */
  add     t1, t1, t3                  /* i += vl */
  bnez    t2, .Linner
  j       .Lgrp
.Lred:
  vsetvli t0, zero, e32, m4, ta, ma
  vmv.s.x v16, zero
  vfredusum.vs v28, v28, v16          /* tree-reduce acc lanes -> element 0 */
  vfmv.f.s fa0, v28                   /* return the row sum */
  ret
  .size ms_matrow_q8, .-ms_matrow_q8

/*
 * float ms_matrow_i8(const int8_t *q, const int8_t *xq, const float *s, int in, int gs)
 *   returns sum_g s[g] * sum_{i in g} q[i]*xq[i]    (activation scale sx + bias applied by caller)
 *
 * int8 x int8 -> int32 inner product. Both operands sign-extended int8->int32, integer FMA into a
 * per-group int32 accumulator (v16, no overflow: 64 * 127*127 << 2^31), then per group convert to
 * f32, scale by the group's weight scale, and add to a persistent f32 accumulator (v28). One float
 * reduction per row. Single e32,m4 vtype throughout (avoids the GCC intrinsic widening bug).
 *
 * a0=q a1=xq a2=s a3=in a4=gs; t0-t4 scratch; fa5 scale; v0 reduce-scalar, v4-7 xq widened,
 * v8/v9 int8 loads, v16-19 int32 acc, v20-23 f32 temp, v24-27 q widened, v28-31 f32 acc.
 */
  .balign 4
  .globl ms_matrow_i8
  .type  ms_matrow_i8, @function
ms_matrow_i8:
  fmv.w.x fa0, zero                   /* row float sum = 0 */
  li      t1, 0                       /* i */
.Li8g:
  bge     t1, a3, .Li8done
  flw     fa5, 0(a2)                  /* weight scale s[g] */
  addi    a2, a2, 4
  vsetvli t0, zero, e32, m1, ta, ma
  vmv.s.x v16, zero                   /* int32 group-dot accumulator (scalar in v16[0]) */
  mv      t2, a4                      /* remaining in group */
.Li8i:
  vsetvli t3, t2, e8, m2, ta, ma      /* up to 64 int8 lanes per op (VLEN=256) */
  vle8.v  v8, (a0)                    /* int8 weights  */
  add     a0, a0, t3
  vle8.v  v10, (a1)                   /* int8 activations */
  add     a1, a1, t3
  vwmul.vv v12, v8, v10               /* int8 * int8 -> int16 (no early int32 widen) */
  vsetvli zero, t3, e16, m4, ta, ma   /* source width for the widening reduce */
  vwredsum.vs v16, v12, v16           /* sum(int16) accumulated into the int32 scalar v16[0] */
  sub     t2, t2, t3
  add     t1, t1, t3
  bnez    t2, .Li8i
  vsetvli t0, zero, e32, m1, ta, ma
  vmv.x.s t4, v16                     /* int32 group dot */
  fcvt.s.w fa4, t4
  fmul.s  fa4, fa4, fa5               /* * weight scale */
  fadd.s  fa0, fa0, fa4               /* row sum += */
  j       .Li8g
.Li8done:
  ret
  .size ms_matrow_i8, .-ms_matrow_i8
