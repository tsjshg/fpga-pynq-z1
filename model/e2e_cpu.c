/*
 * 通しで動かすときの CPU 側の小物（段階11）。Mac でもボードでも同じこのファイルを焼く。
 *
 * ボードの numpy は NEON を使わず 1要素 20〜100 ns かかり、小物だけで 305 ms/トークンになった。
 * 素直な C（VFP のスカラ）にしても 55 ms。VFP は比較のたびに流れが止まり、
 * double の足し込みは前の結果待ちで 1要素 9 サイクルかかる。
 * → **NEON の組み込み関数で書く**。arm_neon.h は Mac（AArch64）にもあるので、同じソースが両方で焼ける。
 *
 * 【Mac とボードで結果をビット単位で揃えるための約束】
 *   ・-ffp-contract=off で焼く（vmulq + vaddq を FMA に融合させない。clang は既定で融合する）
 *   ・-ffast-math は使わない
 *   ・和は4レーンで取り、最後に (0+1)+(2+3) で畳む（順序を固定）。最大値は順序によらず厳密
 *   ・exp は自前（足し算と掛け算だけ）。丸めも整数の小細工で（rintf / expf は実装ごとに違いうる）
 *   ・割り算は NEON に無い（ARMv7）。silu の 1/(1+e) は推定値＋Newton 法2回で出す
 *     （スカラの IEEE 除算だと 4096 回で 1層 0.1 ms 以上かかった）
 *   ・ARMv7 の NEON は非正規化数を 0 に潰すが、ここでは非正規化数は出ない
 *     （出うるのは fexp(-87) 付近だけで、その値は 1+e にしか使わないので結果は同じ）
 * これで PL（整数）と CPU（浮動小数）の両方が一致すれば、logits が完全に一致するはず。
 */
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>

#define H   1536
#define FF  4096
#define NH  16
#define HD  96
#define TAU 8.0
#define SHM 24

typedef float32x4_t f4;
typedef int32x4_t   i4;
#define F(v) vdupq_n_f32(v)

/* 最近接偶数への丸め。1.5·2^23 を足すと仮数の下位ビットに整数が残る。|v| < 2^22 で正しい */
static inline i4 rnd4(f4 v)
{
    return vsubq_s32(vreinterpretq_s32_f32(vaddq_f32(v, F(12582912.0f))), vdupq_n_s32(0x4B400000));
}

/* 8 要素を [lo, 127] に収めて INT8 で書く */
static inline void st8(int8_t *q, i4 a, i4 b, int32_t lo)
{
    i4 hi = vdupq_n_s32(127), lw = vdupq_n_s32(lo);
    a = vmaxq_s32(vminq_s32(a, hi), lw);
    b = vmaxq_s32(vminq_s32(b, hi), lw);
    vst1_s8(q, vmovn_s16(vcombine_s16(vmovn_s32(a), vmovn_s32(b))));
}

static inline float hmax(f4 v)
{
    float32x2_t t = vpmax_f32(vget_low_f32(v), vget_high_f32(v));
    return vget_lane_f32(vpmax_f32(t, t), 0);
}

static inline f4 ld_i(const int32_t *p) { return vcvtq_f32_s32(vld1q_s32(p)); }

/* exp。x = k·ln2 + r（|r| ≤ ln2/2）に分け、e^r は 6 次の多項式、2^k は指数部に直接書く。誤差 4 ulp 以内 */
static inline f4 fexp4(f4 x)
{
    x = vmaxq_f32(vminq_f32(x, F(88.0f)), F(-87.0f));
    i4 k = rnd4(vmulq_f32(x, F(1.44269504f)));
    f4 kf = vcvtq_f32_s32(k);
    f4 r = vsubq_f32(vsubq_f32(x, vmulq_f32(kf, F(0.693145751953125f))),
                     vmulq_f32(kf, F(1.428606765330187e-06f)));
    f4 p = vaddq_f32(F(0.0083333338f), vmulq_f32(r, F(0.0013888889f)));
    p = vaddq_f32(F(0.041666668f), vmulq_f32(r, p));
    p = vaddq_f32(F(0.16666667f),  vmulq_f32(r, p));
    p = vaddq_f32(F(0.5f),         vmulq_f32(r, p));
    p = vaddq_f32(F(1.0f),         vmulq_f32(r, p));
    p = vaddq_f32(F(1.0f),         vmulq_f32(r, p));
    return vmulq_f32(p, vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(k, vdupq_n_s32(127)), 23)));
}

static float quant(const float *h, int n, int8_t *q)
{
    f4 mv = F(0.0f);
    for (int i = 0; i < n; i += 4) mv = vmaxq_f32(mv, vabsq_f32(vld1q_f32(h + i)));
    float m = hmax(mv);
    if (m < 1e-5f) m = 1e-5f;
    float s = 127.0f / m;
    f4 sv = F(s);
    for (int i = 0; i < n; i += 8)
        st8(q + i, rnd4(vmulq_f32(vld1q_f32(h + i), sv)),
                   rnd4(vmulq_f32(vld1q_f32(h + i + 4), sv)), -128);
    return s;
}

/* h = x / sqrt(mean(x²) + eps) · w を INT8 に。戻り値は s（= 127 / max|h|） */
float rmsq(const float *x, const float *w, int n, float eps, float *h, int8_t *q)
{
    f4 a0 = F(0.0f), a1 = F(0.0f);                  /* 2本に分けて足し込みの待ちを隠す */
    for (int i = 0; i < n; i += 8) {
        f4 x0 = vld1q_f32(x + i), x1 = vld1q_f32(x + i + 4);
        a0 = vaddq_f32(a0, vmulq_f32(x0, x0));
        a1 = vaddq_f32(a1, vmulq_f32(x1, x1));
    }
    f4 a = vaddq_f32(a0, a1);
    float ss = (vgetq_lane_f32(a, 0) + vgetq_lane_f32(a, 1))
             + (vgetq_lane_f32(a, 2) + vgetq_lane_f32(a, 3));
    f4 inv = F(1.0f / sqrtf(ss / (float)n + eps));
    for (int i = 0; i < n; i += 4)
        vst1q_f32(h + i, vmulq_f32(vmulq_f32(vld1q_f32(x + i), inv), vld1q_f32(w + i)));
    return quant(h, n, q);
}

/* qkv の行列積の後: 逆量子化 → RoPE → K,V を固定尺度で INT8、q はヘッドごとに INT8 と倍率 m */
void qkv_post(const int32_t *acc, float gq, float gk, float gv, float s,
              const float *cs, const float *sn, const float *ik, const float *iv,
              const double *sk, int8_t *kq, int8_t *vq, int8_t *qi, int64_t *mm)
{
    f4 fq = F(gq / s), fk = F(gk / s), fv = F(gv / s);
    const double rs = 1.0 / sqrt((double)HD);
    for (int h = 0; h < NH; h++) {
        float q[HD] __attribute__((aligned(16))), k[HD] __attribute__((aligned(16)));
        const int32_t *aq = acc + h*HD, *ak = acc + H + h*HD, *av = acc + 2*H + h*HD;
        f4 qmv = F(0.0f);
        for (int j = 0; j < HD/2; j += 4) {
            f4 c0 = vld1q_f32(cs + j), c1 = vld1q_f32(cs + j + HD/2);
            f4 s0 = vld1q_f32(sn + j), s1 = vld1q_f32(sn + j + HD/2);
            f4 q0 = vmulq_f32(ld_i(aq + j), fq), q1 = vmulq_f32(ld_i(aq + j + HD/2), fq);
            f4 k0 = vmulq_f32(ld_i(ak + j), fk), k1 = vmulq_f32(ld_i(ak + j + HD/2), fk);
            f4 qa = vaddq_f32(vmulq_f32(q0, c0), vmulq_f32(vnegq_f32(q1), s0));
            f4 qb = vaddq_f32(vmulq_f32(q1, c1), vmulq_f32(q0, s1));
            vst1q_f32(q + j, qa); vst1q_f32(q + j + HD/2, qb);
            vst1q_f32(k + j,        vaddq_f32(vmulq_f32(k0, c0), vmulq_f32(vnegq_f32(k1), s0)));
            vst1q_f32(k + j + HD/2, vaddq_f32(vmulq_f32(k1, c1), vmulq_f32(k0, s1)));
            qmv = vmaxq_f32(qmv, vmaxq_f32(vabsq_f32(qa), vabsq_f32(qb)));
        }
        f4 ikh = F(ik[h]), ivh = F(iv[h]);
        for (int j = 0; j < HD; j += 8) {
            st8(kq + h*HD + j, rnd4(vmulq_f32(vld1q_f32(k + j), ikh)),
                               rnd4(vmulq_f32(vld1q_f32(k + j + 4), ikh)), -127);
            st8(vq + h*HD + j, rnd4(vmulq_f32(vmulq_f32(ld_i(av + j), fv), ivh)),
                               rnd4(vmulq_f32(vmulq_f32(ld_i(av + j + 4), fv), ivh)), -127);
        }
        float qm = hmax(qmv);
        if (qm < 1e-9f) qm = 1e-9f;
        f4 inv = F(127.0f / qm);
        for (int j = 0; j < HD; j += 8)
            st8(qi + h*HD + j, rnd4(vmulq_f32(vld1q_f32(q + j), inv)),
                               rnd4(vmulq_f32(vld1q_f32(q + j + 4), inv)), -127);
        double sq = (double)qm / 127.0;
        double m = rint(TAU * sq * sk[h] * rs * (double)(1 << SHM));
        if (m < 1.0) m = 1.0;
        if (m > 262143.0) m = 262143.0;
        mm[h] = (int64_t)m;
    }
}

/* attention の後: 分子 ÷ 分母 × sv → inner_attn_ln → INT8。o は (16, 97) の int32（97 番目が分母） */
float attn_post(const int32_t *o, const double *sv, const float *w, float eps,
                float *h, int8_t *q)
{
    float t[H] __attribute__((aligned(16)));
    for (int hh = 0; hh < NH; hh++) {
        f4 f = F((float)(sv[hh] / (double)o[hh*97 + HD]));
        for (int j = 0; j < HD; j += 4)
            vst1q_f32(t + hh*HD + j, vmulq_f32(ld_i(o + hh*97 + j), f));
    }
    return rmsq(t, w, H, eps, h, q);
}

/* 残差: x += acc · (g/s)。続けて次の正規化と量子化 */
float resid_rmsq(float *x, const int32_t *acc, float g, float s,
                 const float *w, float eps, float *h, int8_t *q)
{
    f4 f = F(g / s);
    for (int i = 0; i < H; i += 4)
        vst1q_f32(x + i, vaddq_f32(vld1q_f32(x + i), vmulq_f32(ld_i(acc + i), f)));
    return rmsq(x, w, H, eps, h, q);
}

/* 1/d。推定値（8bit）から Newton 法を2回。vrecpsq は AArch64 だと融合演算になり
 * ARMv7 と結果が変わるので使わず、掛け算と引き算を分けて書く */
static inline f4 recip4(f4 d)
{
    f4 r = vrecpeq_f32(d);
    r = vmulq_f32(r, vsubq_f32(F(2.0f), vmulq_f32(d, r)));
    r = vmulq_f32(r, vsubq_f32(F(2.0f), vmulq_f32(d, r)));
    return r;
}

/* gate/up の後: silu(gate) · up → ffn_layernorm → INT8 */
float mlp_post(const int32_t *acc, float gg, float gu, float s,
               const float *w, float eps, float *h, int8_t *q)
{
    float t[FF] __attribute__((aligned(16)));
    f4 fg = F(gg / s), fu = F(gu / s);
    /* exp の多項式は前の結果待ちの一本道なので、2本ずつ並べて待ちを埋める（A9 は順序通り実行） */
    for (int i = 0; i < FF; i += 8) {
        f4 a0 = vmulq_f32(ld_i(acc + i), fg),     a1 = vmulq_f32(ld_i(acc + i + 4), fg);
        f4 b0 = vmulq_f32(ld_i(acc + FF + i), fu), b1 = vmulq_f32(ld_i(acc + FF + i + 4), fu);
        f4 e0 = fexp4(vnegq_f32(a0)),              e1 = fexp4(vnegq_f32(a1));
        f4 r0 = recip4(vaddq_f32(F(1.0f), e0)),    r1 = recip4(vaddq_f32(F(1.0f), e1));
        vst1q_f32(t + i,     vmulq_f32(vmulq_f32(a0, r0), b0));
        vst1q_f32(t + i + 4, vmulq_f32(vmulq_f32(a1, r1), b1));
    }
    return rmsq(t, w, FF, eps, h, q);
}

/* lm_head の後: logits と、その最大の位置（numpy の argmax は 32002 語で 0.4 ms かかる） */
/* lg が NULL なら logits は作らず argmax だけ（生成には最大の位置しか要らない） */
int head_post(const int32_t *acc, float g, float s, int n, float *lg)
{
    int i = 0, best = 0;
    if (lg) {
        f4 f = F(g / s);
        for (; i + 4 <= n; i += 4) vst1q_f32(lg + i, vmulq_f32(ld_i(acc + i), f));
        for (; i < n; i++) lg[i] = (float)acc[i] * (g / s);
    }
    /* argmax。g/s > 0 なので整数で比べてよい。レーンごとに最大と最初の位置を持ち、最後に畳む */
    i4 mv = vdupq_n_s32(INT32_MIN);
    uint32x4_t iv = vdupq_n_u32(0), cur = {0, 1, 2, 3};
    for (i = 0; i + 4 <= n; i += 4) {
        i4 v = vld1q_s32(acc + i);
        uint32x4_t gt = vcgtq_s32(v, mv);
        mv = vmaxq_s32(mv, v);
        iv = vbslq_u32(gt, cur, iv);
        cur = vaddq_u32(cur, vdupq_n_u32(4));
    }
    int32_t m[4]; uint32_t ix[4];
    vst1q_s32(m, mv); vst1q_u32(ix, iv);
    int32_t mx = m[0]; best = (int)ix[0];
    for (int j = 1; j < 4; j++)
        if (m[j] > mx || (m[j] == mx && (int)ix[j] < best)) { mx = m[j]; best = (int)ix[j]; }
    for (; i < n; i++) if (acc[i] > mx) { mx = acc[i]; best = i; }
    return best;
}
