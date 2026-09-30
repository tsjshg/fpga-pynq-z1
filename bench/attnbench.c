/* SmolLM2-135M の「行列積以外」を実寸法で測る（PYNQ-Z1 の CPU 上）
 *
 *   使い方: ./attnbench [文脈長T] [繰り返し]     既定 512, 20
 *
 * 目的: PL が行列積を 14.68 ms/トークンでこなせることは実測済み。
 *       残り（attention・softmax・正規化など）を CPU に任せた場合の
 *       時間を測り、「行列積は PL、残りは CPU」で成立するかを判定する。
 *
 * 測るもの（すべて1トークンぶん・全30層合計）:
 *   A. KV キャッシュの逐次リードだけ  … 帯域の下限。ここは絶対に削れない
 *   B. attention 一式                 … QK^T + softmax + AV
 *   C. softmax だけ                   … exp の回数が効く
 *   D. 小物一式                       … RMSNorm ×2 / RoPE / SiLU / 残差
 *
 * 行列積（Q/K/V/O/gate/up/down/lm_head）は PL が担当するので含めない。
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>

/* SmolLM2-135M の構成（config.json で確認済み） */
#define HID  576
#define NL   30
#define NH   9            /* query ヘッド */
#define NKV  3            /* key/value ヘッド（GQA） */
#define HD   64
#define FFN  1536
#define KVD  (NKV*HD)     /* 192 */

static int T;             /* 文脈長 */
static int nthr;

/* KV キャッシュ: [層][kvヘッド][位置][次元] の INT8 */
static int8_t *Kc, *Vc;
static size_t kv_layer_sz;          /* 1層1方向ぶんのバイト数 = KVD*T */

static float  *qbuf;                /* [層][NH*HD] */
static float  *xbuf;                /* [層][HID]   小物用 */
static float  *fbuf;                /* [層][FFN]   小物用 */
static double  sink;                /* 最適化除け */

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

/* ---- A. KV キャッシュの逐次リードだけ ---- */
static void *kv_read(void *a){
    int id=(int)(long)a; uint64_t s=0;
    for(int l=id; l<NL; l+=nthr){
        const uint64_t *p=(const uint64_t*)(Kc + (size_t)l*kv_layer_sz);
        const uint64_t *q=(const uint64_t*)(Vc + (size_t)l*kv_layer_sz);
        for(size_t i=0;i<kv_layer_sz/8;i++) s += p[i]+q[i];
    }
    sink += (double)(s & 1); return 0;
}

/* ---- B. attention 一式 ---- */
static void *attn(void *a){
    int id=(int)(long)a;
    float *sc = malloc(sizeof(float)*T);
    float acc[HD];
    for(int l=id; l<NL; l+=nthr){
        const int8_t *K = Kc + (size_t)l*kv_layer_sz;
        const int8_t *V = Vc + (size_t)l*kv_layer_sz;
        for(int h=0; h<NH; h++){
            const float *q = qbuf + (size_t)l*NH*HD + (size_t)h*HD;
            int kh = h / (NH/NKV);                       /* GQA: 3ヘッドで1つを共有 */
            const int8_t *Kh = K + (size_t)kh*T*HD;
            const int8_t *Vh = V + (size_t)kh*T*HD;
            /* QK^T */
            float mx = -1e30f;
            for(int t=0;t<T;t++){
                const int8_t *k = Kh + (size_t)t*HD;
                float s=0; for(int d=0;d<HD;d++) s += q[d]*(float)k[d];
                s *= 0.125f;                              /* 1/sqrt(64) */
                sc[t]=s; if(s>mx) mx=s;
            }
            /* softmax */
            float sum=0;
            for(int t=0;t<T;t++){ sc[t]=expf(sc[t]-mx); sum+=sc[t]; }
            float inv=1.0f/sum;
            /* AV */
            for(int d=0;d<HD;d++) acc[d]=0;
            for(int t=0;t<T;t++){
                float p=sc[t]*inv;
                const int8_t *v = Vh + (size_t)t*HD;
                for(int d=0;d<HD;d++) acc[d] += p*(float)v[d];
            }
            sink += acc[0];
        }
    }
    free(sc); return 0;
}

/* ---- B2. attention（INT8 のまま計算する現実的な実装） ----
 * B は要素ごとに int8→float 変換していて、素朴すぎる。
 * 実用実装は q も確率も量子化して整数のまま積和する。こちらが CPU の実力。 */
static void *attn8(void *a){
    int id=(int)(long)a;
    float *sc = malloc(sizeof(float)*T);
    uint8_t *pq = malloc(T);
    int8_t qq[HD]; int32_t acc[HD];
    for(int d=0;d<HD;d++) qq[d]=(int8_t)(d%127-63);
    for(int l=id; l<NL; l+=nthr){
        const int8_t *K = Kc + (size_t)l*kv_layer_sz;
        const int8_t *V = Vc + (size_t)l*kv_layer_sz;
        for(int h=0; h<NH; h++){
            int kh = h / (NH/NKV);
            const int8_t *Kh = K + (size_t)kh*T*HD;
            const int8_t *Vh = V + (size_t)kh*T*HD;
            float mx=-1e30f;
            for(int t=0;t<T;t++){
                const int8_t *k = Kh + (size_t)t*HD;
                int32_t s=0;
                for(int d=0;d<HD;d++) s += (int32_t)qq[d]*(int32_t)k[d];
                float f=(float)s*0.001f; sc[t]=f; if(f>mx) mx=f;
            }
            float sum=0;
            for(int t=0;t<T;t++){ sc[t]=expf(sc[t]-mx); sum+=sc[t]; }
            float inv=255.0f/sum;
            for(int t=0;t<T;t++) pq[t]=(uint8_t)(sc[t]*inv);
            for(int d=0;d<HD;d++) acc[d]=0;
            for(int t=0;t<T;t++){
                int32_t p=pq[t];
                const int8_t *v = Vh + (size_t)t*HD;
                for(int d=0;d<HD;d++) acc[d] += p*(int32_t)v[d];
            }
            sink += (double)acc[0];
        }
    }
    free(sc); free(pq); return 0;
}

/* ---- C. softmax だけ（exp の回数だけを見る） ---- */
static int use_fast_exp;
static inline float fexp(float x){
    /* よくある高速近似: 2^x をビット演算で作る。llama.cpp 等が使う類のもの */
    x = 1.4426950408889634f*x;                 /* log2(e) */
    if(x < -126.0f) return 0.0f;
    float xi = floorf(x), xf = x - xi;
    float m = 1.0f + xf*(0.6960656421638072f + xf*(0.224494337302845f + xf*0.07944154167983575f));
    union { uint32_t u; float f; } u;
    u.u = (uint32_t)((int)xi + 127) << 23;
    return m * u.f;
}
static void *smax(void *a){
    int id=(int)(long)a;
    float *sc = malloc(sizeof(float)*T);
    for(int t=0;t<T;t++) sc[t] = (float)(t%17) * 0.1f - 0.8f;
    for(int l=id; l<NL; l+=nthr){
        for(int h=0; h<NH; h++){
            float sum=0;
            if(use_fast_exp) for(int t=0;t<T;t++) sum += fexp(sc[t]);
            else             for(int t=0;t<T;t++) sum += expf(sc[t]);
            sink += sum;
        }
    }
    free(sc); return 0;
}

/* ---- D. 小物一式: RMSNorm ×2 / RoPE / SiLU / 残差 ---- */
static void *misc(void *a){
    int id=(int)(long)a;
    for(int l=id; l<NL; l+=nthr){
        float *x = xbuf + (size_t)l*HID;
        float *f = fbuf + (size_t)l*FFN;
        /* RMSNorm ×2 */
        for(int r=0;r<2;r++){
            float s=0; for(int i=0;i<HID;i++) s += x[i]*x[i];
            float inv = 1.0f/sqrtf(s/HID + 1e-5f);
            for(int i=0;i<HID;i++) x[i] *= inv;
        }
        /* RoPE: q(576) と k(192) の回転。cos/sin は事前計算される前提で乗算のみ */
        for(int i=0;i<(HID+KVD)/2;i++){
            float c=0.9998f, s=0.0175f;
            float a0=x[(2*i)%HID], a1=x[(2*i+1)%HID];
            x[(2*i)%HID]     = a0*c - a1*s;
            x[(2*i+1)%HID]   = a0*s + a1*c;
        }
        /* SiLU + gate*up の要素積 */
        for(int i=0;i<FFN;i++){
            float g=f[i];
            f[i] = g/(1.0f+expf(-g)) * (g*0.5f);
        }
        /* 残差2回 */
        for(int i=0;i<HID;i++) x[i] += 0.001f;
        for(int i=0;i<HID;i++) x[i] += 0.001f;
        sink += x[0] + f[0];
    }
    return 0;
}

static double run(void *(*fn)(void*), int rep){
    pthread_t th[64];
    /* 1回空回しして温める */
    for(long i=0;i<nthr;i++) pthread_create(&th[i],0,fn,(void*)i);
    for(int i=0;i<nthr;i++) pthread_join(th[i],0);
    double t0=now();
    for(int r=0;r<rep;r++){
        for(long i=0;i<nthr;i++) pthread_create(&th[i],0,fn,(void*)i);
        for(int i=0;i<nthr;i++) pthread_join(th[i],0);
    }
    return (now()-t0)/rep*1000.0;      /* ms / トークン */
}

int main(int argc,char**argv){
    T    = (argc>1)? atoi(argv[1]) : 512;
    int rep = (argc>2)? atoi(argv[2]) : 20;
    nthr = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if(nthr<1) nthr=1; if(nthr>64) nthr=64;

    kv_layer_sz = (size_t)KVD*T;
    size_t kvtot = kv_layer_sz*NL*2;

    printf("SmolLM2-135M / hidden %d / %d層 / Q %d ヘッド / KV %d ヘッド / head_dim %d / FFN %d\n",
           HID, NL, NH, NKV, HD, FFN);
    printf("文脈長 T = %d / スレッド %d / 繰り返し %d\n", T, nthr, rep);
    printf("KV キャッシュ = %.2f MB（INT8・%d バイト/位置）\n\n",
           kvtot/1048576.0, NL*2*KVD);

    Kc = malloc(kv_layer_sz*NL);  Vc = malloc(kv_layer_sz*NL);
    qbuf = malloc(sizeof(float)*NL*NH*HD);
    xbuf = malloc(sizeof(float)*NL*HID);
    fbuf = malloc(sizeof(float)*NL*FFN);
    if(!Kc||!Vc||!qbuf||!xbuf||!fbuf){ fprintf(stderr,"確保できません\n"); return 1; }
    srand(1);
    for(size_t i=0;i<kv_layer_sz*NL;i++){ Kc[i]=(int8_t)(rand()%255-127); Vc[i]=(int8_t)(rand()%255-127); }
    for(size_t i=0;i<(size_t)NL*NH*HD;i++) qbuf[i]=(float)(rand()%200-100)*0.01f;
    for(size_t i=0;i<(size_t)NL*HID;i++)   xbuf[i]=(float)(rand()%200-100)*0.01f;
    for(size_t i=0;i<(size_t)NL*FFN;i++)   fbuf[i]=(float)(rand()%200-100)*0.01f;

    double a = run(kv_read, rep);
    double b = run(attn,    rep);
    double b8= run(attn8,   rep);
    use_fast_exp=0; double c1 = run(smax, rep);
    use_fast_exp=1; double c2 = run(smax, rep);
    double d = run(misc,    rep);

    double gemv = 134.48e6/9.16e9*1000.0;     /* PL の実測 9.16 G重み/s から */
    double cpu  = b8 + d;                      /* 速いほうの実装で予算を組む */
    double macs = (double)NL*NH*T*HD*2;        /* QK^T と AV */

    printf("=== 1トークンあたり（全30層の合計・%d スレッド）===\n", nthr);
    printf("  A. KV キャッシュの逐次リードのみ : %7.2f ms   (%.2f GB/s)\n", a, kvtot/(a/1000.0)/1e9);
    printf("  B. attention  素朴（float変換） : %7.2f ms\n", b);
    printf("  B2. 同        INT8 のまま      : %7.2f ms   (%.0f M MAC → %.2f G MAC/s)\n",
           b8, macs/1e6, macs/(b8/1000.0)/1e9);
    printf("  C. softmax だけ  expf()         : %7.2f ms   (%d 回)\n", c1, NL*NH*T);
    printf("     同           高速近似        : %7.2f ms\n", c2);
    printf("  D. 小物（正規化/RoPE/SiLU/残差） : %7.2f ms\n", d);
    printf("\n=== 予算表 ===\n");
    printf("  行列積（PL・実測 9.16 G重み/s）  : %7.2f ms   134.5 M 重み\n", gemv);
    printf("  それ以外（CPU・B2 + D）         : %7.2f ms\n", cpu);
    printf("  合計                            : %7.2f ms  → %.1f tok/s\n", gemv+cpu, 1000.0/(gemv+cpu));
    printf("\n  CPU 側が占める割合              : %5.1f%%\n", 100.0*cpu/(gemv+cpu));
    printf("  行列積だけなら                  : %5.1f tok/s\n", 1000.0/gemv);
    printf("\n  参考: この attention を PL の INT8 演算器（実測 1.90 G MAC/s）に載せると\n");
    printf("        %.2f ms。KV の読み出しだけなら %.2f ms（1.91 GB/s）\n",
           macs/1.90e9*1000.0, kvtot/1.91e9*1000.0);
    if(cpu > gemv*0.25)
        printf("\n  → CPU 側が重い。ここも回路にしないと PL の速さが活きない\n");
    else
        printf("\n  → CPU 側は十分軽い。「行列積は PL、残りは CPU」で成立する\n");
    if(sink==1234.5) printf("");
    return 0;
}
