/* 「行列積以外」を任意の寸法で測る（PYNQ-Z1 の CPU 上）
 *
 *   使い方: ./attnbench2 T rep HID NL NH NKV HD FFN NRM_H NRM_F
 *     SmolLM2-135M      : 512 20  576 30  9  3 64 1536 2 0
 *     bitnet_b1_58-large: 512 20 1536 24 16 16 96 4096 3 1
 *
 * attnbench.c は SmolLM2 の寸法を #define で埋め込んでいた。
 * モデルを乗り換えたので実行時引数にした。**まず SmolLM2 の寸法で
 * 旧版の値を再現できることを確かめてから**、新しいモデルを測ること。
 *
 * 測るもの（すべて1トークンぶん・全層合計）:
 *   A. KV キャッシュの逐次リードだけ  … 帯域の下限。絶対に削れない
 *   B. attention（素朴な float 版）
 *   B2. attention（INT8 のまま。こちらが CPU の実力）
 *   C. softmax だけ
 *   D. 小物（RMSNorm ×NRM_H(hidden) + ×NRM_F(FFN) / RoPE / SiLU / 残差）
 *
 * 行列積は PL が担当するので含めない。
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

static int T, nthr;
static int HID, NL, NH, NKV, HD, FFN, NRM_H, NRM_F;
static int KVD;                      /* NKV*HD */
static float QSCALE;                 /* 1/sqrt(HD) */

static int8_t *Kc, *Vc;
static size_t kv_layer_sz;
static float  *qbuf, *xbuf, *fbuf;
static double  sink;

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

/* ---- A. KV の逐次リードだけ ---- */
static void *kv_read(void *a){
    int id=(int)(long)a; uint64_t s=0;
    for(int l=id; l<NL; l+=nthr){
        const uint64_t *p=(const uint64_t*)(Kc + (size_t)l*kv_layer_sz);
        const uint64_t *q=(const uint64_t*)(Vc + (size_t)l*kv_layer_sz);
        for(size_t i=0;i<kv_layer_sz/8;i++) s += p[i]+q[i];
    }
    sink += (double)(s & 1); return 0;
}

/* ---- B. attention（素朴な float 版）---- */
static void *attn(void *a){
    int id=(int)(long)a;
    float *sc = malloc(sizeof(float)*T);
    float *acc = malloc(sizeof(float)*HD);
    int rep_kv = NH/NKV;
    for(int l=id; l<NL; l+=nthr){
        const int8_t *K = Kc + (size_t)l*kv_layer_sz;
        const int8_t *V = Vc + (size_t)l*kv_layer_sz;
        for(int h=0; h<NH; h++){
            const float *q = qbuf + (size_t)l*NH*HD + (size_t)h*HD;
            int kh = h / rep_kv;
            const int8_t *Kh = K + (size_t)kh*T*HD;
            const int8_t *Vh = V + (size_t)kh*T*HD;
            float mx = -1e30f;
            for(int t=0;t<T;t++){
                const int8_t *k = Kh + (size_t)t*HD;
                float s=0; for(int d=0;d<HD;d++) s += q[d]*(float)k[d];
                s *= QSCALE; sc[t]=s; if(s>mx) mx=s;
            }
            float sum=0;
            for(int t=0;t<T;t++){ sc[t]=expf(sc[t]-mx); sum+=sc[t]; }
            float inv=1.0f/sum;
            for(int d=0;d<HD;d++) acc[d]=0;
            for(int t=0;t<T;t++){
                float p=sc[t]*inv;
                const int8_t *v = Vh + (size_t)t*HD;
                for(int d=0;d<HD;d++) acc[d] += p*(float)v[d];
            }
            sink += acc[0];
        }
    }
    free(sc); free(acc); return 0;
}

/* ---- B2. attention（INT8 のまま）---- */
static void *attn8(void *a){
    int id=(int)(long)a;
    float *sc = malloc(sizeof(float)*T);
    uint8_t *pq = malloc(T);
    int8_t *qq = malloc(HD);
    int32_t *acc = malloc(sizeof(int32_t)*HD);
    int rep_kv = NH/NKV;
    for(int d=0;d<HD;d++) qq[d]=(int8_t)(d%127-63);
    for(int l=id; l<NL; l+=nthr){
        const int8_t *K = Kc + (size_t)l*kv_layer_sz;
        const int8_t *V = Vc + (size_t)l*kv_layer_sz;
        for(int h=0; h<NH; h++){
            int kh = h / rep_kv;
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
    free(sc); free(pq); free(qq); free(acc); return 0;
}

/* ---- C. softmax だけ ---- */
static int use_fast_exp;
static inline float fexp(float x){
    x = 1.4426950408889634f*x;
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

/* ---- D. 小物 ---- */
static void *misc(void *a){
    int id=(int)(long)a;
    for(int l=id; l<NL; l+=nthr){
        float *x = xbuf + (size_t)l*HID;
        float *f = fbuf + (size_t)l*FFN;
        for(int r=0;r<NRM_H;r++){                      /* hidden 幅の RMSNorm */
            float s=0; for(int i=0;i<HID;i++) s += x[i]*x[i];
            float inv = 1.0f/sqrtf(s/HID + 1e-5f);
            for(int i=0;i<HID;i++) x[i] *= inv;
        }
        for(int r=0;r<NRM_F;r++){                      /* FFN 幅の RMSNorm（BitNet 固有）*/
            float s=0; for(int i=0;i<FFN;i++) s += f[i]*f[i];
            float inv = 1.0f/sqrtf(s/FFN + 1e-5f);
            for(int i=0;i<FFN;i++) f[i] *= inv;
        }
        for(int i=0;i<(HID+KVD)/2;i++){                /* RoPE（乗算のみ）*/
            float c=0.9998f, s=0.0175f;
            float a0=x[(2*i)%HID], a1=x[(2*i+1)%HID];
            x[(2*i)%HID]   = a0*c - a1*s;
            x[(2*i+1)%HID] = a0*s + a1*c;
        }
        for(int i=0;i<FFN;i++){                        /* SiLU + 要素積 */
            float g=f[i];
            f[i] = g/(1.0f+expf(-g)) * (g*0.5f);
        }
        for(int i=0;i<HID;i++) x[i] += 0.001f;         /* 残差 2回 */
        for(int i=0;i<HID;i++) x[i] += 0.001f;
        sink += x[0] + f[0];
    }
    return 0;
}

static double run(void *(*fn)(void*), int rep){
    pthread_t th[64];
    for(long i=0;i<nthr;i++) pthread_create(&th[i],0,fn,(void*)i);
    for(int i=0;i<nthr;i++) pthread_join(th[i],0);
    double t0=now();
    for(int r=0;r<rep;r++){
        for(long i=0;i<nthr;i++) pthread_create(&th[i],0,fn,(void*)i);
        for(int i=0;i<nthr;i++) pthread_join(th[i],0);
    }
    return (now()-t0)/rep*1000.0;
}

int main(int argc,char**argv){
    T     = (argc> 1)? atoi(argv[1]) : 512;
    int rep=(argc> 2)? atoi(argv[2]) : 20;
    HID   = (argc> 3)? atoi(argv[3]) : 576;
    NL    = (argc> 4)? atoi(argv[4]) : 30;
    NH    = (argc> 5)? atoi(argv[5]) : 9;
    NKV   = (argc> 6)? atoi(argv[6]) : 3;
    HD    = (argc> 7)? atoi(argv[7]) : 64;
    FFN   = (argc> 8)? atoi(argv[8]) : 1536;
    NRM_H = (argc> 9)? atoi(argv[9]) : 2;
    NRM_F = (argc>10)? atoi(argv[10]): 0;
    KVD = NKV*HD; QSCALE = 1.0f/sqrtf((float)HD);
    nthr = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if(nthr<1) nthr=1; if(nthr>64) nthr=64;

    kv_layer_sz = (size_t)KVD*T;
    size_t kvtot = kv_layer_sz*(size_t)NL*2;

    printf("hidden %d / %d層 / Q %d ヘッド / KV %d ヘッド / head_dim %d / FFN %d"
           " / RMSNorm %d(hidden)+%d(FFN)\n", HID, NL, NH, NKV, HD, FFN, NRM_H, NRM_F);
    printf("文脈長 T = %d / スレッド %d / 繰り返し %d\n", T, nthr, rep);
    printf("KV キャッシュ = %.2f MB（INT8・%d バイト/位置）\n\n",
           kvtot/1048576.0, NL*2*KVD);

    Kc = malloc(kv_layer_sz*NL); Vc = malloc(kv_layer_sz*NL);
    qbuf = malloc(sizeof(float)*(size_t)NL*NH*HD);
    xbuf = malloc(sizeof(float)*(size_t)NL*HID);
    fbuf = malloc(sizeof(float)*(size_t)NL*FFN);
    if(!Kc||!Vc||!qbuf||!xbuf||!fbuf){ printf("メモリが足りません\n"); return 1; }
    for(size_t i=0;i<kv_layer_sz*NL;i++){ Kc[i]=(int8_t)(i%251-125); Vc[i]=(int8_t)(i%241-120); }
    for(size_t i=0;i<(size_t)NL*NH*HD;i++) qbuf[i]=(float)((i%97)-48)*0.01f;
    for(size_t i=0;i<(size_t)NL*HID;i++)   xbuf[i]=(float)((i%89)-44)*0.01f;
    for(size_t i=0;i<(size_t)NL*FFN;i++)   fbuf[i]=(float)((i%83)-41)*0.01f;

    double a=run(kv_read,rep), b=run(attn,rep), b2=run(attn8,rep);
    use_fast_exp=0; double c=run(smax,rep);
    use_fast_exp=1; double c2=run(smax,rep);
    double d=run(misc,rep);

    double macs = (double)NL*NH*T*HD*2;
    printf("A. KV の逐次リードだけ     : %8.2f ms   (%.2f GB/s)\n", a, kvtot/(a*1e-3)/1e9);
    printf("B. attention（素朴 float） : %8.2f ms\n", b);
    printf("B2. attention（INT8 のまま）: %8.2f ms   (%.3f G MAC/s)  ★これが CPU の実力\n",
           b2, macs/(b2*1e-3)/1e9);
    printf("C. softmax だけ            : %8.2f ms  (expf) / %8.2f ms (高速近似)\n", c, c2);
    printf("D. 小物一式                : %8.2f ms\n", d);
    printf("\n1トークンの attention + 小物 = %.2f ms  （B2 + D）\n", b2+d);
    printf("  MAC 数 = %.1f M（%d層 × %d ヘッド × T %d × %d × 2）\n",
           macs/1e6, NL, NH, T, HD);
    (void)sink; return 0;
}
