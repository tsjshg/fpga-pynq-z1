/* DDR 実効帯域と量子化カーネルの測定（ボード非依存版）
 *   使い方: ./ddrbw2 [バッファMB]   既定 128
 *   バッファは L2 を大きく超え、かつ MemAvailable を下回るように選ぶこと
 *   （スワップに落ちると測定値が無意味になる）
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>

static size_t BUFSZ;
static uint8_t *buf;
static int nthr;

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

/* 逐次リード: 帯域の上限を見る */
static void *rd(void *a){
    size_t id=(size_t)a, chunk=BUFSZ/nthr;
    uint64_t s=0; const uint64_t *p=(const uint64_t*)(buf+id*chunk);
    for(size_t i=0;i<chunk/8;i+=8) s+=p[i]+p[i+1]+p[i+2]+p[i+3]+p[i+4]+p[i+5]+p[i+6]+p[i+7];
    *(volatile uint64_t*)buf |= (s & 1); return 0;
}
/* INT8 内積: 重み1バイトにつき1 MAC */
static void *i8(void *a){
    size_t id=(size_t)a, chunk=BUFSZ/nthr;
    const int8_t *w=(const int8_t*)(buf+id*chunk); int32_t acc=0;
    for(size_t i=0;i<chunk;i++) acc += (int32_t)w[i]*3;
    *(volatile int32_t*)buf |= (acc & 1); return 0;
}
/* 三値展開: 1バイトに5重み。テーブル引き（div/mod を使う旧版より現実的） */
static int8_t T[256][5];
static void tern_init(void){
    for(int b=0;b<256;b++){ int v=b; for(int k=0;k<5;k++){ int t=v%3; v/=3; T[b][k]=(t==0)?0:((t==1)?1:-1); } }
}
static void *tern(void *a){
    size_t id=(size_t)a, chunk=BUFSZ/nthr;
    const uint8_t *w=(const uint8_t*)(buf+id*chunk); int32_t acc=0;
    for(size_t i=0;i<chunk;i++){ const int8_t *t=T[w[i]];
        acc += t[0]*7 + t[1]*7 + t[2]*7 + t[3]*7 + t[4]*7; }
    *(volatile int32_t*)buf |= (acc & 1); return 0;
}

static void run(const char *name,void*(*fn)(void*),int n,double bpw){
    nthr=n; pthread_t th[64]; double t0=now();
    for(int i=0;i<n;i++) pthread_create(&th[i],0,fn,(void*)(size_t)i);
    for(int i=0;i<n;i++) pthread_join(th[i],0);
    double dt=now()-t0, gbs=(double)BUFSZ/dt/1e9;
    printf("  %-16s %2dコア  %6.2f GB/s", name, n, gbs);
    if(bpw>0) printf("   → %6.2f G重み/s", gbs/bpw);
    printf("\n"); fflush(stdout);
}

int main(int argc,char**argv){
    int mb = (argc>1)? atoi(argv[1]) : 128;
    if(mb<8) mb=8;
    BUFSZ=(size_t)mb*1024*1024;
    int ncpu=(int)sysconf(_SC_NPROCESSORS_ONLN); if(ncpu<1) ncpu=1; if(ncpu>64) ncpu=64;

    buf=aligned_alloc(4096,BUFSZ);
    if(!buf){ perror("alloc"); return 1; }
    memset(buf,0x11,BUFSZ);
    tern_init();

    printf("バッファ %d MB / オンラインCPU %d\n\n", mb, ncpu);
    int counts[8]; int nc=0;
    for(int n=1;n<=ncpu;n*=2) counts[nc++]=n;
    if(counts[nc-1]!=ncpu) counts[nc++]=ncpu;

    printf("[1] 逐次リード = DDR 実効帯域の上限\n");
    for(int i=0;i<nc;i++) run("read",rd,counts[i],0);
    printf("\n[2] INT8 GEMV 相当 (1バイト=1重み)\n");
    for(int i=0;i<nc;i++) run("int8 dot",i8,counts[i],1.0);
    printf("\n[3] 三値展開・テーブル引き (0.2バイト/重み)\n");
    for(int i=0;i<nc;i++) run("ternary",tern,counts[i],0.2);

    free(buf); return 0;
}
