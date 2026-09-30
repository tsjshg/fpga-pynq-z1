/* DDR 実効帯域の測定: 逐次リード / コピー / ターナリ展開の模擬
   バッファは L2 (2MB) を大きく超える 256MB を使う */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>

#define BUFSZ (256UL*1024*1024)

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

static uint8_t *buf; static int nthr;

/* 単純な逐次リード（帯域の上限を見る） */
static void *rd(void *a){
    size_t id=(size_t)a, chunk=BUFSZ/nthr;
    volatile uint64_t s=0; const uint64_t *p=(const uint64_t*)(buf+id*chunk);
    for(size_t i=0;i<chunk/8;i+=8){ s+=p[i]+p[i+1]+p[i+2]+p[i+3]+p[i+4]+p[i+5]+p[i+6]+p[i+7]; }
    (void)s; return 0;
}
/* INT8 の内積: 重み1バイトにつき 1 MAC */
static void *i8(void *a){
    size_t id=(size_t)a, chunk=BUFSZ/nthr;
    const int8_t *w=(const int8_t*)(buf+id*chunk); int32_t acc=0;
    for(size_t i=0;i<chunk;i++) acc += (int32_t)w[i] * 3;
    ((volatile int32_t*)buf)[0]=acc; return 0;
}
/* ターナリ(1.58bit) の展開を模擬: 1バイトに5個詰め、取り出して加減算 */
static void *tern(void *a){
    size_t id=(size_t)a, chunk=BUFSZ/nthr;
    const uint8_t *w=(const uint8_t*)(buf+id*chunk); int32_t acc=0;
    for(size_t i=0;i<chunk;i++){
        uint8_t b=w[i];
        for(int k=0;k<5;k++){ int t=b%3; b/=3; acc += (t==0)?0:((t==1)?7:-7); }
    }
    ((volatile int32_t*)buf)[0]=acc; return 0;
}

static void run(const char *name,void*(*fn)(void*),int n,double bytes_per_weight){
    nthr=n; pthread_t th[8]; double t0=now();
    for(int i=0;i<n;i++) pthread_create(&th[i],0,fn,(void*)(size_t)i);
    for(int i=0;i<n;i++) pthread_join(th[i],0);
    double dt=now()-t0, gbs=BUFSZ/dt/1e9;
    printf("  %-16s %dコア  %6.2f GB/s", name, n, gbs);
    if(bytes_per_weight>0) printf("   → %6.2f G重み/s", gbs/bytes_per_weight);
    printf("\n");
}

int main(void){
    buf=aligned_alloc(4096,BUFSZ); if(!buf){perror("alloc");return 1;}
    memset(buf,0x11,BUFSZ);
    printf("バッファ %lu MB (L2 2MB を大きく超える)\n\n", BUFSZ/1048576);
    printf("[1] 純粋な逐次リード = DDR 実効帯域の上限\n");
    run("read",rd,1,0); run("read",rd,2,0); run("read",rd,4,0);
    printf("\n[2] INT8 GEMV 相当 (重み1バイト=1MAC)\n");
    run("int8 dot",i8,1,1.0); run("int8 dot",i8,2,1.0); run("int8 dot",i8,4,1.0);
    printf("\n[3] ターナリ展開 (1バイトに5重み = 0.2バイト/重み)\n");
    run("ternary unpack",tern,1,0.2); run("ternary unpack",tern,2,0.2); run("ternary unpack",tern,4,0.2);
    free(buf); return 0;
}
