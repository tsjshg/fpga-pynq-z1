/*
 * DMA を叩く部分だけを C にしたもの（段階11・ボード専用）。
 *
 * Python（numpy 経由の MMIO）だとレジスタ1回の読み書きに 6〜14 µs かかり、
 * 1回の行列積（2ポート × 書き 4 + 読み 数回 + x の書き込み + 結果の写し）で数百 µs を食う。
 * 1トークン 121 転送なので、ここが数十 ms になる。
 *
 * レジスタは PYNQ が mmap した領域をそのまま使う（Python から番地を渡す）。
 * 数値の計算は一切しないので、Mac 側の照合とは無関係。
 */
#include <stdint.h>
#include <string.h>

#define DMACR   (0x00 >> 2)
#define MM2S_SR (0x04 >> 2)
#define MM2S_TAIL (0x10 >> 2)
#define MM2S_SA (0x18 >> 2)
#define MM2S_LEN (0x28 >> 2)
#define S2MM_SR (0x34 >> 2)
#define S2MM_TAIL (0x40 >> 2)
#define S2MM_DA (0x48 >> 2)
#define S2MM_LEN (0x58 >> 2)

/* 非キャッシュ領域への書き込みを、DMA を起こす前に DDR まで届かせる */
static inline void wbarrier(void)
{
#if defined(__arm__)
    __asm__ volatile("dsb st" ::: "memory");
#else
    __sync_synchronize();
#endif
}

/*
 * 行列積1回。b は Python が組ごとに作っておく数表（uint32 × 16）:
 *   [0] ポート0 のレジスタ番地  [1] ポート1 のレジスタ番地
 *   [2] 出力先0 [3] 出力長0 [4] 入力元0 [5] 入力長0
 *   [6] 出力先1 [7] 出力長1 [8] 入力元1 [9] 入力長1
 *   [10] x の置き場0（仮想番地） [11] x の置き場1
 *   [12] 結果0（仮想番地） [13] 語数0 [14] 結果1 [15] 語数1
 * 戻り値は結果の語数。out に両ポートぶんをつなげて写す。
 */
int dma_gemv(const uint32_t *b, const int8_t *x, int nx, int32_t *out)
{
    volatile uint32_t *r0 = (volatile uint32_t *)(uintptr_t)b[0];
    volatile uint32_t *r1 = (volatile uint32_t *)(uintptr_t)b[1];
    memcpy((void *)(uintptr_t)b[10], x, nx);
    memcpy((void *)(uintptr_t)b[11], x, nx);
    wbarrier();
    r0[S2MM_DA] = b[2]; r0[S2MM_LEN] = b[3];
    r1[S2MM_DA] = b[6]; r1[S2MM_LEN] = b[7];
    r0[MM2S_SA] = b[4]; r1[MM2S_SA] = b[8];
    r0[MM2S_LEN] = b[5]; r1[MM2S_LEN] = b[9];       /* 長さを書いた瞬間に走り出す */
    while (!(r0[MM2S_SR] & 2)) ;
    while (!(r0[S2MM_SR] & 2)) ;
    while (!(r1[MM2S_SR] & 2)) ;
    while (!(r1[S2MM_SR] & 2)) ;
    memcpy(out, (const void *)(uintptr_t)b[12], b[13] * 4);
    memcpy(out + b[13], (const void *)(uintptr_t)b[14], b[15] * 4);
    return (int)(b[13] + b[15]);
}

/*
 * attention 1層。KV の置き場へ書いてから、両ポートの TAILDESC を書いて走らせる。
 *   a[0],a[1]   レジスタ番地（ポート0/1）
 *   a[2],a[3]   その層の 8ヘッドぶんの置き場の先頭（仮想番地）
 *   a[4],a[5]   MM2S の TAILDESC に書く物理番地
 *   a[6],a[7]   S2MM の TAILDESC に書く物理番地
 *   a[8],a[9]   S2MM 記述子の状態語（仮想番地）
 *   a[10],a[11] 結果（仮想番地・8×97 語ずつ）
 * slot は1ヘッドの置き場の大きさ、k0 / v0 はその中の K / V の書き込み位置。
 * 見出しは [31:0]=T, [49:32]=m（ヘッドごとの倍率）。
 */
int dma_attn(const uint32_t *a, int slot, int k0, int v0, int T, const int64_t *mm,
             const int8_t *qi, const int8_t *kq, const int8_t *vq, int32_t *out)
{
    for (int p = 0; p < 2; p++) {
        uint8_t *base = (uint8_t *)(uintptr_t)a[2 + p];
        for (int h = 0; h < 8; h++) {
            int hh = p*8 + h;
            uint8_t *s = base + h*slot;
            uint64_t hd = (uint64_t)(uint32_t)T | ((uint64_t)mm[hh] << 32);
            memcpy(s, &hd, 8);
            memcpy(s + 8, qi + hh*96, 96);
            memcpy(s + k0, kq + hh*96, 96);
            memcpy(s + v0, vq + hh*96, 96);
        }
    }
    wbarrier();
    for (int p = 0; p < 2; p++) {
        volatile uint32_t *r = (volatile uint32_t *)(uintptr_t)a[p];
        r[S2MM_TAIL] = a[6 + p];
        r[MM2S_TAIL] = a[4 + p];
    }
    for (int p = 0; p < 2; p++) {
        volatile uint32_t *st = (volatile uint32_t *)(uintptr_t)a[8 + p];
        long n = 0;
        while (!(*st & 0x80000000u))
            if (++n > 200000000L) return -1 - p;      /* 止まったら呼び出し側で状態を見る */
    }
    memcpy(out, (const void *)(uintptr_t)a[10], 8*97*4);
    memcpy(out + 8*97, (const void *)(uintptr_t)a[11], 8*97*4);
    return 0;
}
